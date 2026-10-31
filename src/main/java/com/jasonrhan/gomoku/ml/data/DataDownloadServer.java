package com.jasonrhan.gomoku.ml.data;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.BindException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;

public final class DataDownloadServer implements AutoCloseable {
    private final GameArchiveStore archive;
    private final String token;
    private HttpServer server;
    private int port;

    public DataDownloadServer(GameArchiveStore archive, String token) {
        this.archive = archive;
        this.token = token == null ? "" : token;
    }

    public int start(String host, int firstPort, int lastPort) throws IOException {
        IOException lastFailure = null;
        for (int candidate = firstPort; candidate <= lastPort; candidate++) {
            try {
                server = HttpServer.create(
                        new InetSocketAddress(host, candidate),
                        0
                );
                port = candidate;
                registerContexts();
                server.setExecutor(Executors.newFixedThreadPool(2));
                server.start();
                return port;
            } catch (BindException failure) {
                lastFailure = failure;
            }
        }
        throw new IOException(
                "No available data port in range "
                        + firstPort + "-" + lastPort,
                lastFailure
        );
    }

    public int port() {
        return port;
    }

    @Override
    public void close() {
        if (server != null) {
            server.stop(0);
            server = null;
        }
    }

    private void registerContexts() {
        server.createContext("/api/status", this::handleStatus);
        server.createContext("/api/download", this::handleDownload);
    }

    private void handleStatus(HttpExchange exchange) throws IOException {
        if (!authorized(exchange)) {
            sendText(exchange, 401, "{\"error\":\"unauthorized\"}");
            return;
        }
        String body = """
                {
                  "completed_games": %d,
                  "downloadable_games": %d,
                  "batch_size": 100,
                  "download_format": "jsonl.gz"
                }
                """.formatted(
                archive.completedGames(),
                archive.downloadableGames()
        );
        sendText(exchange, 200, body);
    }

    private void handleDownload(HttpExchange exchange) throws IOException {
        if (!authorized(exchange)) {
            sendText(exchange, 401, "{\"error\":\"unauthorized\"}");
            return;
        }
        Map<String, String> query = parseQuery(exchange.getRequestURI());
        long from = parseLong(query.get("from"), 0L);
        long requestedCount = parseLong(
                query.get("count"),
                archive.downloadableGames() - from
        );
        long downloadable = archive.downloadableGames();
        if (from < 0 || from % 100 != 0 || requestedCount < 0) {
            sendText(exchange, 400, "{\"error\":\"from and count must be valid\"}");
            return;
        }
        long availableCount = Math.max(
                0L,
                Math.min(requestedCount, downloadable - from)
        );
        long count = (availableCount / 100) * 100;
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/x-ndjson");
        headers.set("Content-Encoding", "gzip");
        headers.set("X-Completed-Games", Long.toString(archive.completedGames()));
        headers.set("X-Downloadable-Games", Long.toString(downloadable));
        headers.set("X-Game-From", Long.toString(from));
        headers.set("X-Game-Count", Long.toString(count));
        exchange.sendResponseHeaders(200, 0);
        try (OutputStream output = exchange.getResponseBody()) {
            archive.writeRange(from, count, output);
        }
    }

    private boolean authorized(HttpExchange exchange) {
        if (token.isBlank()) {
            return true;
        }
        String authorization = exchange.getRequestHeaders().getFirst("Authorization");
        return authorization != null
                && authorization.equals("Bearer " + token);
    }

    private void sendText(HttpExchange exchange, int status, String body)
            throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(bytes);
        }
    }

    private Map<String, String> parseQuery(URI uri) {
        Map<String, String> values = new HashMap<>();
        String query = uri.getRawQuery();
        if (query == null || query.isBlank()) {
            return values;
        }
        for (String part : query.split("&")) {
            String[] pair = part.split("=", 2);
            if (pair.length == 2) {
                values.put(pair[0], pair[1]);
            }
        }
        return values;
    }

    private long parseLong(String value, long fallback) {
        try {
            return value == null ? fallback : Long.parseLong(value);
        } catch (NumberFormatException exception) {
            return fallback;
        }
    }
}
