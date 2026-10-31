package com.jasonrhan.gomoku.ml.data;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public final class DataDownloadClient {
    private DataDownloadClient() {
    }

    public static void download(
            String baseUrl,
            Path outputDirectory,
            String token
    ) throws IOException, InterruptedException {
        Files.createDirectories(outputDirectory);
        HttpClient client = HttpClient.newHttpClient();
        long completed = readLong(
                request(client, baseUrl + "/api/status", token),
                "completed_games"
        );
        long downloadable = (completed / 100) * 100;
        Path statePath = outputDirectory.resolve(".downloaded-games");
        long downloaded = readState(statePath);
        downloaded = Math.min(downloadable, (downloaded / 100) * 100);
        if (downloaded >= downloadable) {
            System.out.printf(
                    "No complete 100-game batch available. completed=%d downloaded=%d%n",
                    completed,
                    downloaded
            );
            return;
        }

        for (long from = downloaded; from < downloadable; from += 100) {
            String url = baseUrl + "/api/download?from=" + from + "&count=100";
            HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url));
            if (token != null && !token.isBlank()) {
                builder.header("Authorization", "Bearer " + token);
            }
            HttpResponse<byte[]> response = client.send(
                    builder.GET().build(),
                    HttpResponse.BodyHandlers.ofByteArray()
            );
            if (response.statusCode() != 200) {
                throw new IOException(
                        "Download failed with HTTP " + response.statusCode()
                );
            }
            Path target = outputDirectory.resolve(
                    "games-%07d-%07d.jsonl.gz".formatted(from, from + 100)
            );
            Files.write(
                    target,
                    response.body(),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING
            );
            Files.writeString(
                    statePath,
                    Long.toString(from + 100),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING
            );
            System.out.printf(
                    "Downloaded games %d-%d / %d%n",
                    from + 1,
                    from + 100,
                    downloadable
            );
        }
    }

    private static String request(
            HttpClient client,
            String url,
            String token
    ) throws IOException, InterruptedException {
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url));
        if (token != null && !token.isBlank()) {
            builder.header("Authorization", "Bearer " + token);
        }
        HttpResponse<String> response = client.send(
                builder.GET().build(),
                HttpResponse.BodyHandlers.ofString()
        );
        if (response.statusCode() != 200) {
            throw new IOException("Status request failed: " + response.statusCode());
        }
        return response.body();
    }

    private static long readState(Path statePath) {
        try {
            return Long.parseLong(Files.readString(statePath).trim());
        } catch (Exception ignored) {
            return 0L;
        }
    }

    private static long readLong(String json, String key) throws IOException {
        String marker = "\"" + key + "\"";
        int start = json.indexOf(marker);
        int colon = start < 0 ? -1 : json.indexOf(':', start);
        if (colon < 0) {
            throw new IOException("Missing status field: " + key);
        }
        int end = colon + 1;
        while (end < json.length() && Character.isWhitespace(json.charAt(end))) {
            end++;
        }
        int valueEnd = end;
        while (valueEnd < json.length() && Character.isDigit(json.charAt(valueEnd))) {
            valueEnd++;
        }
        try {
            return Long.parseLong(json.substring(end, valueEnd));
        } catch (RuntimeException exception) {
            throw new IOException("Invalid status field: " + key, exception);
        }
    }
}
