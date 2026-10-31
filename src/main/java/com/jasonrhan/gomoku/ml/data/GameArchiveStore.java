package com.jasonrhan.gomoku.ml.data;

import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.selfplay.SelfPlayRunner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.function.Consumer;
import java.util.zip.GZIPOutputStream;

public final class GameArchiveStore implements AutoCloseable {
    private final Path path;
    private final BufferedWriter writer;
    private long completedGames;

    public GameArchiveStore(Path path) throws IOException {
        this.path = path.toAbsolutePath().normalize();
        Files.createDirectories(this.path.getParent());
        this.completedGames = countExistingGames();
        this.writer = Files.newBufferedWriter(
                this.path,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
        );
    }

    public synchronized long append(SelfPlayRunner.GameRecord record) throws IOException {
        long gameId = completedGames++;
        writer.write(toJson(gameId, record));
        writer.newLine();
        writer.flush();
        return gameId;
    }

    public synchronized long completedGames() {
        return completedGames;
    }

    public Path path() {
        return path;
    }

    public synchronized void writeRange(
            long from,
            long count,
            OutputStream response
    ) throws IOException {
        long normalizedFrom = Math.max(0, from);
        long available = downloadableGames();
        long normalizedCount = Math.max(0, Math.min(count, available - normalizedFrom));
        try (
                BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
                GZIPOutputStream gzip = new GZIPOutputStream(response);
                BufferedWriter output = new BufferedWriter(
                        new OutputStreamWriter(gzip, StandardCharsets.UTF_8)
                )
        ) {
            long index = 0;
            long written = 0;
            String line;
            while ((line = reader.readLine()) != null && written < normalizedCount) {
                if (index++ < normalizedFrom) {
                    continue;
                }
                output.write(line);
                output.newLine();
                written++;
            }
            output.flush();
        }
    }

    public long downloadableGames() {
        return (completedGames / 100) * 100;
    }

    public void forEachLine(Consumer<String> consumer) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                consumer.accept(line);
            }
        }
    }

    @Override
    public synchronized void close() throws IOException {
        writer.close();
    }

    private long countExistingGames() throws IOException {
        if (!Files.exists(path)) {
            return 0L;
        }
        try (var lines = Files.lines(path, StandardCharsets.UTF_8)) {
            return lines.count();
        }
    }

    private String toJson(long gameId, SelfPlayRunner.GameRecord record) {
        StringBuilder json = new StringBuilder();
        json.append("{\"game_id\":").append(gameId)
                .append(",\"result\":\"").append(record.result()).append("\"")
                .append(",\"moves\":[");
        List<Move> moves = record.moveList();
        for (int i = 0; i < moves.size(); i++) {
            if (i > 0) {
                json.append(',');
            }
            Move move = moves.get(i);
            json.append("{\"x\":").append(move.x())
                    .append(",\"y\":").append(move.y())
                    .append(",\"color\":\"").append(move.color()).append("\"}");
        }
        return json.append("]}").toString();
    }
}
