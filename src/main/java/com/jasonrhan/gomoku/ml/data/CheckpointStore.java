package com.jasonrhan.gomoku.ml.data;

import com.jasonrhan.gomoku.ml.model.AdamOptimizer;
import com.jasonrhan.gomoku.ml.model.ModelSerializer;
import com.jasonrhan.gomoku.ml.model.PolicyValueNetwork;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.Optional;

public final class CheckpointStore {
    public record LoadedCheckpoint(
            PolicyValueNetwork network,
            AdamOptimizer optimizer,
            int iteration,
            long games
    ) {
    }

    public void save(
            Path runDirectory,
            int iteration,
            long games,
            PolicyValueNetwork network,
            AdamOptimizer optimizer
    ) throws IOException {
        Path checkpoints = runDirectory.resolve("checkpoints");
        Files.createDirectories(checkpoints);
        Path directory = checkpoints.resolve("checkpoint-%06d".formatted(iteration));
        Files.createDirectories(directory);
        Path temporaryModel = directory.resolve("model.gmk.tmp");
        ModelSerializer.save(temporaryModel, network, optimizer);
        Files.move(
                temporaryModel,
                directory.resolve("model.gmk"),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE
        );
        String checkpoint = """
                {
                  "iteration": %d,
                  "games": %d,
                  "model": "%s"
                }
                """.formatted(
                iteration,
                games,
                directory.resolve("model.gmk").toAbsolutePath()
        );
        Files.writeString(directory.resolve("checkpoint.json"), checkpoint);
        Files.writeString(
                checkpoints.resolve("latest.json"),
                checkpoint,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    public Optional<LoadedCheckpoint> loadLatest(Path runDirectory) throws IOException {
        Path checkpoints = runDirectory.resolve("checkpoints");
        if (!Files.isDirectory(checkpoints)) {
            return Optional.empty();
        }
        Optional<Path> latest;
        try (var paths = Files.list(checkpoints)) {
            latest = paths
                    .filter(Files::isDirectory)
                    .filter(path -> Files.exists(path.resolve("model.gmk")))
                    .max(Comparator.comparing(Path::getFileName));
        }
        if (latest.isEmpty()) {
            return Optional.empty();
        }
        Path directory = latest.get();
        PolicyValueNetwork.LoadedNetwork loaded = ModelSerializer.load(
                directory.resolve("model.gmk")
        );
        String metadata = Files.readString(directory.resolve("checkpoint.json"));
        int iteration = parseInt(metadata, "iteration", 0);
        long games = parseLong(metadata, "games", 0L);
        return Optional.of(new LoadedCheckpoint(
                loaded.network(),
                loaded.optimizer(),
                iteration,
                games
        ));
    }

    private int parseInt(String json, String key, int fallback) {
        return (int) parseLong(json, key, fallback);
    }

    private long parseLong(String json, String key, long fallback) {
        String marker = "\"" + key + "\"";
        int start = json.indexOf(marker);
        if (start < 0) {
            return fallback;
        }
        int colon = json.indexOf(':', start);
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
            return fallback;
        }
    }
}
