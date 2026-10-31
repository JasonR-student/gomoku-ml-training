package com.jasonrhan.gomoku.ml.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

public final class MetricsWriter implements AutoCloseable {
    private static final String HEADER = """
            iteration,games,black_wins,white_wins,draws,average_game_length,policy_loss,value_loss,total_loss,replay_size,model_version,elapsed_seconds
            """;

    private final Path path;

    public MetricsWriter(Path path) throws IOException {
        this.path = path;
        Files.createDirectories(path.toAbsolutePath().normalize().getParent());
        if (!Files.exists(path) || Files.size(path) == 0) {
            Files.writeString(
                    path,
                    HEADER,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND
            );
        }
    }

    public synchronized void write(
            int iteration,
            long games,
            int blackWins,
            int whiteWins,
            int draws,
            double averageGameLength,
            double policyLoss,
            double valueLoss,
            double totalLoss,
            int replaySize,
            String modelVersion,
            double elapsedSeconds
    ) throws IOException {
        String row = String.format(
                Locale.ROOT,
                "%d,%d,%d,%d,%d,%.4f,%.8f,%.8f,%.8f,%d,%s,%.2f%n",
                iteration,
                games,
                blackWins,
                whiteWins,
                draws,
                averageGameLength,
                policyLoss,
                valueLoss,
                totalLoss,
                replaySize,
                modelVersion,
                elapsedSeconds
        );
        Files.writeString(path, row, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    @Override
    public void close() {
    }
}
