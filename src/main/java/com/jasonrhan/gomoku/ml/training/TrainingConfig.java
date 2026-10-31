package com.jasonrhan.gomoku.ml.training;

import java.nio.file.Path;

public record TrainingConfig(
        long games,
        double hours,
        long seed,
        Path outputDirectory,
        Path resumeDirectory,
        int simulations,
        int gamesPerIteration,
        int trainEpochs,
        int batchSize,
        double learningRate,
        int replayCapacity,
        int checkpointEveryGames,
        int evaluationGames,
        double cPuct,
        double rootNoise,
        int candidateRadius,
        int candidateLimit,
        boolean serveData,
        String dataHost,
        int dataPortStart,
        int dataPortEnd,
        String dataToken
) {
    public TrainingConfig normalized() {
        return new TrainingConfig(
                Math.max(0L, games),
                Math.max(0.0, hours),
                seed,
                outputDirectory.toAbsolutePath().normalize(),
                resumeDirectory == null
                        ? null
                        : resumeDirectory.toAbsolutePath().normalize(),
                Math.max(1, simulations),
                Math.max(1, gamesPerIteration),
                Math.max(1, trainEpochs),
                Math.max(1, batchSize),
                Math.max(1.0e-6, learningRate),
                Math.max(100, replayCapacity),
                Math.max(1, checkpointEveryGames),
                Math.max(0, evaluationGames),
                Math.max(0.1, cPuct),
                Math.max(0.0, Math.min(1.0, rootNoise)),
                Math.max(1, candidateRadius),
                Math.max(1, candidateLimit),
                serveData,
                dataHost == null || dataHost.isBlank() ? "0.0.0.0" : dataHost,
                Math.max(1, dataPortStart),
                Math.max(dataPortStart, dataPortEnd),
                dataToken == null ? "" : dataToken
        );
    }
}
