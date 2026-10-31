package com.jasonrhan.gomoku.ml.training;

import com.jasonrhan.gomoku.ml.data.CheckpointStore;
import com.jasonrhan.gomoku.ml.data.DataDownloadServer;
import com.jasonrhan.gomoku.ml.data.GameArchiveStore;
import com.jasonrhan.gomoku.ml.data.MetricsWriter;
import com.jasonrhan.gomoku.ml.data.ReplayBufferStore;
import com.jasonrhan.gomoku.ml.domain.BoardState;
import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.model.AdamOptimizer;
import com.jasonrhan.gomoku.ml.model.BoardEncoder;
import com.jasonrhan.gomoku.ml.model.ModelMetadata;
import com.jasonrhan.gomoku.ml.model.ModelSerializer;
import com.jasonrhan.gomoku.ml.model.PolicyValueNetwork;
import com.jasonrhan.gomoku.ml.rules.ForbiddenMoveService;
import com.jasonrhan.gomoku.ml.selfplay.EvaluationRunner;
import com.jasonrhan.gomoku.ml.selfplay.SelfPlayRunner;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

public final class TrainingLoop {
    private static final DateTimeFormatter RUN_ID = DateTimeFormatter
            .ofPattern("yyyyMMdd-HHmmss")
            .withZone(ZoneOffset.UTC);

    private final TrainingConfig config;
    private final AtomicBoolean stopRequested = new AtomicBoolean();

    public TrainingLoop(TrainingConfig config) {
        this.config = config.normalized();
    }

    public Path run() throws Exception {
        Path runDirectory = resolveRunDirectory();
        Files.createDirectories(runDirectory);
        writeConfig(runDirectory);

        PolicyValueNetwork network;
        AdamOptimizer optimizer;
        int iteration;
        long gamesCompleted;
        CheckpointStore checkpointStore = new CheckpointStore();
        Optional<CheckpointStore.LoadedCheckpoint> checkpoint =
                config.resumeDirectory() == null
                        ? Optional.empty()
                        : checkpointStore.loadLatest(runDirectory);
        if (checkpoint.isPresent()) {
            CheckpointStore.LoadedCheckpoint loaded = checkpoint.orElseThrow();
            network = loaded.network();
            optimizer = loaded.optimizer();
            iteration = loaded.iteration();
            gamesCompleted = loaded.games();
        } else {
            network = new PolicyValueNetwork(config.seed());
            optimizer = new AdamOptimizer();
            iteration = 0;
            gamesCompleted = 0;
        }

        ForbiddenMoveService forbiddenMoveService = new ForbiddenMoveService();
        Random random = new Random(config.seed() + gamesCompleted);
        SelfPlayRunner selfPlay = new SelfPlayRunner(
                forbiddenMoveService,
                random,
                config.simulations(),
                config.cPuct(),
                config.rootNoise(),
                config.candidateRadius(),
                config.candidateLimit()
        );
        Path replayDirectory = runDirectory.resolve("replay");
        try (
                ReplayBufferStore replay = new ReplayBufferStore(
                        replayDirectory, config.replayCapacity()
                );
                GameArchiveStore archive = new GameArchiveStore(
                        runDirectory.resolve("games.jsonl")
                );
                MetricsWriter metrics = new MetricsWriter(
                        runDirectory.resolve("metrics.csv")
                )
        ) {
            DataDownloadServer downloadServer = null;
            if (config.serveData()) {
                downloadServer = new DataDownloadServer(
                        archive, config.dataToken()
                );
                int port = downloadServer.start(
                        config.dataHost(),
                        config.dataPortStart(),
                        config.dataPortEnd()
                );
                System.out.printf(
                        "Data server: http://%s:%d%n",
                        config.dataHost(),
                        port
                );
                if (!config.dataToken().isBlank()) {
                    System.out.println("Data server authentication: bearer token enabled");
                }
            }

            Runtime.getRuntime().addShutdownHook(new Thread(
                    () -> stopRequested.set(true),
                    "training-shutdown-hook"
            ));
            long startNanos = System.nanoTime();
            try {
                while (shouldContinue(gamesCompleted, startNanos)) {
                    int gamesThisIteration = gamesToPlay(gamesCompleted);
                    int blackWins = 0;
                    int whiteWins = 0;
                    int draws = 0;
                    int moveTotal = 0;
                    for (int game = 0; game < gamesThisIteration; game++) {
                        SelfPlayRunner.GameRecord record = selfPlay.playOneGame(network);
                        for (var example : record.examples()) {
                            replay.append(example);
                        }
                        archive.append(record);
                        moveTotal += record.moves();
                        if (record.result() == GameResult.BLACK_WIN) {
                            blackWins++;
                        } else if (record.result() == GameResult.WHITE_WIN) {
                            whiteWins++;
                        } else {
                            draws++;
                        }
                        gamesCompleted++;
                        if (!shouldContinue(gamesCompleted, startNanos)) {
                            break;
                        }
                    }

                    PolicyValueNetwork.BatchMetrics trainingMetrics =
                            train(network, optimizer, replay, random);
                    iteration++;
                    checkpointStore.save(
                            runDirectory,
                            iteration,
                            gamesCompleted,
                            network,
                            optimizer
                    );
                    writeManifest(runDirectory, iteration, gamesCompleted, network);
                    EvaluationRunner.EvaluationSummary evaluation =
                            config.evaluationGames() == 0
                                    ? new EvaluationRunner.EvaluationSummary(0, 0, 0, 0)
                                    : new EvaluationRunner(
                                            forbiddenMoveService,
                                            random,
                                            Math.max(4, config.simulations() / 2)
                                    ).playAgainstRandom(
                                            network,
                                            config.evaluationGames()
                                    );
                    metrics.write(
                            iteration,
                            gamesCompleted,
                            blackWins,
                            whiteWins,
                            draws,
                            gamesThisIteration == 0
                                    ? 0.0
                                    : (double) moveTotal / gamesThisIteration,
                            trainingMetrics.policyLoss(),
                            trainingMetrics.valueLoss(),
                            trainingMetrics.totalLoss(),
                            replay.size(),
                            "checkpoint-%06d".formatted(iteration),
                            elapsedSeconds(startNanos)
                    );
                    writeSummary(
                            runDirectory,
                            gamesCompleted,
                            blackWins,
                            whiteWins,
                            draws,
                            evaluation,
                            elapsedSeconds(startNanos)
                    );
                    System.out.printf(
                            "iteration=%d games=%d selfplay=%d/%d/%d loss=%.5f value=%.5f replay=%d eval=%d/%d/%d%n",
                            iteration,
                            gamesCompleted,
                            blackWins,
                            whiteWins,
                            draws,
                            trainingMetrics.policyLoss(),
                            trainingMetrics.valueLoss(),
                            replay.size(),
                            evaluation.wins(),
                            evaluation.losses(),
                            evaluation.draws()
                    );
                    if (stopRequested.get()) {
                        break;
                    }
                }
            } finally {
                if (downloadServer != null) {
                    downloadServer.close();
                }
            }
        }
        return runDirectory;
    }

    private PolicyValueNetwork.BatchMetrics train(
            PolicyValueNetwork network,
            AdamOptimizer optimizer,
            ReplayBufferStore replay,
            Random random
    ) {
        PolicyValueNetwork.BatchMetrics latest =
                new PolicyValueNetwork.BatchMetrics(0.0, 0.0, 0.0);
        if (replay.size() == 0) {
            return latest;
        }
        int steps = Math.max(1, Math.min(
                64,
                (replay.size() + config.batchSize() - 1) / config.batchSize()
        ));
        for (int epoch = 0; epoch < config.trainEpochs(); epoch++) {
            for (int step = 0; step < steps; step++) {
                List<com.jasonrhan.gomoku.ml.selfplay.TrainingExample> batch =
                        replay.sample(config.batchSize(), random);
                latest = network.trainBatch(
                        batch,
                        optimizer,
                        config.learningRate(),
                        1.0e-5,
                        0.5
                );
            }
        }
        return latest;
    }

    private boolean shouldContinue(long gamesCompleted, long startNanos) {
        if (stopRequested.get()) {
            return false;
        }
        if (config.games() > 0 && gamesCompleted >= config.games()) {
            return false;
        }
        return config.hours() <= 0.0
                || elapsedSeconds(startNanos) < config.hours() * 3600.0;
    }

    private int gamesToPlay(long gamesCompleted) {
        if (config.games() <= 0) {
            return config.gamesPerIteration();
        }
        return (int) Math.min(
                config.gamesPerIteration(),
                Math.max(1, config.games() - gamesCompleted)
        );
    }

    private Path resolveRunDirectory() throws IOException {
        if (config.resumeDirectory() != null) {
            return config.resumeDirectory();
        }
        return config.outputDirectory().resolve(
                "run-" + RUN_ID.format(Instant.now())
        );
    }

    private void writeConfig(Path runDirectory) throws IOException {
        String json = """
                {
                  "games": %d,
                  "hours": %.4f,
                  "seed": %d,
                  "simulations": %d,
                  "games_per_iteration": %d,
                  "train_epochs": %d,
                  "batch_size": %d,
                  "learning_rate": %.8f,
                  "replay_capacity": %d,
                  "checkpoint_every_games": %d,
                  "data_port_start": %d,
                  "data_port_end": %d
                }
                """.formatted(
                config.games(),
                config.hours(),
                config.seed(),
                config.simulations(),
                config.gamesPerIteration(),
                config.trainEpochs(),
                config.batchSize(),
                config.learningRate(),
                config.replayCapacity(),
                config.checkpointEveryGames(),
                config.dataPortStart(),
                config.dataPortEnd()
        );
        Files.writeString(
                runDirectory.resolve("config.json"),
                json,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    private void writeManifest(
            Path runDirectory,
            int iteration,
            long gamesCompleted,
            PolicyValueNetwork network
    ) throws IOException {
        Path model = runDirectory.resolve("checkpoints")
                .resolve("checkpoint-%06d".formatted(iteration))
                .resolve("model.gmk");
        if (!Files.exists(model)) {
            return;
        }
        Path publishedModel = runDirectory.resolve("gomoku-model.gmk");
        Path temporaryPublishedModel = runDirectory.resolve("gomoku-model.gmk.tmp");
        Files.copy(model, temporaryPublishedModel, StandardCopyOption.REPLACE_EXISTING);
        Files.move(
                temporaryPublishedModel,
                publishedModel,
                StandardCopyOption.REPLACE_EXISTING
        );
        String sha256 = sha256(model);
        ModelMetadata metadata = new ModelMetadata(
                1,
                "checkpoint-%06d".formatted(iteration),
                BoardState.BOARD_SIZE,
                BoardState.INPUT_PLANES,
                BoardState.POLICY_SIZE,
                "gomoku-model.gmk",
                sha256,
                gamesCompleted,
                config.simulations(),
                Instant.now(),
                "mlp-mcts-self-play"
        );
        Files.writeString(
                runDirectory.resolve("model-manifest.json"),
                metadata.toJson(),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    private void writeSummary(
            Path runDirectory,
            long gamesCompleted,
            int blackWins,
            int whiteWins,
            int draws,
            EvaluationRunner.EvaluationSummary evaluation,
            double elapsedSeconds
    ) throws IOException {
        String summary = """
                {
                  "completed_games": %d,
                  "black_wins_last_iteration": %d,
                  "white_wins_last_iteration": %d,
                  "draws_last_iteration": %d,
                  "evaluation_games": %d,
                  "evaluation_wins": %d,
                  "evaluation_losses": %d,
                  "evaluation_draws": %d,
                  "elapsed_seconds": %.2f
                }
                """.formatted(
                gamesCompleted,
                blackWins,
                whiteWins,
                draws,
                evaluation.games(),
                evaluation.wins(),
                evaluation.losses(),
                evaluation.draws(),
                elapsedSeconds
        );
        Files.writeString(
                runDirectory.resolve("summary.json"),
                summary,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    private String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(path)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    digest.update(buffer, 0, read);
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException exception) {
            throw new IOException("SHA-256 is unavailable", exception);
        }
    }

    private double elapsedSeconds(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000_000.0;
    }
}
