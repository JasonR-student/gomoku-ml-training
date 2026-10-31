package com.jasonrhan.gomoku.ml.cli;

import com.jasonrhan.gomoku.ml.data.DataDownloadClient;
import com.jasonrhan.gomoku.ml.data.DataDownloadServer;
import com.jasonrhan.gomoku.ml.data.GameArchiveStore;
import com.jasonrhan.gomoku.ml.training.TrainingConfig;
import com.jasonrhan.gomoku.ml.training.TrainingLoop;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

public final class TrainingMain {
    private TrainingMain() {
    }

    public static void main(String[] args) {
        try {
            if (args.length == 0 || "--help".equals(args[0])) {
                printUsage();
                return;
            }
            String command = args[0];
            Map<String, String> options = parseOptions(args);
            switch (command) {
                case "train" -> runTraining(options);
                case "download-data" -> runDownload(options);
                case "serve-data" -> runServer(options);
                default -> throw new IllegalArgumentException(
                        "Unknown command: " + command
                );
            }
        } catch (Exception exception) {
            System.err.println("Command failed: " + exception.getMessage());
            exception.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static void runTraining(Map<String, String> options) throws Exception {
        TrainingConfig config = new TrainingConfig(
                longOption(options, "games", 0L),
                doubleOption(options, "hours", 0.0),
                longOption(options, "seed", 20261031L),
                Path.of(options.getOrDefault("output", "./runs")),
                options.containsKey("resume")
                        ? Path.of(options.get("resume"))
                        : null,
                intOption(options, "simulations", 64),
                intOption(options, "games-per-iteration", 4),
                intOption(options, "train-epochs", 1),
                intOption(options, "batch-size", 64),
                doubleOption(options, "learning-rate", 0.001),
                intOption(options, "replay-capacity", 50000),
                intOption(options, "checkpoint-every", 100),
                intOption(options, "evaluation-games", 4),
                doubleOption(options, "c-puct", 1.25),
                doubleOption(options, "root-noise", 0.25),
                intOption(options, "candidate-radius", 2),
                intOption(options, "candidate-limit", 64),
                Boolean.parseBoolean(options.getOrDefault("serve-data", "false")),
                options.getOrDefault("data-host", "0.0.0.0"),
                intOption(options, "data-port", 5000),
                intOption(options, "data-port-end", 5005),
                options.getOrDefault("data-token", "")
        );
        Path runDirectory = new TrainingLoop(config).run();
        System.out.println("Training run: " + runDirectory.toAbsolutePath());
    }

    private static void runDownload(Map<String, String> options)
            throws Exception {
        String url = required(options, "url");
        Path output = Path.of(options.getOrDefault("output", "./downloads"));
        DataDownloadClient.download(
                url.replaceAll("/+$", ""),
                output,
                options.getOrDefault("token", "")
        );
    }

    private static void runServer(Map<String, String> options) throws Exception {
        Path runDirectory = Path.of(required(options, "run"));
        try (GameArchiveStore archive = new GameArchiveStore(
                runDirectory.resolve("games.jsonl")
        )) {
            DataDownloadServer server = new DataDownloadServer(
                    archive,
                    options.getOrDefault("data-token", "")
            );
            int port = server.start(
                    options.getOrDefault("data-host", "0.0.0.0"),
                    intOption(options, "data-port", 5000),
                    intOption(options, "data-port-end", 5005)
            );
            System.out.printf(
                    "Data server: http://%s:%d%n",
                    options.getOrDefault("data-host", "0.0.0.0"),
                    port
            );
            Runtime.getRuntime().addShutdownHook(new Thread(
                    server::close,
                    "data-server-shutdown-hook"
            ));
            new CountDownLatch(1).await();
        }
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            String argument = args[i];
            if (!argument.startsWith("--")) {
                continue;
            }
            String key = argument.substring(2);
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                options.put(key, args[++i]);
            } else {
                options.put(key, "true");
            }
        }
        return options;
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing --" + key);
        }
        return value;
    }

    private static int intOption(Map<String, String> options, String key, int fallback) {
        return options.containsKey(key)
                ? Integer.parseInt(options.get(key))
                : fallback;
    }

    private static long longOption(Map<String, String> options, String key, long fallback) {
        return options.containsKey(key)
                ? Long.parseLong(options.get(key))
                : fallback;
    }

    private static double doubleOption(
            Map<String, String> options,
            String key,
            double fallback
    ) {
        return options.containsKey(key)
                ? Double.parseDouble(options.get(key))
                : fallback;
    }

    private static void printUsage() {
        System.out.println("""
                Gomoku ML Training

                Train:
                  java -jar gomoku-ml-training-1.0.0.jar train --hours 24 --serve-data --data-token CHANGE_ME

                Resume:
                  java -jar gomoku-ml-training-1.0.0.jar train --resume ./runs/run-YYYYMMDD-HHmmss --hours 24 --serve-data

                Download complete 100-game batches:
                  java -jar gomoku-ml-training-1.0.0.jar download-data --url http://SERVER:5000 --output ./downloads --token CHANGE_ME

                Serve an existing archive:
                  java -jar gomoku-ml-training-1.0.0.jar serve-data --run ./runs/run-YYYYMMDD-HHmmss --data-port 5000 --data-port-end 5005
                """);
    }
}
