package com.jasonrhan.gomoku.ml.model;

import java.time.Instant;

public record ModelMetadata(
        int schemaVersion,
        String modelVersion,
        int boardSize,
        int inputPlanes,
        int policySize,
        String modelFile,
        String sha256,
        long trainingGames,
        int simulations,
        Instant createdAt,
        String algorithm
) {
    public String toJson() {
        return """
                {
                  "schema_version": %d,
                  "model_version": "%s",
                  "board_size": %d,
                  "input_planes": %d,
                  "policy_size": %d,
                  "model_file": "%s",
                  "sha256": "%s",
                  "training_games": %d,
                  "simulations": %d,
                  "created_at": "%s",
                  "algorithm": "%s"
                }
                """.formatted(
                schemaVersion,
                escape(modelVersion),
                boardSize,
                inputPlanes,
                policySize,
                escape(modelFile),
                escape(sha256),
                trainingGames,
                simulations,
                createdAt,
                escape(algorithm)
        );
    }

    private String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
