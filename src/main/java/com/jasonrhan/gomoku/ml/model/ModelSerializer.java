package com.jasonrhan.gomoku.ml.model;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class ModelSerializer {
    private static final String MAGIC = "GOMOKU_MLP";
    private static final int VERSION = 1;

    private ModelSerializer() {
    }

    public static void save(
            Path path,
            PolicyValueNetwork network,
            AdamOptimizer optimizer
    ) throws IOException {
        Files.createDirectories(path.toAbsolutePath().normalize().getParent());
        try (DataOutputStream output = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path))
        )) {
            output.writeUTF(MAGIC);
            output.writeInt(VERSION);
            output.writeInt(PolicyValueNetwork.INPUT_SIZE);
            output.writeInt(PolicyValueNetwork.HIDDEN_ONE);
            output.writeInt(PolicyValueNetwork.HIDDEN_TWO);
            output.writeInt(PolicyValueNetwork.POLICY_SIZE);
            output.writeInt(network.parameters().size());
            for (TrainableParameter parameter : network.parameters()) {
                output.writeUTF(parameter.name());
                output.writeInt(parameter.values().length);
                for (float value : parameter.values()) {
                    output.writeFloat(value);
                }
            }
            optimizer.writeTo(output, network.parameters());
        }
    }

    public static PolicyValueNetwork.LoadedNetwork load(Path path) throws IOException {
        try (DataInputStream input = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path))
        )) {
            if (!MAGIC.equals(input.readUTF())) {
                throw new IOException("Invalid model magic");
            }
            if (input.readInt() != VERSION
                    || input.readInt() != PolicyValueNetwork.INPUT_SIZE
                    || input.readInt() != PolicyValueNetwork.HIDDEN_ONE
                    || input.readInt() != PolicyValueNetwork.HIDDEN_TWO
                    || input.readInt() != PolicyValueNetwork.POLICY_SIZE) {
                throw new IOException("Unsupported model shape or version");
            }
            PolicyValueNetwork network = PolicyValueNetwork.empty();
            int parameterCount = input.readInt();
            if (parameterCount != network.parameters().size()) {
                throw new IOException("Unexpected parameter count");
            }
            for (int i = 0; i < parameterCount; i++) {
                String name = input.readUTF();
                int length = input.readInt();
                TrainableParameter parameter = network.parameters().stream()
                        .filter(candidate -> candidate.name().equals(name))
                        .findFirst()
                        .orElseThrow(() -> new IOException("Unknown parameter: " + name));
                if (parameter.values().length != length) {
                    throw new IOException("Parameter length mismatch: " + name);
                }
                for (int j = 0; j < length; j++) {
                    parameter.values()[j] = input.readFloat();
                }
            }
            AdamOptimizer optimizer = AdamOptimizer.readFrom(input, network.parameters());
            return new PolicyValueNetwork.LoadedNetwork(network, optimizer);
        }
    }
}
