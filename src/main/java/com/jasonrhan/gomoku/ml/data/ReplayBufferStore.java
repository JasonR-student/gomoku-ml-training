package com.jasonrhan.gomoku.ml.data;

import com.jasonrhan.gomoku.ml.domain.BoardState;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.selfplay.TrainingExample;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Random;

public final class ReplayBufferStore implements AutoCloseable {
    private static final int FILE_MAGIC = 0x474D5250;
    private static final int VERSION = 1;
    private static final int RECORD_MAGIC = 0x52454331;
    private static final int SHARD_SIZE = 4096;

    private final Path directory;
    private final int capacity;
    private final Deque<TrainingExample> recent = new ArrayDeque<>();
    private DataOutputStream output;
    private int currentShardCount;
    private int nextShardIndex;

    public ReplayBufferStore(Path directory, int capacity) throws IOException {
        this.directory = directory.toAbsolutePath().normalize();
        this.capacity = Math.max(1, capacity);
        Files.createDirectories(this.directory);
        loadExisting();
        openNewShard();
    }

    public synchronized void append(TrainingExample example) throws IOException {
        if (currentShardCount >= SHARD_SIZE) {
            openNewShard();
        }
        if (output == null) {
            openNewShard();
        }
        writeRecord(output, example);
        output.flush();
        currentShardCount++;
        recent.addLast(example);
        while (recent.size() > capacity) {
            recent.removeFirst();
        }
    }

    public synchronized List<TrainingExample> sample(
            int count,
            Random random
    ) {
        List<TrainingExample> values = new ArrayList<>(recent);
        if (values.isEmpty()) {
            return List.of();
        }
        List<TrainingExample> result = new ArrayList<>(Math.min(count, values.size()));
        for (int i = 0; i < count; i++) {
            result.add(values.get(random.nextInt(values.size())));
        }
        return result;
    }

    public synchronized int size() {
        return recent.size();
    }

    public Path directory() {
        return directory;
    }

    @Override
    public synchronized void close() throws IOException {
        if (output != null) {
            output.flush();
            output.close();
            output = null;
        }
    }

    private void loadExisting() throws IOException {
        List<Path> shards;
        try (var paths = Files.list(directory)) {
            shards = paths
                    .filter(path -> path.getFileName().toString().startsWith("shard-"))
                    .filter(path -> path.getFileName().toString().endsWith(".bin"))
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
        }
        for (Path shard : shards) {
            int shardNumber = parseShardNumber(shard);
            nextShardIndex = Math.max(nextShardIndex, shardNumber + 1);
            readShard(shard);
        }
    }

    private void readShard(Path shard) throws IOException {
        try (DataInputStream input = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(shard))
        )) {
            if (input.readInt() != FILE_MAGIC || input.readInt() != VERSION) {
                return;
            }
            while (true) {
                try {
                    if (input.readInt() != RECORD_MAGIC) {
                        return;
                    }
                    TrainingExample example = readRecord(input);
                    recent.addLast(example);
                    while (recent.size() > capacity) {
                        recent.removeFirst();
                    }
                } catch (EOFException endOfFile) {
                    return;
                } catch (RuntimeException invalidRecord) {
                    return;
                }
            }
        }
    }

    private void openNewShard() throws IOException {
        if (output != null) {
            output.flush();
            output.close();
        }
        Path shard = directory.resolve(
                "shard-%06d.bin".formatted(nextShardIndex++)
        );
        output = new DataOutputStream(new BufferedOutputStream(
                Files.newOutputStream(
                        shard,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE
                )
        ));
        output.writeInt(FILE_MAGIC);
        output.writeInt(VERSION);
        output.flush();
        currentShardCount = 0;
    }

    private void writeRecord(
            DataOutputStream output,
            TrainingExample example
    ) throws IOException {
        output.writeInt(RECORD_MAGIC);
        float[] features = example.featuresUnsafe();
        output.writeInt(features.length);
        for (float feature : features) {
            output.writeByte(Math.max(0, Math.min(255, Math.round(feature * 255.0f))));
        }
        float[] policy = example.policyTargetUnsafe();
        output.writeInt(policy.length);
        for (float value : policy) {
            output.writeFloat(value);
        }
        output.writeFloat(example.valueTarget());
        output.writeInt(example.action());
        output.writeByte(example.currentPlayer() == PlayerColor.BLACK ? 0 : 1);
    }

    private TrainingExample readRecord(DataInputStream input) throws IOException {
        int featureLength = input.readInt();
        if (featureLength != BoardState.INPUT_SIZE) {
            throw new IOException("Unexpected feature length");
        }
        float[] features = new float[featureLength];
        for (int i = 0; i < featureLength; i++) {
            features[i] = input.readUnsignedByte() / 255.0f;
        }
        int policyLength = input.readInt();
        if (policyLength != BoardState.POLICY_SIZE) {
            throw new IOException("Unexpected policy length");
        }
        float[] policy = new float[policyLength];
        for (int i = 0; i < policyLength; i++) {
            policy[i] = input.readFloat();
        }
        float value = input.readFloat();
        int action = input.readInt();
        int player = input.readUnsignedByte();
        PlayerColor color = player == 0 ? PlayerColor.BLACK : PlayerColor.WHITE;
        return new TrainingExample(features, policy, value, action, color);
    }

    private int parseShardNumber(Path shard) {
        String name = shard.getFileName().toString();
        int start = name.indexOf('-') + 1;
        int end = name.lastIndexOf('.');
        try {
            return Integer.parseInt(name.substring(start, end));
        } catch (RuntimeException exception) {
            return nextShardIndex;
        }
    }
}
