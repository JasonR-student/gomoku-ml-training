package com.jasonrhan.gomoku.ml;

import com.jasonrhan.gomoku.ml.model.AdamOptimizer;
import com.jasonrhan.gomoku.ml.model.ModelSerializer;
import com.jasonrhan.gomoku.ml.model.PolicyValueNetwork;
import com.jasonrhan.gomoku.ml.selfplay.TrainingExample;
import com.jasonrhan.gomoku.ml.domain.BoardState;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelTest {
    @Test
    void forwardOutputHasStableShape() {
        PolicyValueNetwork network = new PolicyValueNetwork(7L);
        var prediction = network.predict(new float[BoardState.INPUT_SIZE]);
        assertEquals(BoardState.POLICY_SIZE, prediction.policy().length);
        assertEquals(1.0, sum(prediction.policy()), 1.0e-4);
        assertTrue(prediction.value() >= -1.0f && prediction.value() <= 1.0f);
    }

    @Test
    void trainingAndSerializationPreservePredictions() throws Exception {
        PolicyValueNetwork network = new PolicyValueNetwork(7L);
        AdamOptimizer optimizer = new AdamOptimizer();
        float[] policy = new float[BoardState.POLICY_SIZE];
        policy[112] = 1.0f;
        TrainingExample example = new TrainingExample(
                new float[BoardState.INPUT_SIZE],
                policy,
                1.0f,
                112,
                PlayerColor.BLACK
        );
        network.trainBatch(
                java.util.List.of(example),
                optimizer,
                0.001,
                1.0e-5,
                0.5
        );
        var before = network.predict(example.featuresUnsafe());
        var path = Files.createTempFile("gomoku-model", ".gmk");
        ModelSerializer.save(path, network, optimizer);
        var loaded = ModelSerializer.load(path);
        var after = loaded.network().predict(example.featuresUnsafe());
        assertEquals(before.value(), after.value(), 1.0e-6);
        assertEquals(before.policy()[112], after.policy()[112], 1.0e-6);
    }

    private double sum(float[] values) {
        double total = 0.0;
        for (float value : values) {
            total += value;
        }
        return total;
    }
}
