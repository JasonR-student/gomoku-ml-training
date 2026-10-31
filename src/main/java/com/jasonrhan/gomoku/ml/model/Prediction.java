package com.jasonrhan.gomoku.ml.model;

import java.util.Arrays;

public record Prediction(float[] policy, float value) {
    public Prediction {
        policy = Arrays.copyOf(policy, policy.length);
    }

    @Override
    public float[] policy() {
        return Arrays.copyOf(policy, policy.length);
    }
}
