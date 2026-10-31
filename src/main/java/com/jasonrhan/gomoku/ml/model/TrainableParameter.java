package com.jasonrhan.gomoku.ml.model;

import java.util.Arrays;

public final class TrainableParameter {
    private final String name;
    private final float[] values;
    private final float[] gradients;

    public TrainableParameter(String name, float[] values) {
        this.name = name;
        this.values = values;
        this.gradients = new float[values.length];
    }

    public String name() {
        return name;
    }

    public float[] values() {
        return values;
    }

    public float[] gradients() {
        return gradients;
    }

    public void clearGradient() {
        Arrays.fill(gradients, 0.0f);
    }
}
