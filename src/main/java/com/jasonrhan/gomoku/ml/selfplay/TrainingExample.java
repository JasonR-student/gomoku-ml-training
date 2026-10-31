package com.jasonrhan.gomoku.ml.selfplay;

import com.jasonrhan.gomoku.ml.domain.PlayerColor;

import java.util.Arrays;

public record TrainingExample(
        float[] features,
        float[] policyTarget,
        float valueTarget,
        int action,
        PlayerColor currentPlayer
) {
    public TrainingExample {
        features = Arrays.copyOf(features, features.length);
        policyTarget = Arrays.copyOf(policyTarget, policyTarget.length);
    }

    @Override
    public float[] features() {
        return Arrays.copyOf(features, features.length);
    }

    @Override
    public float[] policyTarget() {
        return Arrays.copyOf(policyTarget, policyTarget.length);
    }

    public float[] featuresUnsafe() {
        return features;
    }

    public float[] policyTargetUnsafe() {
        return policyTarget;
    }
}
