package com.jasonrhan.gomoku.ml.search;

import java.util.Arrays;

public record SearchResult(int action, float[] policyTarget, float valueEstimate) {
    public SearchResult {
        policyTarget = Arrays.copyOf(policyTarget, policyTarget.length);
    }

    @Override
    public float[] policyTarget() {
        return Arrays.copyOf(policyTarget, policyTarget.length);
    }
}
