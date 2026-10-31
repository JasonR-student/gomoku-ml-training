package com.jasonrhan.gomoku.ml.domain;

import java.util.Arrays;

public final class BoardState {
    public static final int BOARD_SIZE = 15;
    public static final int INPUT_PLANES = 3;
    public static final int POLICY_SIZE = BOARD_SIZE * BOARD_SIZE;
    public static final int INPUT_SIZE = INPUT_PLANES * POLICY_SIZE;

    private final float[] features;
    private final PlayerColor currentPlayer;
    private final int lastMoveIndex;

    public BoardState(float[] features, PlayerColor currentPlayer, int lastMoveIndex) {
        if (features.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Expected " + INPUT_SIZE + " features");
        }
        this.features = Arrays.copyOf(features, features.length);
        this.currentPlayer = currentPlayer;
        this.lastMoveIndex = lastMoveIndex;
    }

    public float[] features() {
        return Arrays.copyOf(features, features.length);
    }

    public PlayerColor currentPlayer() {
        return currentPlayer;
    }

    public int lastMoveIndex() {
        return lastMoveIndex;
    }
}
