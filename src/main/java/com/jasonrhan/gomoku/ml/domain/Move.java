package com.jasonrhan.gomoku.ml.domain;

import java.util.Objects;

public record Move(int x, int y, PlayerColor color) {
    public Move {
        Objects.requireNonNull(color, "color");
    }

    public int index(int boardSize) {
        return y * boardSize + x;
    }
}
