package com.jasonrhan.gomoku.ml.domain;

public enum PlayerColor {
    BLACK,
    WHITE;

    public PlayerColor opponent() {
        return this == BLACK ? WHITE : BLACK;
    }
}
