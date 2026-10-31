package com.jasonrhan.gomoku.ml.domain;

public enum GameResult {
    PLAYING,
    BLACK_WIN,
    WHITE_WIN,
    DRAW;

    public static GameResult fromWinner(PlayerColor color) {
        return color == PlayerColor.BLACK ? BLACK_WIN : WHITE_WIN;
    }

    public PlayerColor winner() {
        return switch (this) {
            case BLACK_WIN -> PlayerColor.BLACK;
            case WHITE_WIN -> PlayerColor.WHITE;
            default -> null;
        };
    }
}
