package com.jasonrhan.gomoku.ml.model;

import com.jasonrhan.gomoku.ml.domain.BoardState;
import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.rules.GomokuBoard;

public final class BoardEncoder {
    private BoardEncoder() {
    }

    public static BoardState encode(GomokuBoard board, PlayerColor currentPlayer) {
        if (board.size() != BoardState.BOARD_SIZE) {
            throw new IllegalArgumentException("Only 15x15 boards are supported");
        }
        float[] features = new float[BoardState.INPUT_SIZE];
        PlayerColor opponent = currentPlayer.opponent();
        for (int y = 0; y < board.size(); y++) {
            for (int x = 0; x < board.size(); x++) {
                PlayerColor color = board.get(x, y);
                int index = y * board.size() + x;
                if (color == currentPlayer) {
                    features[index] = 1.0f;
                } else if (color == opponent) {
                    features[BoardState.POLICY_SIZE + index] = 1.0f;
                }
            }
        }
        int lastMoveIndex = -1;
        if (!board.history().isEmpty()) {
            Move last = board.history().get(board.history().size() - 1);
            lastMoveIndex = last.index(board.size());
            features[2 * BoardState.POLICY_SIZE + lastMoveIndex] = 1.0f;
        }
        return new BoardState(features, currentPlayer, lastMoveIndex);
    }
}
