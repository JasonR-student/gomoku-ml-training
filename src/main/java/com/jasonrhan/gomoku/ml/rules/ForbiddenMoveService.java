package com.jasonrhan.gomoku.ml.rules;

import com.jasonrhan.gomoku.ml.domain.PlayerColor;

import java.util.HashSet;
import java.util.Set;

public final class ForbiddenMoveService {
    private static final int[][] DIRECTIONS = {
            {1, 0},
            {0, 1},
            {1, 1},
            {1, -1}
    };

    public ForbiddenMoveResult check(
            GomokuBoard board,
            int x,
            int y,
            PlayerColor color
    ) {
        if (color != PlayerColor.BLACK || !board.isEmpty(x, y)) {
            return null;
        }
        board.place(x, y, color);
        try {
            int[] lengths = new int[DIRECTIONS.length];
            for (int i = 0; i < DIRECTIONS.length; i++) {
                int[] direction = DIRECTIONS[i];
                lengths[i] = board.lineLength(x, y, color, direction[0], direction[1]);
            }
            for (int length : lengths) {
                if (length > 5) {
                    return new ForbiddenMoveResult(
                            ForbiddenMoveType.OVERLINE,
                            "black overline"
                    );
                }
            }
            for (int length : lengths) {
                if (length == 5) {
                    return null;
                }
            }

            int fourCount = 0;
            int threeCount = 0;
            for (int[] direction : DIRECTIONS) {
                Set<Point> winningPoints = directionWinningPoints(
                        board, x, y, color, direction[0], direction[1]
                );
                if (!winningPoints.isEmpty()) {
                    fourCount++;
                } else if (directionHasOpenThree(
                        board, x, y, color, direction[0], direction[1]
                )) {
                    threeCount++;
                }
            }
            if (fourCount >= 2) {
                return new ForbiddenMoveResult(
                        ForbiddenMoveType.DOUBLE_FOUR,
                        "black double four"
                );
            }
            if (threeCount >= 2) {
                return new ForbiddenMoveResult(
                        ForbiddenMoveType.DOUBLE_THREE,
                        "black double three"
                );
            }
            return null;
        } finally {
            board.undo();
        }
    }

    private Set<Point> directionWinningPoints(
            GomokuBoard board,
            int x,
            int y,
            PlayerColor color,
            int dx,
            int dy
    ) {
        Set<Point> points = new HashSet<>();
        for (int offset = -4; offset <= 4; offset++) {
            int nextX = x + dx * offset;
            int nextY = y + dy * offset;
            if (!board.isEmpty(nextX, nextY)) {
                continue;
            }
            board.place(nextX, nextY, color);
            try {
                if (board.lineLength(nextX, nextY, color, dx, dy) == 5) {
                    points.add(new Point(nextX, nextY));
                }
            } finally {
                board.undo();
            }
        }
        return points;
    }

    private boolean directionHasOpenThree(
            GomokuBoard board,
            int x,
            int y,
            PlayerColor color,
            int dx,
            int dy
    ) {
        for (int offset = -4; offset <= 4; offset++) {
            int nextX = x + dx * offset;
            int nextY = y + dy * offset;
            if (!board.isEmpty(nextX, nextY)) {
                continue;
            }
            board.place(nextX, nextY, color);
            try {
                if (board.winner() != null
                        || board.lineLength(nextX, nextY, color, dx, dy) > 5) {
                    continue;
                }
                if (directionWinningPoints(
                        board, nextX, nextY, color, dx, dy
                ).size() >= 2) {
                    return true;
                }
            } finally {
                board.undo();
            }
        }
        return false;
    }

    private record Point(int x, int y) {
    }
}
