package com.jasonrhan.gomoku.ml.rules;

import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class GomokuBoard {
    public static final int DEFAULT_SIZE = 15;
    private static final int[][] DIRECTIONS = {
            {1, 0},
            {0, 1},
            {1, 1},
            {1, -1}
    };

    private final int size;
    private final PlayerColor[][] grid;
    private final List<Move> history = new ArrayList<>();

    public GomokuBoard() {
        this(DEFAULT_SIZE);
    }

    public GomokuBoard(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("Board size must be positive");
        }
        this.size = size;
        this.grid = new PlayerColor[size][size];
    }

    public int size() {
        return size;
    }

    public GomokuBoard copy() {
        GomokuBoard copy = new GomokuBoard(size);
        for (int y = 0; y < size; y++) {
            System.arraycopy(grid[y], 0, copy.grid[y], 0, size);
        }
        copy.history.addAll(history);
        return copy;
    }

    public void reset() {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                grid[y][x] = null;
            }
        }
        history.clear();
    }

    public boolean isInside(int x, int y) {
        return x >= 0 && x < size && y >= 0 && y < size;
    }

    public PlayerColor get(int x, int y) {
        return isInside(x, y) ? grid[y][x] : null;
    }

    public boolean isEmpty(int x, int y) {
        return isInside(x, y) && grid[y][x] == null;
    }

    public boolean place(int x, int y, PlayerColor color) {
        if (!isEmpty(x, y)) {
            return false;
        }
        grid[y][x] = color;
        history.add(new Move(x, y, color));
        return true;
    }

    public Move undo() {
        if (history.isEmpty()) {
            return null;
        }
        Move move = history.remove(history.size() - 1);
        grid[move.y()][move.x()] = null;
        return move;
    }

    public List<Move> history() {
        return Collections.unmodifiableList(history);
    }

    public int occupiedCount() {
        return history.size();
    }

    public boolean isFull() {
        return occupiedCount() >= size * size;
    }

    public List<Move> legalMoves(PlayerColor color, ForbiddenMoveService forbiddenMoveService) {
        List<Move> moves = new ArrayList<>();
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                if (!isEmpty(x, y)) {
                    continue;
                }
                if (color == PlayerColor.BLACK
                        && forbiddenMoveService.check(this, x, y, color) != null) {
                    continue;
                }
                moves.add(new Move(x, y, color));
            }
        }
        if (moves.isEmpty() && !isFull()) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    if (isEmpty(x, y)) {
                        moves.add(new Move(x, y, color));
                    }
                }
            }
        }
        return moves;
    }

    public List<Move> candidateMoves(
            PlayerColor color,
            ForbiddenMoveService forbiddenMoveService,
            int radius,
            int limit
    ) {
        if (history.isEmpty()) {
            return List.of(new Move(size / 2, size / 2, color));
        }
        List<Move> candidates = new ArrayList<>();
        boolean[][] seen = new boolean[size][size];
        List<Move> stones = List.copyOf(history);
        for (Move stone : stones) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int x = stone.x() + dx;
                    int y = stone.y() + dy;
                    if (!isInside(x, y) || seen[y][x] || !isEmpty(x, y)) {
                        continue;
                    }
                    seen[y][x] = true;
                    if (color != PlayerColor.BLACK
                            || forbiddenMoveService.check(this, x, y, color) == null) {
                        candidates.add(new Move(x, y, color));
                    }
                }
            }
        }
        candidates.sort((first, second) -> {
            int firstDensity = neighborCount(first.x(), first.y(), radius);
            int secondDensity = neighborCount(second.x(), second.y(), radius);
            if (firstDensity != secondDensity) {
                return Integer.compare(secondDensity, firstDensity);
            }
            double center = (size - 1) / 2.0;
            double firstDistance = Math.abs(first.x() - center) + Math.abs(first.y() - center);
            double secondDistance = Math.abs(second.x() - center) + Math.abs(second.y() - center);
            return Double.compare(firstDistance, secondDistance);
        });
        if (candidates.size() >= limit) {
            return new ArrayList<>(candidates.subList(0, limit));
        }
        if (!candidates.isEmpty()) {
            return candidates;
        }
        return legalMoves(color, forbiddenMoveService);
    }

    private int neighborCount(int x, int y, int radius) {
        int count = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if ((dx != 0 || dy != 0) && isInside(x + dx, y + dy)
                        && get(x + dx, y + dy) != null) {
                    count++;
                }
            }
        }
        return count;
    }

    public int countDirection(int x, int y, PlayerColor color, int dx, int dy) {
        int count = 0;
        int nextX = x + dx;
        int nextY = y + dy;
        while (isInside(nextX, nextY) && grid[nextY][nextX] == color) {
            count++;
            nextX += dx;
            nextY += dy;
        }
        return count;
    }

    public int lineLength(int x, int y, PlayerColor color, int dx, int dy) {
        return 1
                + countDirection(x, y, color, dx, dy)
                + countDirection(x, y, color, -dx, -dy);
    }

    public int maxLineLength(int x, int y, PlayerColor color) {
        int max = 0;
        for (int[] direction : DIRECTIONS) {
            max = Math.max(max, lineLength(x, y, color, direction[0], direction[1]));
        }
        return max;
    }

    public boolean isWinningMove(int x, int y, PlayerColor color) {
        if (!isInside(x, y)) {
            return false;
        }
        for (int[] direction : DIRECTIONS) {
            if (lineLength(x, y, color, direction[0], direction[1]) >= 5) {
                return true;
            }
        }
        return false;
    }

    public PlayerColor winner() {
        if (!history.isEmpty()) {
            Move last = history.get(history.size() - 1);
            if (isWinningMove(last.x(), last.y(), last.color())) {
                return last.color();
            }
        }
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                PlayerColor color = grid[y][x];
                if (color != null && isWinningMove(x, y, color)) {
                    return color;
                }
            }
        }
        return null;
    }

    public GameResult gameResult() {
        PlayerColor winner = winner();
        if (winner != null) {
            return GameResult.fromWinner(winner);
        }
        return isFull() ? GameResult.DRAW : GameResult.PLAYING;
    }
}
