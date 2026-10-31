package com.jasonrhan.gomoku.ml;

import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.rules.ForbiddenMoveService;
import com.jasonrhan.gomoku.ml.rules.ForbiddenMoveType;
import com.jasonrhan.gomoku.ml.rules.GomokuBoard;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RulesTest {
    private final ForbiddenMoveService forbidden = new ForbiddenMoveService();

    @Test
    void detectsWinAndRejectsDuplicateMoves() {
        GomokuBoard board = new GomokuBoard();
        for (int x = 0; x < 5; x++) {
            assertTrue(board.place(x, 7, PlayerColor.BLACK));
        }
        assertEquals(GameResult.BLACK_WIN, board.gameResult());
        assertTrue(!board.place(0, 7, PlayerColor.WHITE));
    }

    @Test
    void detectsForbiddenMoveOnlyForBlack() {
        GomokuBoard board = new GomokuBoard();
        board.place(6, 7, PlayerColor.BLACK);
        board.place(8, 7, PlayerColor.BLACK);
        board.place(7, 6, PlayerColor.BLACK);
        board.place(7, 8, PlayerColor.BLACK);
        var result = forbidden.check(board, 7, 7, PlayerColor.BLACK);
        assertNotNull(result);
        assertEquals(ForbiddenMoveType.DOUBLE_THREE, result.type());
        assertNull(forbidden.check(board, 7, 7, PlayerColor.WHITE));
    }
}
