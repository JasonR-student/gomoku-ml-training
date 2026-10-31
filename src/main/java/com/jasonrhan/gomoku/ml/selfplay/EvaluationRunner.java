package com.jasonrhan.gomoku.ml.selfplay;

import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.rules.ForbiddenMoveService;
import com.jasonrhan.gomoku.ml.rules.GomokuBoard;
import com.jasonrhan.gomoku.ml.model.PolicyValueNetwork;
import com.jasonrhan.gomoku.ml.search.MctsSearch;

import java.util.List;
import java.util.Random;

public final class EvaluationRunner {
    public record EvaluationSummary(
            int games,
            int wins,
            int losses,
            int draws
    ) {
    }

    private final ForbiddenMoveService forbiddenMoveService;
    private final Random random;
    private final int simulations;

    public EvaluationRunner(
            ForbiddenMoveService forbiddenMoveService,
            Random random,
            int simulations
    ) {
        this.forbiddenMoveService = forbiddenMoveService;
        this.random = random;
        this.simulations = simulations;
    }

    public EvaluationSummary playAgainstRandom(
            PolicyValueNetwork network,
            int games
    ) {
        int wins = 0;
        int losses = 0;
        int draws = 0;
        for (int game = 0; game < games; game++) {
            PlayerColor modelColor = game % 2 == 0 ? PlayerColor.BLACK : PlayerColor.WHITE;
            GameResult result = playOne(network, modelColor);
            if (result == GameResult.DRAW) {
                draws++;
            } else if (result.winner() == modelColor) {
                wins++;
            } else {
                losses++;
            }
        }
        return new EvaluationSummary(games, wins, losses, draws);
    }

    private GameResult playOne(PolicyValueNetwork network, PlayerColor modelColor) {
        GomokuBoard board = new GomokuBoard();
        PlayerColor current = PlayerColor.BLACK;
        MctsSearch search = new MctsSearch(
                network,
                forbiddenMoveService,
                random,
                simulations,
                1.25,
                0.0,
                2,
                64
        );
        while (board.gameResult() == GameResult.PLAYING) {
            if (current == modelColor) {
                int action = search.search(
                        board, current, board.occupiedCount()
                ).action();
                board.place(action % board.size(), action / board.size(), current);
            } else {
                List<Move> moves = board.candidateMoves(
                        current, forbiddenMoveService, 2, 64
                );
                if (moves.isEmpty()) {
                    return GameResult.DRAW;
                }
                Move move = moves.get(random.nextInt(moves.size()));
                board.place(move.x(), move.y(), current);
            }
            current = current.opponent();
        }
        return board.gameResult();
    }
}
