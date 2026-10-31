package com.jasonrhan.gomoku.ml.selfplay;

import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.model.BoardEncoder;
import com.jasonrhan.gomoku.ml.model.PolicyValueNetwork;
import com.jasonrhan.gomoku.ml.rules.ForbiddenMoveService;
import com.jasonrhan.gomoku.ml.rules.GomokuBoard;
import com.jasonrhan.gomoku.ml.search.MctsSearch;
import com.jasonrhan.gomoku.ml.search.SearchResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SelfPlayRunner {
    public record GameRecord(
            GameResult result,
            int moves,
            List<Move> moveList,
            List<TrainingExample> examples
    ) {
    }

    private final ForbiddenMoveService forbiddenMoveService;
    private final Random random;
    private final int simulations;
    private final double cPuct;
    private final double rootNoise;
    private final int candidateRadius;
    private final int candidateLimit;

    public SelfPlayRunner(
            ForbiddenMoveService forbiddenMoveService,
            Random random,
            int simulations,
            double cPuct,
            double rootNoise,
            int candidateRadius,
            int candidateLimit
    ) {
        this.forbiddenMoveService = forbiddenMoveService;
        this.random = random;
        this.simulations = simulations;
        this.cPuct = cPuct;
        this.rootNoise = rootNoise;
        this.candidateRadius = candidateRadius;
        this.candidateLimit = candidateLimit;
    }

    public GameRecord playOneGame(PolicyValueNetwork network) {
        GomokuBoard board = new GomokuBoard();
        PlayerColor currentPlayer = PlayerColor.BLACK;
        List<PendingExample> pending = new ArrayList<>();
        List<Move> moveList = new ArrayList<>();
        MctsSearch search = new MctsSearch(
                network,
                forbiddenMoveService,
                random,
                simulations,
                cPuct,
                rootNoise,
                candidateRadius,
                candidateLimit
        );

        while (board.gameResult() == GameResult.PLAYING
                && board.occupiedCount() < GomokuBoard.DEFAULT_SIZE * GomokuBoard.DEFAULT_SIZE) {
            float[] features = BoardEncoder.encode(board, currentPlayer).features();
            SearchResult result = search.search(board, currentPlayer, board.occupiedCount());
            int action = result.action();
            int x = action % board.size();
            int y = action / board.size();
            if (!board.place(x, y, currentPlayer)) {
                List<com.jasonrhan.gomoku.ml.domain.Move> legal = board.legalMoves(
                        currentPlayer, forbiddenMoveService
                );
                if (legal.isEmpty()) {
                    break;
                }
                var fallback = legal.get(random.nextInt(legal.size()));
                action = fallback.index(board.size());
                x = fallback.x();
                y = fallback.y();
                if (!board.place(x, y, currentPlayer)) {
                    break;
                }
            }
            pending.add(new PendingExample(
                    features,
                    result.policyTarget(),
                    action,
                    currentPlayer
            ));
            moveList.add(new Move(x, y, currentPlayer));
            currentPlayer = currentPlayer.opponent();
        }

        GameResult result = board.gameResult();
        if (result == GameResult.PLAYING) {
            result = GameResult.DRAW;
        }
        PlayerColor winner = result.winner();
        List<TrainingExample> examples = new ArrayList<>(pending.size());
        for (PendingExample example : pending) {
            float value = winner == null
                    ? 0.0f
                    : winner == example.currentPlayer() ? 1.0f : -1.0f;
            examples.add(new TrainingExample(
                    example.features(),
                    example.policyTarget(),
                    value,
                    example.action(),
                    example.currentPlayer()
            ));
        }
        return new GameRecord(result, pending.size(), moveList, examples);
    }

    private record PendingExample(
            float[] features,
            float[] policyTarget,
            int action,
            PlayerColor currentPlayer
    ) {
    }
}
