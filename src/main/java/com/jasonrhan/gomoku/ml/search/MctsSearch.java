package com.jasonrhan.gomoku.ml.search;

import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.model.BoardEncoder;
import com.jasonrhan.gomoku.ml.model.PolicyValueNetwork;
import com.jasonrhan.gomoku.ml.model.Prediction;
import com.jasonrhan.gomoku.ml.rules.ForbiddenMoveService;
import com.jasonrhan.gomoku.ml.rules.GomokuBoard;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class MctsSearch {
    private final PolicyValueNetwork network;
    private final ForbiddenMoveService forbiddenMoveService;
    private final Random random;
    private final int simulations;
    private final double cPuct;
    private final double rootNoise;
    private final int candidateRadius;
    private final int candidateLimit;

    public MctsSearch(
            PolicyValueNetwork network,
            ForbiddenMoveService forbiddenMoveService,
            Random random,
            int simulations,
            double cPuct,
            double rootNoise,
            int candidateRadius,
            int candidateLimit
    ) {
        this.network = network;
        this.forbiddenMoveService = forbiddenMoveService;
        this.random = random;
        this.simulations = Math.max(1, simulations);
        this.cPuct = cPuct;
        this.rootNoise = Math.max(0.0, Math.min(1.0, rootNoise));
        this.candidateRadius = candidateRadius;
        this.candidateLimit = candidateLimit;
    }

    public SearchResult search(
            GomokuBoard board,
            PlayerColor currentPlayer,
            int moveNumber
    ) {
        MctsNode root = new MctsNode(
                board.copy(), currentPlayer, -1, 1.0, null
        );
        double rootValue = expand(root);
        applyRootNoise(root);
        for (int simulation = 0; simulation < simulations; simulation++) {
            runSimulation(root);
        }

        float[] policy = new float[GomokuBoard.DEFAULT_SIZE * GomokuBoard.DEFAULT_SIZE];
        int totalVisits = root.children.values().stream()
                .mapToInt(child -> child.visits)
                .sum();
        if (totalVisits == 0) {
            policy[root.children.keySet().stream().findFirst().orElse(112)] = 1.0f;
        } else {
            for (MctsNode child : root.children.values()) {
                policy[child.action] = (float) child.visits / totalVisits;
            }
        }

        double temperature = moveNumber < 8 ? 1.0 : 0.0;
        int action = chooseAction(root.children, temperature);
        return new SearchResult(action, policy, (float) rootValue);
    }

    private double expand(MctsNode node) {
        GameResult result = node.board.gameResult();
        if (result != GameResult.PLAYING) {
            node.expanded = true;
            return terminalValue(node, result);
        }
        Prediction prediction = network.predict(
                BoardEncoder.encode(node.board, node.playerToMove).features()
        );
        List<Move> legalMoves = node.board.candidateMoves(
                node.playerToMove,
                forbiddenMoveService,
                candidateRadius,
                candidateLimit
        );
        double totalPrior = 0.0;
        for (Move move : legalMoves) {
            totalPrior += Math.max(0.0, prediction.policy()[move.index(node.board.size())]);
        }
        if (totalPrior <= 1.0e-8) {
            totalPrior = legalMoves.size();
        }
        for (Move move : legalMoves) {
            double prior = prediction.policy()[move.index(node.board.size())];
            if (prior <= 0.0) {
                prior = 1.0;
            }
            node.addChild(move, prior / totalPrior);
        }
        node.expanded = true;
        return prediction.value();
    }

    private void applyRootNoise(MctsNode root) {
        if (root.children.isEmpty() || rootNoise <= 0.0) {
            return;
        }
        double uniform = 1.0 / root.children.size();
        for (MctsNode child : root.children.values()) {
            child.prior = (1.0 - rootNoise) * child.prior + rootNoise * uniform;
        }
    }

    private void runSimulation(MctsNode root) {
        MctsNode node = root;
        List<MctsNode> path = new ArrayList<>();
        path.add(node);
        while (node.expanded
                && !node.children.isEmpty()
                && node.board.gameResult() == GameResult.PLAYING) {
            node = selectChild(node);
            path.add(node);
        }

        double value;
        if (node.board.gameResult() == GameResult.PLAYING) {
            value = expand(node);
        } else {
            value = terminalValue(node, node.board.gameResult());
        }

        for (int index = path.size() - 1; index >= 0; index--) {
            MctsNode visited = path.get(index);
            visited.visits++;
            visited.valueSum += value;
            value = -value;
        }
    }

    private MctsNode selectChild(MctsNode node) {
        double parentVisits = Math.sqrt(Math.max(1, node.visits));
        MctsNode best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (MctsNode child : node.children.values()) {
            double exploration = cPuct * child.prior * parentVisits / (1.0 + child.visits);
            double score = -child.qValue() + exploration;
            if (score > bestScore) {
                bestScore = score;
                best = child;
            }
        }
        return best;
    }

    private int chooseAction(Map<Integer, MctsNode> children, double temperature) {
        if (children.isEmpty()) {
            return 112;
        }
        if (temperature <= 0.0) {
            return children.values().stream()
                    .max((first, second) -> Integer.compare(first.visits, second.visits))
                    .orElseThrow()
                    .action;
        }
        double total = 0.0;
        for (MctsNode child : children.values()) {
            total += Math.pow(Math.max(1, child.visits), 1.0 / temperature);
        }
        double target = random.nextDouble() * total;
        double running = 0.0;
        for (MctsNode child : children.values()) {
            running += Math.pow(Math.max(1, child.visits), 1.0 / temperature);
            if (running >= target) {
                return child.action;
            }
        }
        return children.values().stream().reduce((first, second) -> second).orElseThrow().action;
    }

    private double terminalValue(MctsNode node, GameResult result) {
        if (result == GameResult.DRAW) {
            return 0.0;
        }
        return result.winner() == node.playerToMove ? 1.0 : -1.0;
    }
}
