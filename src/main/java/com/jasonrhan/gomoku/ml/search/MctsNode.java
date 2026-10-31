package com.jasonrhan.gomoku.ml.search;

import com.jasonrhan.gomoku.ml.domain.Move;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.rules.GomokuBoard;

import java.util.LinkedHashMap;
import java.util.Map;

final class MctsNode {
    final GomokuBoard board;
    final PlayerColor playerToMove;
    final int action;
    final MctsNode parent;
    final Map<Integer, MctsNode> children = new LinkedHashMap<>();
    double prior;
    double valueSum;
    int visits;
    boolean expanded;

    MctsNode(
            GomokuBoard board,
            PlayerColor playerToMove,
            int action,
            double prior,
            MctsNode parent
    ) {
        this.board = board;
        this.playerToMove = playerToMove;
        this.action = action;
        this.prior = prior;
        this.parent = parent;
    }

    double qValue() {
        return visits == 0 ? 0.0 : valueSum / visits;
    }

    void addChild(Move move, double childPrior) {
        GomokuBoard childBoard = board.copy();
        childBoard.place(move.x(), move.y(), move.color());
        int childAction = move.index(childBoard.size());
        children.put(
                childAction,
                new MctsNode(
                        childBoard,
                        playerToMove.opponent(),
                        childAction,
                        childPrior,
                        this
                )
        );
    }
}
