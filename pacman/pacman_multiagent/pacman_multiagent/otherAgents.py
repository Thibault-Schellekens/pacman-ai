import time

from multiAgents import AlphaBetaAgent
from transpositionTable import TranspositionTable
from pacman import GameState
from moveOrdering import *

import util

class PVSAgent(AlphaBetaAgent):
    def __init__(self, evalFn="score_evaluation_function", depth="2", moveOrder=True):
        super(PVSAgent, self).__init__(evalFn, depth)
        self.moveOrder = moveOrder in [True, "True", "true", "1"]
        self.fails = 0

    def get_action(self, gameState: GameState):
        return self.pvs_alphabeta(gameState, self.depth, 0, float("-inf"), float("inf"))[0]

    def pvs_alphabeta(self, gameState: GameState, depth, currentAgent, alpha, beta):
        self.nodes += 1

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        isMaximizing = currentAgent == 0

        bestValue = float("-inf") if isMaximizing else float("inf")
        bestAction = None
        actions = gameState.getLegalActions(currentAgent)

        if self.moveOrder:
            actions = scoreMoves(gameState, actions, currentAgent)

        firstMove = True
        for i in range(len(actions)):
            if self.moveOrder:
                pickMove(actions, i)
                action, _, successor = actions[i]
            else:
                action = actions[i]
                successor = gameState.generateSuccessor(currentAgent, action)

            nextAgent = (currentAgent + 1) % numAgents
            nextDepth = depth - (1 if currentAgent == 0 else 0)

            if firstMove:
                # Full depth search for the first move
                _, value = self.pvs_alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                firstMove = False
            else:
                if isMaximizing:
                    # Null window search for maximizing node
                    _, value = self.pvs_alphabeta(successor, nextDepth, nextAgent, alpha, alpha + 1)
                    # Re-search if it might be better than our current best
                    if alpha < value < beta:
                        self.fails += 1
                        _, value = self.pvs_alphabeta(successor, nextDepth, nextAgent, value, beta)
                else:
                    # Null window search for minimizing node
                    _, value = self.pvs_alphabeta(successor, nextDepth, nextAgent, beta - 1, beta)
                    # Re-search if it might be better than our current best

                    if alpha < value < beta:
                        self.fails += 1
                        _, value = self.pvs_alphabeta(successor, nextDepth, nextAgent, alpha, value)

            if isMaximizing:
                if value > bestValue:
                    bestValue, bestAction = value, action
                alpha = max(alpha, value)
            else:
                if value < bestValue:
                    bestValue, bestAction = value, action
                beta = min(beta, value)

            if alpha >= beta:
                self.cuts += 1
                break

        return bestAction, bestValue

    def show_stats(self, numGames):
        super().show_stats(numGames)
        print(f"Average of fails per game : {self.fails / numGames}")

class AlphaBetaCachedAgent(AlphaBetaAgent):
    def __init__(self, evalFn="score_evaluation_function", depth="2"):
        super(AlphaBetaCachedAgent, self).__init__(evalFn, depth)
        self.tt = TranspositionTable()
        self.hits = 0

    def get_action(self, gameState : GameState):
        self.tt.clear()
        return self.cached_alphabeta(gameState, self.depth, 0, float("-inf"), float("inf"))[0]

    def cached_alphabeta(self, gameState : GameState, depth, currentAgent, alpha, beta):
        self.nodes += 1

        if depth <= 0 or gameState.isWin() or gameState.isLose():
            eval_value = self.evaluationFunction(gameState)
            return None, eval_value

        hit, move, score = self.tt.lookup(gameState, depth, alpha, beta)
        if hit:
            self.hits += 1
            return move, score

        numAgents = gameState.getNumAgents()
        actions = gameState.getLegalActions(currentAgent)
        bestAction = None

        if currentAgent == 0:
            bestValue = float("-inf")
            for action in actions:
                successor = gameState.generateSuccessor(currentAgent, action)
                nextAgent = (currentAgent + 1) % numAgents
                nextDepth = depth - (1 if nextAgent == 0 else 0)

                _, value = self.cached_alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                alpha = max(alpha, value)
                if bestValue >= beta:
                    self.cuts += 1
                    break
        else:
            bestValue = float("inf")
            for action in actions:
                successor = gameState.generateSuccessor(currentAgent, action)
                nextAgent = (currentAgent + 1) % numAgents
                nextDepth = depth - (1 if nextAgent == 0 else 0)

                _, value = self.cached_alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                if value < bestValue:
                    bestValue = value
                    bestAction = action
                beta = min(beta, value)
                if bestValue <= alpha:
                    self.cuts += 1
                    break

        self.tt.put(gameState, [bestAction] if bestAction else [], depth, bestValue, alpha, beta)
        return bestAction, bestValue

    def show_stats(self, numGames):
        super().show_stats(numGames)
        print("Transposition Table Informations:")
        print(f"Average of tt hits per game : {self.hits / numGames}")
        print(f"Average of tt clear per game : {self.tt.clears / numGames}")


class PVSCachedAgent(AlphaBetaAgent):
    def __init__(self, evalFn="score_evaluation_function", depth="2"):
        super(PVSCachedAgent, self).__init__(evalFn, depth)
        self.tt = TranspositionTable()
        self.hits = 0
        self.fails = 0

    def get_action(self, gameState : GameState):
        # Do I have to clear before each move ?
        self.tt.clear()
        return self.search(gameState, self.depth, 0, float("-inf"), float("inf"))[0]

    def search(self, gameState : GameState, depth, currentAgent, alpha, beta):
        self.nodes += 1

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        hit, move, score = self.tt.lookup(gameState, depth, alpha, beta)
        if hit:
            self.hits += 1
            if score != float("-inf"):
                return move, score

        numAgents = gameState.getNumAgents()
        isMaximizing = currentAgent == 0

        bestValue = float("-inf") if isMaximizing else float("inf")

        bestAction = None
        actions = gameState.getLegalActions(currentAgent)

        actions = scoreMoves(gameState, actions, currentAgent)

        firstMove = True
        for i in range(len(actions)):

            pickMove(actions, i)
            action, _, successor = actions[i]

            nextAgent = (currentAgent + 1) % numAgents
            nextDepth = depth - (1 if currentAgent == 0 else 0)

            if firstMove:
                # Full depth search for the first move
                _, value = self.search(successor, nextDepth, nextAgent, alpha, beta)
                firstMove = False
            else:
                if isMaximizing:
                    # Null window search for maximizing node
                    _, value = self.search(successor, nextDepth, nextAgent, alpha, alpha + 1)
                    # Re-search if it might be better than our current best
                    if alpha < value < beta:
                        self.fails += 1
                        _, value = self.search(successor, nextDepth, nextAgent, value, beta)
                else:
                    # Null window search for minimizing node
                    _, value = self.search(successor, nextDepth, nextAgent, beta - 1, beta)
                    # Re-search if it might be better than our current best
                    if alpha < value < beta:
                        self.fails += 1
                        _, value = self.search(successor, nextDepth, nextAgent, alpha, value)

            if isMaximizing:
                if value > bestValue:
                    bestValue, bestAction = value, action
                alpha = max(alpha, value)
            else:
                if value < bestValue:
                    bestValue, bestAction = value, action
                beta = min(beta, value)

            if alpha >= beta:
                self.cuts += 1
                break


        self.tt.put(gameState, [bestAction] if bestAction else [], depth, bestValue, alpha, beta)

        return bestAction, bestValue

    def show_stats(self, numGames):
        super().show_stats(numGames)
        print(f"Average of fails per game : {self.fails / numGames}")
        print("Transposition Table Informations:")
        print(f"Average of tt hits per game : {self.hits / numGames}")
        print(f"Average of tt clear per game : {self.tt.clears / numGames}")
