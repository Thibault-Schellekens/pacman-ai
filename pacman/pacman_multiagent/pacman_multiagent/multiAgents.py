# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
from sys import flags

import util

from game import Agent
from multiAgentsHelper import *


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.getLegalActions()
        legal_moves = [move for move in legal_moves if move != Directions.STOP]
        # Choose one of the best actions
        scores = [
            self.evaluation_function(game_state, action) for action in legal_moves
        ]
        best_score = max(scores)
        best_indices = [
            index for index in range(len(scores)) if scores[index] == best_score
        ]
        # Pick randomly among the best
        chosen_index = random.choice(best_indices)

        "Add more of your code here if you want to"
        move = legal_moves[chosen_index]
        return move

    def evaluation_function(self, current_game_state: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state: GameState = current_game_state.generatePacmanSuccessor(
            action
        )
        if successor_game_state.isWin():
            return float("inf")
        elif successor_game_state.isLose():
            return float("-inf")

        new_pos = successor_game_state.getPacmanPosition()
        new_ghost_states = successor_game_state.getGhostStates()
        new_capsules = successor_game_state.getCapsules()

        "*** YOUR CODE HERE ***"
        score = successor_game_state.getScore()

        score += 200 if new_pos in current_game_state.getCapsules() else 0

        scared_ghosts = [ghost for ghost in new_ghost_states if ghost.scaredTimer > 0]
        non_scared_ghosts = [ghost for ghost in new_ghost_states if ghost.scaredTimer == 0]

        if scared_ghosts:
            closest_ghost = (
                min(scared_ghosts, key=lambda x: manhattan_distance(new_pos, x.getPosition()))).getPosition()
            closest_ghost = util.nearest_point(closest_ghost)
            distance_to_closest_ghost, _ = astar_search(current_game_state, new_pos, closest_ghost)

            score += 50 / distance_to_closest_ghost
        elif new_capsules and non_scared_ghosts:
            closest_capsule = min(new_capsules, key=lambda x: manhattan_distance(new_pos, x))
            distance_to_closest_capsule, _ = astar_search(current_game_state, new_pos, closest_capsule)
            score += 20 / distance_to_closest_capsule
        else:
            closest_food = bfs(successor_game_state, bfs_has_food)
            score += 1 / closest_food

        for ghost in non_scared_ghosts:
            closest_ghost = manhattan_distance(new_pos, ghost.getPosition())
            if closest_ghost <= 1:
                return float("-inf")
            elif closest_ghost < 4:
                score -= 400. / closest_ghost

        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search game
    (not reflex game).
    """
    return current_game_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search game.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="score_evaluation_function", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.nodes = 0

    def show_stats(self, numGames):
        print(f"Average of nodes per game : {self.nodes / numGames}")


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of game in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(game_state, self.depth, 0)[0]

    def minimax(self, game_state, currentDepth, currentAgent):
        self.nodes += 1

        if game_state.isWin() or game_state.isLose() or currentDepth == 0:
            return None, self.evaluationFunction(game_state)

        num_agents = game_state.getNumAgents()
        # Maximizing player
        if currentAgent == 0:
            bestValue = float("-inf")
            bestAction = None
            for action in game_state.getLegalActions(currentAgent):
                next_game_state = game_state.generateSuccessor(currentAgent, action)
                next_agent = (currentAgent + 1) % num_agents
                next_depth = currentDepth - (1 if next_agent == 0 else 0)

                value = self.minimax(next_game_state, next_depth, next_agent)[1]
                if value > bestValue:
                    bestValue = value
                    bestAction = action

        # Minimizing player (Ghosts)
        else:
            bestValue = float("inf")
            bestAction = None
            for action in game_state.getLegalActions(currentAgent):
                next_game_state = game_state.generateSuccessor(currentAgent, action)
                next_agent = (currentAgent + 1) % num_agents
                next_depth = currentDepth - (1 if next_agent == 0 else 0)

                value = self.minimax(next_game_state, next_depth, next_agent)[1]
                if value < bestValue:
                    bestValue = value
                    bestAction = action
        return bestAction, bestValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, evalFn="score_evaluation_function", depth="2"):
        super(AlphaBetaAgent, self).__init__(evalFn, depth)
        self.cuts = 0

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(game_state, self.depth, 0, float("-inf"), float("inf"))[0]

    def alphabeta(self, game_state: GameState, depth, current_agent, alpha, beta):
        self.nodes += 1

        if game_state.isWin() or game_state.isLose() or depth == 0:
            eval_value = self.evaluationFunction(game_state)
            return None, eval_value

        num_agents = game_state.getNumAgents()
        # Maximizing player (Pacman)
        if current_agent == 0:
            best_value = float("-inf")
            actions = game_state.getLegalActions(current_agent)
            for action in actions:
                next_game_state = game_state.generateSuccessor(current_agent, action)
                next_agent = (current_agent + 1) % num_agents
                next_depth = depth - (1 if next_agent == 0 else 0)

                _, value = self.alphabeta(
                    next_game_state, next_depth, next_agent, alpha, beta
                )
                if value > best_value:
                    best_value, best_action = value, action
                alpha = max(alpha, value)
                if best_value >= beta:
                    self.cuts += 1
                    break

        # Minimizing player (Ghosts)
        else:
            best_value = float("inf")
            actions = game_state.getLegalActions(current_agent)
            for action in actions:
                next_game_state = game_state.generateSuccessor(current_agent, action)
                next_agent = (current_agent + 1) % num_agents
                next_depth = depth - (1 if next_agent == 0 else 0)

                _, value = self.alphabeta(
                    next_game_state, next_depth, next_agent, alpha, beta
                )
                if value < best_value:
                    best_value, best_action = value, action
                beta = min(beta, value)
                if best_value <= alpha:
                    self.cuts += 1
                    break

        return best_action, best_value

    def show_stats(self, numGames):
        super().show_stats(numGames)
        print(f"Average of cuts per game : {self.cuts / numGames}")


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, self.depth, 0)[0]

    def expectimax(self, game_state, currentDepth, currentAgent):
        self.nodes += 1

        if game_state.isWin() or game_state.isLose() or currentDepth == 0:
            return None, self.evaluationFunction(
                game_state
            )

        num_agents = game_state.getNumAgents()
        # Maximizing player
        if currentAgent == 0:
            bestValue = float("-inf")
            bestAction = None
            for action in game_state.getLegalActions(currentAgent):
                next_game_state = game_state.generateSuccessor(currentAgent, action)
                next_agent = (currentAgent + 1) % num_agents
                next_depth = currentDepth - (1 if next_agent == 0 else 0)

                value = self.expectimax(next_game_state, next_depth, next_agent)[1]
                if value > bestValue:
                    bestValue = value
                    bestAction = action

        # Minimizing player (Ghosts)
        else:
            bestValue = 0
            bestAction = None
            allMoves = game_state.getLegalActions(currentAgent)
            probability = 1.0 / len(allMoves)
            for action in allMoves:
                next_game_state = game_state.generateSuccessor(currentAgent, action)
                next_agent = (currentAgent + 1) % num_agents
                next_depth = currentDepth - (1 if next_agent == 0 else 0)

                value = self.expectimax(next_game_state, next_depth, next_agent)[1]
                bestValue += probability * value

        return bestAction, bestValue


def betterEvaluationFunction(current_game_state: GameState):
    coeff_win = 100000
    coeff_ghost = 180
    coeff_capsule = 10
    coeff_food = 1

    score = current_game_state.getScore()
    if current_game_state.isWin():
        return coeff_win + score
    elif current_game_state.isLose():
        return -coeff_win - score

    new_pos = current_game_state.getPacmanPosition()
    new_capsules = current_game_state.getCapsules()

    score -= (coeff_capsule + 1) * len(new_capsules)

    new_ghost_states = current_game_state.getGhostStates()

    scared_ghost = [ghost for ghost in new_ghost_states if ghost.scaredTimer > 0]
    non_scared_ghost = [ghost for ghost in new_ghost_states if ghost.scaredTimer == 0]

    if scared_ghost:
        closest_ghost = (min(scared_ghost, key=lambda x: manhattan_distance(new_pos, x.getPosition()))).getPosition()
        closest_ghost = util.nearest_point(closest_ghost)
        distance_to_closest_ghost, _ = astar_search(current_game_state, new_pos, closest_ghost)

        score += coeff_ghost / distance_to_closest_ghost
    elif new_capsules and new_ghost_states:
        closest_capsule = min(new_capsules, key=lambda x: manhattan_distance(new_pos, x))
        distance_to_closest_capsule, _ = astar_search(current_game_state, new_pos, closest_capsule)
        score += coeff_capsule / distance_to_closest_capsule
    else:
        if current_game_state.getNumFood() < 10:
            new_food = current_game_state.getFood().asList()
            closest_food = min(new_food, key=lambda x: manhattan_distance(new_pos, x))
            distance_to_closest_food, _ = astar_search(current_game_state, new_pos, closest_food)
        else:
            distance_to_closest_food = bfs(current_game_state, bfs_has_food)
        score += coeff_food / distance_to_closest_food

    if non_scared_ghost:
        closest_non_scared_ghost = min(non_scared_ghost, key=lambda x: manhattan_distance(new_pos, x.getPosition()))
        dist = manhattan_distance(new_pos, closest_non_scared_ghost.getPosition())
        if dist <= 2:
            if dist < 1:
                return -100_000
            score -= 100.0 / dist

    return score

better = betterEvaluationFunction
