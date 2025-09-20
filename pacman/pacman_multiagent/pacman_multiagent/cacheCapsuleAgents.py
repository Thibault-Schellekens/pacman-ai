from pacman import GameState
from multiAgents import MultiAgentSearchAgent
from util import manhattan_distance
from multiAgentsHelper import astar_search


class CapsuleAStarCache:
    def __init__(self):
        self.path = []
        self.target_capsule = None

    def update_path(self, gameState: GameState):
        pacman_pos = gameState.getPacmanPosition()
        capsules = gameState.getCapsules()

        # Only recalculate when we need to (no path or target capsule eaten)
        if not self.path or self.target_capsule not in capsules:
            if capsules:
                self.target_capsule = min(capsules, key=lambda x: manhattan_distance(pacman_pos, x))
                _, self.path = astar_search(gameState, pacman_pos, self.target_capsule, 2)
            else:
                self.target_capsule = None
                self.path = []

    def get_next_move(self):
        if self.path:
            return self.path.pop(0)

        return None


class MinimaxCacheCapsuleAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='score_evaluation_function', depth=2):
        super().__init__(evalFn, depth)
        self.cc = CapsuleAStarCache()
        self.nodes = 0

    def get_action(self, game_state):

        in_danger = self.is_ghost_close(game_state)

        if not self.cc.path and not in_danger:
            self.cc.update_path(game_state)

        if self.cc.target_capsule is None or in_danger:
            # Clear path for next action ?
            self.cc.path.clear()
            return self.minimax(game_state, 0, 0)[0]
        else:
            # There is no danger, follow the path
            return self.cc.get_next_move()


    def minimax(self, game_state, currentDepth, currentAgent):
        self.nodes += 1

        if game_state.isWin() or game_state.isLose() or currentDepth == self.depth:
            return None, self.evaluationFunction(game_state)

        if currentAgent == 0:  # Pacman (Maximizing)
            bestValue = float("-inf")
            bestAction = None
            for action in game_state.getLegalActions(currentAgent):
                next_game_state = game_state.generateSuccessor(currentAgent, action)
                next_agent = currentAgent + 1
                next_depth = currentDepth

                if next_agent == game_state.getNumAgents():  # If last agent, increase depth
                    next_agent = 0
                    next_depth += 1

                value = self.minimax(next_game_state, next_depth, next_agent)[1]
                if value > bestValue:
                    bestValue = value
                    bestAction = action

        else:  # Ghosts (Minimizing)
            bestValue = float("inf")
            bestAction = None
            for action in game_state.getLegalActions(currentAgent):
                next_game_state = game_state.generateSuccessor(currentAgent, action)
                next_agent = currentAgent + 1
                next_depth = currentDepth

                if next_agent == game_state.getNumAgents():
                    next_depth += 1
                    next_agent = 0

                value = self.minimax(next_game_state, next_depth, next_agent)[1]
                if value < bestValue:
                    bestValue = value
                    bestAction = action

        return bestAction, bestValue

    def is_ghost_close(self, game_state : GameState):
        pacman_pos = game_state.getPacmanPosition()
        non_scared_ghost = [ghost for ghost in game_state.getGhostStates() if ghost.scaredTimer == 0]
        if not non_scared_ghost:
            return False

        dist_to_closest_ghost = min([manhattan_distance(pacman_pos, ghost.getPosition()) for ghost in non_scared_ghost])

        return dist_to_closest_ghost <= 5


