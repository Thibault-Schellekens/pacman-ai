from pacman import GameState, Directions
from util import manhattan_distance


def scoreMoves(gameState: GameState, moves, currentAgent):
    if currentAgent == 0:
        return [scorePacmanMove(gameState, move) for move in moves]
    else:
        return [scoreGhostMove(gameState, move, currentAgent) for move in moves]

def scorePacmanMove(gameState: GameState, move):
    successor = gameState.generateSuccessor(0, move)

    score = successor.getScore()
    pacman_pos = successor.getPacmanPosition()
    new_ghosts = successor.getGhostStates()
    new_capsules = successor.getCapsules()
    distance_to_closest_scared_ghost = min(
        [manhattan_distance(pacman_pos, ghost.getPosition()) for ghost in new_ghosts if ghost.scaredTimer > 0] or [
            float("inf")])

    distance_to_closest_capsule = min(
        [manhattan_distance(pacman_pos, capsule) for capsule in new_capsules] or [float("inf")])

    score += (100 / distance_to_closest_scared_ghost) + (20 / distance_to_closest_capsule)

    if move == Directions.STOP:
        score -= 100

    return move, score, successor


def scoreGhostMove(gameState: GameState, move, currentAgent):
    successor = gameState.generateSuccessor(currentAgent, move)

    score = -successor.getScore()
    pacman_pos = successor.getPacmanPosition()
    ghost_state = successor.getGhostState(currentAgent)
    ghost_pos, scared_timer = ghost_state.getPosition(), ghost_state.scaredTimer
    distance_to_pacman = manhattan_distance(pacman_pos, ghost_pos)
    if scared_timer > 0:
        score += distance_to_pacman
    else:
        score -= distance_to_pacman

    return move, score, successor


def pickMove(moves, start_index):
    for i in range(start_index + 1, len(moves)):
        if moves[i][1] > moves[start_index][1]:
            moves[start_index], moves[i] = moves[i], moves[start_index]
