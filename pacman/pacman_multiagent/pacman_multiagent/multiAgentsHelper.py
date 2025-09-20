from pacman import GameState
from game import Directions
from util import manhattan_distance, PriorityQueue, Queue


def astar_search(current_game_state, start, goal, alpha=2):

    def heuristic(pos1, pos2):
        return alpha * manhattan_distance(pos1, pos2)

    directions = {
        (-1, 0): Directions.WEST,
        (1, 0): Directions.EAST,
        (0, -1): Directions.SOUTH,
        (0, 1): Directions.NORTH
    }

    walls = current_game_state.getWalls()
    open_list = PriorityQueue()
    min_g_cost = {start: 0}
    came_from = {start: None}
    came_from_action = {start: None}

    open_list.push((start, 0), heuristic(start, goal))

    while not open_list.is_empty():
        current, cost = open_list.pop()

        if current == goal:
            path = []
            while came_from[current] is not None:
                path.append(came_from_action[current])
                current = came_from[current]
            path.reverse()

            return cost, path

        if current not in min_g_cost or cost <= min_g_cost[current]:
            min_g_cost[current] = cost

            x, y = current
            for (dx, dy), action in directions.items():
                next_pos = (x + dx, y + dy)
                if not walls[x + dx][y + dy]:
                    new_g = min_g_cost[current] + 1
                    if next_pos not in min_g_cost or new_g < min_g_cost[next_pos]:
                        f = new_g + heuristic(next_pos, goal)
                        open_list.push((next_pos, new_g), f)
                        came_from[next_pos] = current
                        came_from_action[next_pos] = action

    return float("inf"), []


def bfs(current_game_state, target):
    pacman_pos = current_game_state.getPacmanPosition()
    walls = current_game_state.getWalls()
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    # Queue stores (position, distance)
    q = Queue()
    q.push((pacman_pos, 0))

    visited = set()
    visited.add(pacman_pos)

    while not q.is_empty():
        cur_pos, dist = q.pop()

        if target(current_game_state, cur_pos):
            return dist

        for dx, dy in directions:
            new_pos = (cur_pos[0] + dx, cur_pos[1] + dy)

            if (
                    new_pos not in visited
                    and not walls[new_pos[0]][new_pos[1]]
            ):
                visited.add(new_pos)
                q.push((new_pos, dist + 1))

    return float("inf")

def simple_eval_bfs(currentGameState: GameState):
    score = currentGameState.getScore()
    if currentGameState.isWin():
        return 100000 + score  # Adding score to advantage fast wins
    elif currentGameState.isLose():
        return -100000 + score
    dist_closest_food = bfs(currentGameState, bfs_has_food)
    score += 10 / dist_closest_food
    return score


def simple_eval_base(currentGameState: GameState):
    score = currentGameState.getScore()
    if currentGameState.isWin():
        return 100000 + score
    elif currentGameState.isLose():
        return -100000 + score
    new_pos = currentGameState.getPacmanPosition()
    new_food = currentGameState.getFood().asList()
    dist_closest_food = min([manhattan_distance(new_pos, food) for food in new_food])
    score += 10 / dist_closest_food
    return score

def bfs_has_food(current_game_state: GameState, pos):
    return current_game_state.hasFood(pos[0], pos[1])

def bfs_has_scared_ghost(current_game_state: GameState, pos):
    for ghost in current_game_state.getGhostStates():
        if manhattan_distance(ghost.getPosition(), pos) <= 1 and ghost.scaredTimer > 0:
            return True
    return False


def bfs_has_non_scared_ghost(current_game_state: GameState, pos):
    for ghost in current_game_state.getGhostStates():
        if manhattan_distance(ghost.getPosition(), pos) <= 1 and ghost.scaredTimer == 0:
            return True
    return False

def bfs_has_capsule(current_game_state: GameState, pos):
    return pos in current_game_state.getCapsules()


def capsule_bfs_test(currentGameState: GameState):
    score = currentGameState.getScore()
    if currentGameState.isWin():
        return 100000 + score
    elif currentGameState.isLose():
        return -100000 + score
    dist_closest_capsule = bfs(currentGameState, bfs_has_capsule)
    score += 10 / dist_closest_capsule
    return score


def capsule_astar_test(currentGameState: GameState):
    score = currentGameState.getScore()
    if currentGameState.isWin():
        return 100000 + score
    elif currentGameState.isLose():
        return -100000 + score
    capsules = currentGameState.getCapsules()
    pacman_pos = currentGameState.getPacmanPosition()
    if capsules:
        closest_capsule = min(capsules, key=lambda x: manhattan_distance(pacman_pos, x))
        distance_to_closest_capsule, _ = astar_search(currentGameState, pacman_pos, closest_capsule)
    else:
        distance_to_closest_capsule = float('inf')
    score += 10 / distance_to_closest_capsule
    return score
