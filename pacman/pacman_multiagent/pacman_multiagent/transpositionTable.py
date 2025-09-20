from pacman import GameState
from collections import namedtuple, OrderedDict

"""
Implementation of the Transposition table inspired by: https://github.com/duilio/c4/blob/master/c4/cache.py
"""

Entry = namedtuple('Entry', 'move depth score state')

class TranspositionTable(object):
    EXACT = object()
    LOWERBOUND = object()
    UPPERBOUND = object()

    def __init__(self, maxitems=50000):
        self.__maxitems = maxitems
        self.__table = OrderedDict()
        self.clears = 0

    def put(self, gameState, moves, depth, score, alpha=float("-inf"), beta=float("inf")):
        key = hash(gameState)
        if moves:
            move = moves[0]
        else:
            move = None

        if depth == 0 or depth == -1 or alpha < score < beta:
            state = TranspositionTable.EXACT
        elif score >= beta:
            state = TranspositionTable.LOWERBOUND
            score = beta
        elif score <= alpha:
            state = TranspositionTable.UPPERBOUND
            score = alpha
        else:
            assert False

        entry = Entry(move, depth, score, state)
        self.__table.pop(key, None)
        self.__table[key] = entry

        if len(self.__table) >= self.__maxitems:
            self.__table.popitem(last=False)
            self.clears += 1

    def lookup(self, gameState, depth, alpha=float("-inf"), beta=float("inf")):
        key = hash(gameState)
        if key not in self.__table:
            return False, None, None

        entry = self.__table[key]

        hit = False
        if entry.depth == -1:
            hit = True
        elif entry.depth >= depth:
            if entry.state == TranspositionTable.EXACT:
                hit = True
            elif entry.state == TranspositionTable.LOWERBOUND and entry.score >= beta:
                hit = True
            elif entry.state == TranspositionTable.UPPERBOUND and entry.score <= alpha:
                hit = True

        move = entry.move

        if hit:
            score = entry.score
        else:
            score = None

        return hit, move, score


    def clear(self):
        self.__table.clear()

    def getLen(self):
        current_size = len(self.__table)
        occupancy_rate = (current_size / self.__maxitems) * 100
        return f"Entries: {current_size}, Occupancy: {occupancy_rate:.2f}%"