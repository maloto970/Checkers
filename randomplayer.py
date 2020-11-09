import math
import random

class RandomPlayer:
    def __init__(self, player):
        self.player = player

    def getMove(self, game):
        moves = game.available_moves(self.player)
        return moves[random.randint(0,len(moves)-1)] 