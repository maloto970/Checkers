import tensorflow as tf
import numpy as np

class ANNPlayer:
    

    def __init__(self, player, model):
        self.player = player
        self.model = model
    
    def getMove(self, game):
        
        moves = game.available_moves(self.player)
        best = 0
        move = moves[0]
        for m in moves:
            alt_game = game.copy_game()
            alt_game.tryMove(move, True)
            game_rep = self.preprocess(alt_game.board)
            rating = self.model.move(game_rep,self.player)
            if rating > best:
                move = m
                best = rating
        self.model.choose_move(best, tf.convert_to_tensor([self.player], dtype=tf.float32))
        return move


    def preprocess(self,board):
        
        new_board = [[] for r in board]
        for r,row in enumerate(board):
            for c in row:
                if c == 0:
                    new_board[r].append(0)
                    continue
                new_board[r].append(c.player if self.player == 1 else (c.player+1)%2 * (-1.0 if c.queen else 1.0))
        return tf.cast(tf.reshape(new_board, (1,8,8,1)),tf.float32)