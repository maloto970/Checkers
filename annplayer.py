from deepcheckers import DeepCheckersHandler
from deepcheckers import DeepCheckers
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import Model 
import random

class ANNPlayer:
    
    def __init__(self, player, checkers_model):
        self.player = player
        self.model = checkers_model
        #self.target_model = DeepCheckersHandler()
        #if os.path.isfile(os.getcwd() + "/modelweights.ckpt.data-00000-of-00001"):
        #    checkers_model.model.built = True
        #    self.target_model.model.load_weights(os.getcwd() + "/modelweights.ckpt")
        self.gamma = 0.8
        self.iterations = 0
        self.train_mode = True
        self.game_nr = 0
        self.batch_X = []
        self.batch_m = []
        self.batch_t = []
        self.batch_last_five = []

    def choose_move(self, board, move):
        # Input, Target output, Predicted output
        self.batch_X.append(board)
        self.batch_m.append(move)
        self.batch_last_five.append(self.last_five(False))

    def result(self, won):

        rewards = [0 for i in range(len(self.batch_X)-1)]
        rewards.append(won)
        for j in range(len(self.batch_X)-1):
            G = 0
            for i,r in enumerate(rewards[j:]):
                G += r*self.gamma**i
            self.batch_t.append(G)
        
        self.batch_t.append(won)
        self.model.append_batch(self.batch_X,self.batch_m,self.batch_last_five,self.batch_t)
        self.batch_X = []
        self.batch_m = []
        self.batch_t = []
        self.batch_last_five = []

    def getMove(self, game):

        if self.train_mode:
            # Freeze target model during 20 games
            #if self.game_nr % 200 == 0 and os.path.isfile(os.getcwd() + "/modelweights.ckpt.data-00000-of-00001"):
            #    self.target_model.model.load_weights(os.getcwd() + "/modelweights.ckpt")
            #    print("Updated target net after last save")
            if len(self.model.batch_X) >= 1500:
                self.model.train_on_batch()

        moves = game.available_moves(self.player)
        best = -100
        move = moves[0].copy()
        board = None
        explore = False

        if self.train_mode:
            explore = True if random.randint(0,100) < 7 else False #80/min(80,max(1,self.iterations/(40*2000))) else False

        if explore:
            move = moves[random.randint(0,len(moves)-1)]
            """
            alt_game = game.copy_game()
            alt_game.tryMove(move, True)
            """
            alt_game = game.copy_game()
            game_rep, _ = self.preprocess(alt_game.board)
            board = game_rep
            #best = self.model.move(game_rep, move)
        else:
            alt_game = game.copy_game()
            game_rep, _ = self.preprocess(alt_game.board)
            for m in moves:
                rating = self.model.move(game_rep,[m],self.last_five(True))
                if rating > best:
                    board = game_rep
                    move = m.copy()
                    best = rating

        if self.train_mode == True:
            self.choose_move(board,move)
            self.iterations += 1
        
        return move

    def last_five(self, batch):
        i = max(0,len(self.batch_m) - 1)
        offset = 5-i
        five = []
        for k in range(offset):
            five.extend([0,0,0,0])
        low = max(0,i-5)
        high = max(0,i)
        for ind in range(low,high):
            five.extend(self.batch_m[ind])

        return five if not(batch) else [five]


    def Q_target_value(self, game):
        
        gameOver, winner = game.over()
        winner = 1 if "1" in winner else 2
        if gameOver:
            v = 1 if winner==self.player else 0
            return tf.convert_to_tensor([v*1.0])

        moves = game.available_moves(2 if self.player == 1 else 1)
        best = 0
        for m in moves:
            alt_game = game.copy_game()
            alt_game.tryMove(m, True)
            game_rep, _ = self.preprocess(alt_game.board)
            rating = self.target_model.move(game_rep)
            if rating > best:
                best = rating

        return self.gamma * best

    def preprocess(self,board,reverse=False):
        new_board = [[] for r in board]
        for r,row in enumerate(board):
            for c in row:
                if c == 0:
                    new_board[r].append(0)
                    continue
                player = max(1,(c.player+1)%3) if reverse else c.player
                new_board[r].append((player if self.player == 1 else (1 if player==2 else 2)) * (-1.0 if c.queen else 1.0))
        return tf.cast(tf.reshape(new_board, (1,8,8,1)),tf.float32), new_board