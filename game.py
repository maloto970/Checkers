
from minimaxplayer import MinimaxPlayer
from randomplayer import RandomPlayer
from deepcheckers import DeepCheckersHandler
from keyboardplayer import *
from randomplayer import *
from minimaxplayer import *
from annplayer import *
from deepcheckers import *
#from graphics import *
import math
import numpy as np
import time
import tensorflow as tf
import datetime

class Tile:
    def __init__(self, player, brick, id):
        self.player = player
        self.brick = brick
        self.queen = False
        self.id = id
        

    def __copy__(self):
        c = Tile(self.player, self.brick, self.id)
        c.queen = self.queen
        return c
    def __deepcopy__(self, memo):
        c = Tile(self.player, self.brick, self.id)
        c.queen = self.queen
        return c


class Game:
    def __init__(self, gui, win_height, win_width):
        self.win_height = win_height
        self.win_width = win_width
        self.gui = gui
        self.turn = 0
        self.board = [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 2],
            [2, 0, 2, 0, 2, 0, 2, 0],
            [0, 2, 0, 2, 0, 2, 0, 2]
        ]
        """self.board = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]"""

        
        self.p1_pieces = 0
        self.p2_pieces = 0
        self.repetition = [False, False]
        self.last_moves = [[],[]]

        for r in self.board:
            for c in r:
                if c == 1:
                    self.p1_pieces += 1
                if c == 2:
                    self.p2_pieces += 1

        if self.gui:
            self.window = GraphWin('Checkers', self.win_width, self.win_height)
            self.bricks = [0]*8**2
            self.tile_height = self.win_height/8
            self.tile_width = self.win_width/8
            board_copy = copy.deepcopy(self.board)
            count = 0
            for r_n, row in enumerate(board_copy):
            	y = r_n * self.tile_height
            	y_end = y + self.tile_height
            	for c_n, col in enumerate(self.board):
                    x = c_n * self.tile_width
                    x_end = x + self.tile_width
                    Rectangle(Point(x, y), Point(
                        x_end, y_end)).draw(self.window)
                    tile_content = board_copy[r_n][c_n]
                    if tile_content == 0:
                        continue
                    elif tile_content == 1:
                    	color = "red"
                    elif tile_content == 2:
                        color = "black"
                    marker = Circle(Point((x+x_end)/2, (y+y_end)/2),
                                    0.5*0.45*(self.tile_height+self.tile_width))
                    marker.setFill(color)
                    marker.draw(self.window)
                    prev = board_copy[r_n][c_n]
                    board_copy[r_n][c_n] = Tile(prev, marker, count)
                    count = count + 1
            self.board = copy.deepcopy(board_copy)
            del board_copy
        else: 
            count = 0
            board_copy = copy.deepcopy(self.board)
            for r_n, row in enumerate(board_copy):
                for c_n, col in enumerate(self.board):
                    if board_copy[r_n][c_n] == 0:
                        continue
                    prev = board_copy[r_n][c_n]
                    board_copy[r_n][c_n] = Tile(prev, None, count)
                    count = count + 1
            self.board = copy.deepcopy(board_copy)
            del board_copy



    def copy_game(self):
        c = type(self)(False,self.win_height,self.win_width)
        c.turn = self.turn
        c.p1_pieces = self.p1_pieces
        c.p2_pieces = self.p2_pieces
        c.last_moves = copy.deepcopy(self.last_moves)
        c.repetition = copy.deepcopy(self.repetition)
        c.board = copy.deepcopy(self.board)
        return c

    def getBoard(self):
        return self.board, self.window

    def showBoard(self):
        self.printBoard()

    def printBoard(self):
        for r in self.board:
            row = ""
            for c in r:
                if isinstance(c, Tile):
                    row += str(c.player) + " "
                else:
                    row += str(0) + " "
            print(row)

    def tryMove(self, move, real):
        piece = self.board[move[0]][move[1]]

        if not(isinstance(piece, Tile)):
            return False
        if piece.player-1 != self.turn:
            return False

        delta_x = move[3]-move[1]
        delta_y = move[2]-move[0]

        if delta_x == 0 and delta_y == 0:
            return False

        if self.turn == 0:
            direction = 1
        elif self.turn == 1:
            direction = -1

        if ((direction == 1 and delta_y < 0) or (direction == -1 and delta_y > 0)) and not(piece.queen):
            return False
        
        if self.board[move[2]][move[3]] != 0:
            return False

        if abs(delta_y) == abs(delta_x) and abs(delta_y) == 1:
            pass
        elif abs(delta_y) == abs(delta_x) and abs(delta_y) == 2:
            if not(self.capture([[int(delta_y/2) + move[0], int(delta_x/2) + move[1]]], False, real, [move[0], move[1]])):
                return False
        else:
            # check for multi hop capture
            res, jump = self.multiple_hops(
                [move[0], move[1]], 
                [move[2], move[3]],
                piece.queen
            )
            if not(res):
                return False
            if real:
                captures = [jump[i] for i in range(len(jump)) if i%2 == 1]
                self.capture(captures, True, real)

        if real:
            self.board[move[0]][move[1]] = 0
            self.board[move[2]][move[3]] = piece
            
            if self.gui:
                brick = piece.brick
                x = (move[3]-move[1])*self.tile_width
                y = (move[2]-move[0])*self.tile_height
                brick.move(x, y)
            if (move[2] == 0 and self.turn == 1) or (move[2] == 7 and self.turn == 0):
                piece.queen = True
                if self.gui:
                    color = "orange" if self.turn == 0 else "gray"
                    piece.brick.setFill(color)

            self.last_moves[self.turn].append(move)
            if len(self.last_moves[self.turn]) > 5:
                del self.last_moves[self.turn][0]
            
            self.turn = (self.turn + 1) % 2

        return True

    def capture(self, pieces, multi_hop, real, *_from):
        
        
        if multi_hop:
            valid_captures = pieces.copy()
        else:
            _from = np.array(_from[0].copy())
            pieces = np.array(pieces.copy())
            valid_captures = []
            deltas = pieces - _from
            end_tiles = _from + deltas*2
            for i, piece in enumerate(pieces):
                end_tile = end_tiles[i]
                capture_piece = self.board[piece[0]][piece[1]]
                if isinstance(capture_piece,Tile):
                    if capture_piece.player-1 != self.turn and self.board[end_tile[0]][end_tile[1]] == 0:
                        valid_captures.append(piece)
                    else:
                        return False
                else:
                    return False

        	
        for vc in valid_captures:
            if not(real):
                break
            if self.gui:
                self.board[vc[0]][vc[1]].brick.move(self.win_height,self.win_width)
            self.board[vc[0]][vc[1]] = 0
            
            if self.turn == 0:
            	self.p2_pieces = self.p2_pieces - 1
            elif self.turn == 1:
            	self.p1_pieces = self.p1_pieces - 1
        return True

    def multiple_hops(self, origin, goal, queen):
        return self.search_hops(origin, goal, None, goal, [], [], queen)

    def search_hops(self, origin, curr, prev, goal, visited, path, queen):
        
        visited.append(curr)
        path.append(curr)
        
        if origin == curr and self.board[prev[0]][prev[1]] != 0:
            return True, path
        
        # Have we reached the origin? Is this tile occupied or not and should it be?
        tile = self.board[curr[0]][curr[1]]
        if len(path) % 2 == 0:
            if tile != 0:
                if tile.player-1 == self.turn:
                    path = path[:-1]
                    return False, None
            else:
                path = path[:-1]
                return False, None
        else:
            if tile != 0:
                path = path[:-1]
                return False, None

		# Not visited diagonals in a valid direction for current piece
        diagonals = self.get_diagonals(curr, prev, visited, queen, tile==0)
        if len(diagonals) == 0:
            return False, None
        res = []
        for d in diagonals:
            res, p = self.search_hops(origin, d, curr, goal, visited, path.copy(), queen)
            if res:
                return res, p
        return False, None


    def get_diagonals(self, curr, prev, visited, queen, branching):
        diags = []
        dirs = [np.array([1,1]), np.array([1,-1]), np.array([-1, 1]), np.array([-1,-1])]
        # We've landed next to a captured piece looking for the next one
        if self.turn == 1 and not(queen):
            dirs = [np.array([1,1]), np.array([1,-1])]
        if self.turn == 0 and not(queen):
            dirs = [np.array([-1, 1]), np.array([-1,-1])]
        if not(branching):
            delta = np.array(curr)-np.array(prev)
            d_l = [list(a) for a in dirs]
            if list(delta) not in d_l:
                return diags
            dirs = [delta]
        curr = np.array(curr.copy())
        
        
        for d in dirs:
            tile = list(d + curr)
            if tile in visited:
                continue
            if tile[0] < 0 or tile[1] < 0 or tile[0] > 7 or tile[1] > 7:
                continue
            diags.append(tile)
        return diags

    def get_pieces(self):
        pieces = [[],[]]
        for i in range(8):
            for j in range(8):
                tile = self.board[i][j]
                if tile != 0:
                    pieces[tile.player-1].append([i,j])
        return pieces

    def available_moves(self, player):
        pieces = self.get_pieces()[player-1]
        moves = []
        for piece in pieces:
            for r in range(8):
                for c in range(8):
                    move = [piece[0], piece[1], r, c]
                    if self.tryMove(move, False):
                        moves.append(move)
        return moves



    def get_turn(self):
        return self.turn

    def over(self):
        if len(self.last_moves[(self.turn + 1) % 2]) >= 5:
            if len(set(["".join(map(str,m)) for m in self.last_moves[(self.turn + 1) % 2]])) == 2:
                self.repetition[(self.turn + 1) % 2] = True
                
        if self.p1_pieces == 0 or self.repetition[0]:
            return True,"Player 2 wins"
        elif self.p2_pieces == 0 or self.repetition[1]:
            return True,"Player 1 wins"
        elif self.available_moves(self.turn+1) == []:
            return True,"Player " + str(2 if self.turn==0 else 1) + " wins"
        else:
            return False, ""

def play_game(p1, p2, gui):
    if isinstance(p1, KeyboardPlayer) or isinstance(p2, KeyboardPlayer):
        gui = True
    players = [p1,p2]
    action = None
    game = Game(gui,550,550)
    game_over, status = game.over()
    
    turns = 0
    while not(game_over):
        t = game.get_turn()
        #print("Turn " + str(t))
        player = players[t]
        while True:
            #game.showBoard()
            action = player.getMove(game)
            if game.tryMove(action, True):
                break
            
        game_over, status = game.over()
        turns = turns + 1
        #time.sleep(1)
    
    print(status)
    return 1 if "1" in status else 2, turns

def main():
    checkers_model = DeepCheckersHandler()
    if os.path.isfile(os.getcwd() + "/modelweights_move_mc.ckpt.data-00000-of-00001"):
        checkers_model.model.built = True
        checkers_model.model.load_weights(os.getcwd() + "/modelweights_move_mc.ckpt")
        print("Loaded weights")

    test_p1_loss = tf.keras.metrics.Mean('p1_random_losses', dtype=tf.float32)
    test_p2_loss = tf.keras.metrics.Mean('p2_random_losses', dtype=tf.float32)
    test_writer = tf.summary.create_file_writer(os.getcwd() + '/logs/test')
    
    p1 = ANNPlayer(1,checkers_model)
    p2 = ANNPlayer(2,checkers_model)
    validation_player1 = RandomPlayer(1) 
    validation_player2 = RandomPlayer(2)

    test_epoch = 0
    turns = 0

    for g in range(100000):
        print("Game " + str(g))
        if g % 20 == 0:
            p1.train_mode = False
            p2.train_mode = False

            winner1 = []
            winner2 = []
            for i in range(10):
                res1,_ = play_game(p1,validation_player2,False)
                res2,_ = play_game(validation_player1, p2,False)
                winner1.append(0 if res1 == 1 else 1)
                winner2.append(0 if res2 == 2 else 1)
            test_p1_loss(winner1)
            test_p2_loss(winner2)

            with test_writer.as_default():
                tf.summary.scalar('p1_random_losses', test_p1_loss.result(), step=test_epoch)
                tf.summary.scalar('p2_random_losses', test_p2_loss.result(), step=test_epoch)
                tf.summary.scalar('game_train_turns', turns/20.0, step=test_epoch)
                turns = 0
            
            p1.train_mode = True
            p2.train_mode = True

            test_epoch += 1
           

        w, t = play_game(p1,p2,False)
        turns += t
        p1.game_nr += 1
        p2.game_nr += 1
        p1.result(1 if w == 1 else -1)
        p2.result(1 if w == 2 else -1)



if __name__ == "__main__":
    main()
