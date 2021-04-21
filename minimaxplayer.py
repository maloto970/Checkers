import copy


class MinimaxPlayer:
    def __init__(self, player):
        self.player = player

    def getMove(self, game):
        rating, move = self.minmax(game, self.player, 0)
        print("Found move with rating " + str(rating))
        return move

    def minmax(self, game, player, depth):
        if depth == 0:
            print("hej")
        over, win = game.over()
        
        if over:
            win = True if str(self.player) in win else False
            if win:
                return 10000, None
            else:
                return -10000, None
        
        if depth == 1:
            rating = self.rate(game)
            return rating, None

        moves = game.available_moves(player)
        best_move = None

        if player == self.player:
            best = -10000
        else:
            best = 10000
        
        next_player = (player + 1) % 2
        depth = depth + 1
        alt_game = None

        for move in moves:
            alt_game = game.copy_game()
            alt_game.tryMove(move, True)
            rating, _ = self.minmax(alt_game, next_player, depth)

            if player == self.player:
                if rating > best:
                    best = rating
                    best_move = move
            else:
                if rating < best:
                    best = rating
                    best_move = move
                    
        return best, best_move

    def rate(self, game):
        pieces = game.get_pieces()
        p_pieces, o_pieces = pieces[self.player-1], pieces[(self.player)%2]
        diff = len(p_pieces) - len(o_pieces)

        return diff