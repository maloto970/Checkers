import math

class KeyboardPlayer:
    def __init__(self, player):
        self.player = player

    def getMove(self, game):
        click1 = game.window.getMouse()
        click2 = game.window.getMouse()
        x,y = click1.x, click1.y
        x_t,y_t = click2.x, click2.y
        move = [
            math.floor(y/game.tile_width),
            math.floor(x/game.tile_height),
            math.floor(y_t/game.tile_width),
            math.floor(x_t/game.tile_height)
        ]
        return move 