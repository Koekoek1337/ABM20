import numpy as np

from game_classes.game import Game
from game_classes.agents import AuctionPlayer

rng = np.random.default_rng(42)

nPlayers = 2

players  = [AuctionPlayer(rng, "null") for _ in range(nPlayers)]

players[1].set_strategy(players[0].getData()[0])


game = Game(players, rng)

game.play(10, 10000)