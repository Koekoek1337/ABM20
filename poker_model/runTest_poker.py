import numpy as np

from game_classes.game import Game
from game_classes.agents import AuctionPlayer
from plotting.plotting import plotWealth

rng = np.random.default_rng(42)

nPlayers = 5

players  = [AuctionPlayer(rng, "null") for _ in range(nPlayers)]
for player in players[1:]:
    player.set_strategy(players[0].getData()[0])

game = Game(players, rng)

data = game.play(1000, 100)

print(data[0][0])
plotWealth(data)