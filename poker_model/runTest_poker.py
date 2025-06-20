import numpy as np

from game_classes.model import IterativeCompetition
from game_classes.game import Game
from game_classes.agents import AuctionPlayer
from plotting.plotting import plotWealth

model = IterativeCompetition(5, "null", {"maxStepsize": 0.1})
data = model.game.play(1000, 10, False)

plotWealth(data)