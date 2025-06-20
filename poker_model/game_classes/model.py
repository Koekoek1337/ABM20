import numpy as np
import mesa

from game_classes.game   import Game
from game_classes.agents import AuctionPlayer

class spatialModel(mesa.Model):
    def __init__(self, *args, seed = None, rng = None, gridDim =(10,10), **kwargs):
        super().__init__(*args, seed=seed, rng=rng, **kwargs)
        self.gameInstance = Game([], self.rng)
        self.space = mesa.discrete_space.OrthogonalVonNeumannGrid(gridDim, True, 1, self.rng, AuctionPlayer)
        
    def tourney(self, agent):
        pass