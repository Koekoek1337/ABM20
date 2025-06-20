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

class IterativeCompetition(mesa.Model):
    def __init__(self, nPlayers, agentUpdateMode="null", updateParms={"maxStepsize":0.1}, *args, 
                 seed = None, rng = None, **kwargs):
        super().__init__(*args, seed=seed, rng=rng, **kwargs)

        AuctionPlayer.create_agents(self, nPlayers, updateMode=agentUpdateMode, updateParms=updateParms)
        self.game = Game([player for player in self.agents], self.rng)