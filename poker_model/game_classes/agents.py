import mesa.discrete_space.cell_agent
import numpy as np
import mesa

from typing import Tuple, List, Dict, Any

RAISELIM   = 0
CALL_LIM   = 1
BLUFF_PROB = 2
BLUFF_SIZE = 3

MAX_BET = 400

HANDS = 10

class AuctionPlayer(mesa.discrete_space.FixedAgent):
    """ Poker Playing agent using "poker-as-an-auction" model

    Players raise up to their raiselim and call up to their call lim. 
    Agents have a chance to bluff, pulling an absolute bluff raise (with no call limit) from BLUFF_SIZE
    """
    #{Hand Indexes}#####################
    HAND_TO_INDEX = {
        "Royal Flush":    9,
        "Straight Flush": 8,
        "Poker":          7, # Four of a Kind
        "Full House":     6,
        "Flush":          5,
        "Straight":       4,
        "Triples":        3,
        "Double Pairs":   2,
        "Pairs":          1,
        "High Card":      0
    }

    def __init__(self, model, updateMode, updateParms, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        self.strategy     = self.randomStrat()
        self.balances     = np.zeros(HANDS, float)

        self.handIndex   = None
        
        self.updateMode  = updateMode
        self.updateParms = updateParms
    
    def step(self):
        self.model.tourney(self)

    def randomStrat(self, minR = 0, maxR=MAX_BET, minC=0, maxC=MAX_BET, maxB=MAX_BET, max_pB=MAX_BET) -> np.ndarray:
        """
        Returns half of a (bad) random strategy
        """
        strategy = np.zeros((4,HANDS), float)
        strategy[RAISELIM,  :] = self.rng.uniform(minR, maxR)
        strategy[CALL_LIM,  :] = self.rng.uniform(minC, maxC)
        strategy[BLUFF_PROB,:] = self.rng.uniform(0,    max_pB, HANDS)
        strategy[BLUFF_SIZE,:] = self.rng.uniform(0,    maxB,   HANDS)
        
        strategy = self.raiseCallOrder(strategy)
        return strategy
    
    def raiseCallOrder(self, strategy: np.ndarray) -> np.ndarray:
        """ 
        Ensures that raiselim <= calllim
        """
        raiseLim = np.min(strategy[RAISELIM:CALL_LIM+1, :], 0)
        call_Lim = np.max(strategy[RAISELIM:CALL_LIM+1, :], 0)
        strategy[RAISELIM,:] = raiseLim
        strategy[CALL_LIM,:] = call_Lim
        return strategy

    def set_strategy(self, strategy: np.ndarray) -> None:
        """
        Strategy setter
        """
        self.strategy = strategy.copy()

    def getBet(self, hand: str) -> np.ndarray[float]:
        """
        Returns the bet corresponding to the hand

        Keeps track of played hand for record keeping between hands, when the result is returned
        """
        self.handIndex = self.HAND_TO_INDEX[hand]
        pBluff = self.strategy[BLUFF_PROB, self.handIndex] / MAX_BET
        if self.rng.random() < pBluff:
            bluff = self.strategy[BLUFF_SIZE, self.handIndex]
            return np.asarray((bluff, bluff))
        return self.strategy[RAISELIM:CALL_LIM+1,self.handIndex]

    def handResult(self, balanceChange) -> None:
        """
        Get the result of the previous hand and update balances based on the hand gotten from getbet
        """
        self.balances[self.handIndex] += balanceChange

    def getData(self):
        """Data Dollection function"""
        return self.strategy.copy(), self.balances.copy()

    def gameEnd(self, loss: bool):
        """
        Handles the game end event
        """
        if loss:    
            if   self.updateMode =="uniform":
                self._upd_uniformRandom(**self.updateParms)
            elif self.updateMode == "null":
                pass
        self.balances = self.balances * 0

    def _upd_uniformRandom(self, maxStepsize, **kwp):
        """
        Uniform random step update function.
        Randomly chooses a point in the strategy to update, then samples the raise and bet points from
        an uniform distribution. If call ever is to go below raiselim, swap the values.

        Not a strong strategy, but easy to implement and a good boilerplate example
        """
        updateIndex = self.rng.choice(self.strategy.shape[1])

        ranges = [[self.strategy[i, updateIndex] - maxStepsize, 
                   self.strategy[i, updateIndex] + maxStepsize] 
                   for i in range(4)]

        for i, (mini, maxi) in enumerate(ranges):
            if mini < 0: ranges[i][0]       = 0
            if maxi > MAX_BET: ranges[i][1] = MAX_BET

        newStrat = np.copy(self.strategy)

        for i, (mini, maxi) in enumerate(ranges):
            newStrat[i, updateIndex] = self.rng.uniform(mini, maxi)

        newStrat = self.raiseCallOrder(newStrat)

        self.strategy = newStrat
        
    def _null(self, **kwp):
        """
        Nullstrat, do not change strategy, for testing.
        """
        return

class gridAgent(mesa.discrete_space.FixedAgent):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        self.playerAgent = AuctionPlayer(model.rng)
    
