import numpy as np
import scipy as sp
import PokerPy

from typing import Tuple, List, Dict, Any

RAISELIM = 0
CALL_LIM = 1

MAX_BET = 400

class AuctionPlayer:
    """ Poker Playing agent using "poker-as-an-auction" model

    Players raise up to ther betlim and 
    
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
    
    def __init__(self, rng: np.random.Generator, updateMode = "uniform", updateParms={"maxStepsize": 10}):
        self.RNG = rng

        self.strategy     = np.zeros((2, 10), float)
        self.strategy[RAISELIM, :] = self.randomStrat(50, 150)
        self.strategy[CALL_LIM,:] = self.randomStrat(150, 200)
        self.balances     = np.zeros(10, float)

        self.handIndex   = None
        
        self.updateMode  = updateMode
        self.updateParms = updateParms

    def randomStrat(self, minB = 50, maxB=200) -> np.ndarray:
        """
        Returns half of a (bad) random strategy
        """
        return self.RNG.uniform(minB, maxB, 10)

    def set_strategy(self, strategy: np.ndarray) -> None:
        """
        Strategy setter
        """
        self.strategy = strategy

    def getBet(self, hand: str) -> np.ndarray[float]:
        """
        Returns the bet corresponding to the hand

        Keeps track of played hand for record keeping between hands, when the result is returned
        """
        self.handIndex = self.HAND_TO_INDEX[hand]
        return self.strategy[:,self.handIndex]

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
        updateIndex = self.RNG.choice(self.strategy.shape[1])

        lowBet    = self.strategy[0, updateIndex] - maxStepsize
        highBet   = self.strategy[0, updateIndex] + maxStepsize

        lowRaise  = self.strategy[0, updateIndex] - maxStepsize
        highRaise = self.strategy[0, updateIndex] + maxStepsize

        # Clamp possible values between 0 and max
        if lowBet < 0: lowBet = 0 
        if highBet > MAX_BET: highBet = MAX_BET

        if lowRaise < 0: lowRaise = 0
        if highRaise > MAX_BET: highRaise = MAX_BET

        newStrat = np.asarray((self.RNG.uniform(lowBet, highBet), self.RNG.uniform(lowRaise, highRaise)))

        self.strategy[RAISELIM, updateIndex]   = np.max(newStrat)
        self.strategy[CALL_LIM, updateIndex] = np.min(newStrat)

    def _null(self, **kwp):
        """
        Nullstrat, do not change strategy, for testing.
        """
        return
