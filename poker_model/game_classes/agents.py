# agents.py
import numpy as np
from mesa import Agent

RAISELIM   = 0
CALL_LIM   = 1
BLUFF_PROB = 2
BLUFF_SIZE = 3

MAX_BET = 400

HANDS = 10

class AuctionPlayer(Agent):
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

    def __init__(self, model, update_mode="null", update_parms=None, *args, **kwargs):
        super().__init__(model)
        self.update_mode = update_mode
        self.update_parms= update_parms or {}
        self.strategy    = self._random_strat()
        self.balances    = np.zeros(HANDS)
        self.hand_index  = 0
    
    def _random_strat(self):
        s = np.zeros((4, HANDS))
        s[RAISELIM]   = self.rng.uniform(0, MAX_BET, HANDS)
        s[CALL_LIM]   = self.rng.uniform(0, MAX_BET, HANDS)
        s[BLUFF_PROB] = self.rng.uniform(0, MAX_BET, HANDS)
        s[BLUFF_SIZE] = self.rng.uniform(0, MAX_BET, HANDS)
        return self._enforce_order(s)

    def _enforce_order(self, s):
        low  = np.minimum(s[RAISELIM], s[CALL_LIM])
        high = np.maximum(s[RAISELIM], s[CALL_LIM])
        s[RAISELIM], s[CALL_LIM] = low, high
        return s

    def getBet(self, hand_name):
        idx = self.HAND_TO_INDEX[hand_name]
        self.hand_index = idx
        if self.rng.random() < (self.strategy[BLUFF_PROB, idx] / MAX_BET):
            b = self.strategy[BLUFF_SIZE, idx]
            return b, b
        return ( self.strategy[RAISELIM, idx],
                 self.strategy[CALL_LIM, idx] )

    def handResult(self, delta):
        self.balances[self.hand_index] += delta

    def gameEnd(self, lost):
        if lost and self.update_mode == "uniform":
            self._upd_uniform(**self.update_parms)
        self.balances[:] = 0

    def _upd_uniform(self, max_stepsize=10):
        i = self.rng.integers(0, HANDS)
        for row in range(4):
            lo = max(0, self.strategy[row,i] - max_stepsize)
            hi = min(MAX_BET, self.strategy[row,i] + max_stepsize)
            self.strategy[row,i] = self.rng.uniform(lo, hi)
        self.strategy = self._enforce_order(self.strategy)
    
    def getData(self):
        """Return strategy and balances for analysis"""
        return self.strategy.copy(), self.balances.copy()