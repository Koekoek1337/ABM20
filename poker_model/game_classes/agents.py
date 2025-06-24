# agents.py
import numpy as np
from mesa import Agent
from mesa.space import MultiGrid

from typing import List

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

    def gameEnd(self, lost, update_mode=None, scores=None, strategies=None):
        if update_mode == None: update_mode = self.update_mode
        if lost:
            if update_mode == "uniform":
                self._upd_uniform(**self.update_parms)

    def _upd_uniform(self, max_stepsize=10):
        """
        Choose a random hand from strategy, then update all parameters of the corresponding hand from an uniform dist.
        """
        i = self.rng.integers(0, HANDS)
        for row in range(4):
            lo = np.max(0, self.strategy[row,i] - max_stepsize)
            hi = np.min(MAX_BET, self.strategy[row,i] + max_stepsize)
            self.strategy[row,i] = self.rng.uniform(lo, hi)
        self.strategy = self._enforce_order(self.strategy)

    def _upd_normal(self, std=10):
        """
        TODO
        """
        pass

    def getStrat(self):
        return self.strategy.copy()

    def getData(self):
        """Return strategy and balances for analysis"""
        return self.strategy.copy(), self.balances.copy()
    
    def challenge(self):
        """Initiate Poker with neighbors"""
        players: List[AuctionPlayer] = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=1)
        self.model.game.setup(players)
        self.model.game.playGame(self.model.n_rounds)
    
    def neighborhood_adapt(self):
        """
        TODO: MAYBE NEVER
        """
        pass

    def _neighbor_selection(self):
        """
        Get neighbors with greater fitness than self.
        """
        fitness = self.balances.sum()
        # assert isinstance(self.model.grid, MultiGrid)
        neighbors = self.model.grid.get_neighbors(self.pos, True, False)
        neigh_fitnesses = [neighbor.balances.sum() for neighbor in neighbors]
        
        # indexes of neighbors with greater score(fitness) than self
        i_candidates = [i for i, val in enumerate(neigh_fitnesses) if val > fitness]
        
        if i_candidates:
            return [neighbors[i] for i in i_candidates], np.asarray([neigh_fitnesses[i] - fitness for i in i_candidates])
        return None, None
    
    def _recombination(self, stratA, stratB):
        """TODO: Maybe never, meant for riskfree and derivatives"""
        pass

    def _clampValue(self, value: float) -> float:
        """
        Clamps a value between 0 and MAX_BET
        """
        if value < 0: return 0
        if value > MAX_BET: return MAX_BET  


class RiskFreeAgent(AuctionPlayer):
    """
    Plays poker as an auction with a fixed hand of values.
    Only raises and does not call.
    """
    def __init__(self, model, update_mode="null", update_parms=None, *args, **kwargs):
        super().__init__(model, update_mode, update_parms, *args, **kwargs)
        self.new_strat = self.strategy.copy()
    
    def getBet(self, hand_name):
        idx = self.HAND_TO_INDEX[hand_name]
        self.hand_index = idx
        return ( self.strategy[idx],
                 self.strategy[idx])
    
    def _random_strat(self):
        return self.rng.uniform(0, MAX_BET, HANDS)
    
    def _upd_uniform_random(self, max_stepsize=10):
        """
        Choose a random hand value, then raise or lower it
        """
        statIndex = self.rng.choice(len(self.strategy))
        lo = np.max(0, self.strategy[statIndex] - max_stepsize)
        hi = np.min(MAX_BET, self.strategy[statIndex] + max_stepsize)
        newVal = self.rng.uniform(lo, hi)
        self.new_strat[statIndex] = newVal
 
    def _upd_normal(self, std = 10):
        statIndex = self.rng.choice(len(self.strategy))
        newVal = self.rng.normal(self.strategy[statIndex], std)
        self.new_strat[statIndex] = self._clampValue(newVal) 

    def _recombination(self, stratA, stratB):
        """
        Returns a recombined genotype for the agent.
        """
        parents = stratA, stratB
        coinflip = self.rng.choice(2,2,False)
        parStart, parEnd = parents[coinflip[0]], parents[coinflip[1]]

        splitpoint = self.rng.choice(len(stratA))
        parStart[splitpoint:] = parEnd[splitpoint:]

        return parStart
    
    def neighborhood_adapt(self, recomb_method="self", mut_method=None, mut_mod = 10):
        """ 
        Adapt strategy to moore neighborhood based on neighbor fitness
        """
        if  mut_method is None: mut_method = self.update_parms

        candidates, scores = self._neighbor_selection()

        if not candidates:
            return
        
        prob_density = scores / scores.sum()

        if recomb_method == "self" or len(candidates) == 1:
            i_partner =  self.rng.choice(len(candidates), p = prob_density)
            self.new_strat = self._recombination(self.strategy.copy(), candidates[i_partner].getStrat())
        elif recomb_method == "fittest":
            i_partners    =  self.rng.choice(len(candidates), 2, False, p = prob_density)
            self.new_strat = self._recombination(candidates[i_partners[0]].getStrat(), candidates[i_partner].getStrat())

        if mut_method   == "normal" : self._upd_normal (mut_mod)
        elif mut_method == "uniform": self._upd_uniform(mut_mod)
        
        return

    def updateStrat(self):
        self.strategy = self.new_strat.copy()

class SingleRiskAgent(RiskFreeAgent):
    I_RISK = HANDS
    # Index of risk factor in agent "genotype"

    def __init__(self, model, update_mode="null", update_parms=None, *args, **kwargs):
        super().__init__(model, update_mode, update_parms, *args, **kwargs)

    def _random_strat(self):
        return self.rng.uniform(0, MAX_BET, HANDS + 1)
    
    def getBet(self, hand_name):
        idx = self.HAND_TO_INDEX[hand_name]
        self.hand_index = idx
        return ( self.strategy[idx],
                 self.strategy[idx] * (1 + self.strategy[self.I_RISK] / MAX_BET))