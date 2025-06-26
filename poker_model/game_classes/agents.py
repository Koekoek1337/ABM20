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

    def __init__(self, model, risk_aversion, update_mode="null", update_parms=None, *args, **kwargs):
        super().__init__(model)
        self.update_mode = update_mode
        self.update_parms= update_parms or {}
        self.risk_aversion = risk_aversion
        
        # Keep original strategy structure for backward compatibility
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

    # New risk aversion helper functions from second implementation
    def _hand_strength(self, hand_name):
        """Normalize hand strength to [0, 1]"""
        return self.HAND_TO_INDEX[hand_name] / (HANDS - 1)

    def _sigmoid(self, x, a=8):
        """Smooth function mapping [-1, 1] → [0, 1]"""
        return 1 / (1 + np.exp(-a * (x - 0.5)))

    def _risk_adjusted_raise(self, base_raise, strength):
        """Adjust raise amount based on risk aversion and hand strength"""
        risk_factor = (1 - self.risk_aversion)
        strength_bonus = self._sigmoid(strength) * risk_factor
        return base_raise * (0.5 + 0.5 * strength_bonus)

    def _risk_adjusted_call(self, base_call, strength):
        """Adjust call limit based on risk aversion"""
        risk_factor = (1 - self.risk_aversion)
        return base_call * (0.7 + 0.3 * risk_factor)

    def _risk_adjusted_bluff_prob(self, base_prob, strength):
        """Adjust bluffing probability based on risk aversion"""
        weak = 1 - strength
        risk_bluff_bonus = (1 - self.risk_aversion) * self._sigmoid(weak) * 0.3
        return (base_prob / MAX_BET) + risk_bluff_bonus

    def _risk_adjusted_bluff_size(self, base_size, strength):
        """Adjust bluff size based on risk aversion"""
        risk_factor = (1 - self.risk_aversion)
        return base_size * (0.5 + 0.5 * risk_factor)

    def getBet(self, hand_name):
        idx = self.HAND_TO_INDEX[hand_name]
        self.hand_index = idx
        strength = self._hand_strength(hand_name)
        
        # Apply risk aversion to bluffing decision
        adjusted_bluff_prob = self._risk_adjusted_bluff_prob(self.strategy[BLUFF_PROB, idx], strength)
        
        if self.rng.random() < adjusted_bluff_prob:
            base_bluff = self.strategy[BLUFF_SIZE, idx]
            adjusted_bluff = self._risk_adjusted_bluff_size(base_bluff, strength)
            return adjusted_bluff, adjusted_bluff
        
        # Apply risk aversion to normal betting
        base_raise = self.strategy[RAISELIM, idx]
        base_call = self.strategy[CALL_LIM, idx]
        
        adjusted_raise = self._risk_adjusted_raise(base_raise, strength)
        adjusted_call = self._risk_adjusted_call(base_call, strength)
        
        return adjusted_raise, adjusted_call

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
        Risk aversion influences the step size.
        """
        i = self.rng.integers(0, HANDS)
        # Risk-averse agents make smaller strategy changes
        adjusted_stepsize = max_stepsize * (1 - 0.5 * self.risk_aversion)
        
        for row in range(4):
            lo = max(0, self.strategy[row,i] - adjusted_stepsize)
            hi = min(MAX_BET, self.strategy[row,i] + adjusted_stepsize)
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
        """Return strategy, balances, and risk aversion for analysis"""
        return self.strategy.copy(), self.balances.copy(), self.risk_aversion
    
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


class fixedRiskAgent(AuctionPlayer):
    """
    Plays poker as an auction with a fixed hand of values.
    Only raises and does not call.
    """
    def __init__(self, model, risk_aversion, update_mode="null", update_parms=None, *args, **kwargs):
        super().__init__(model, risk_aversion, update_mode, update_parms, *args, **kwargs)
        self.new_strat = self.strategy.copy()
    
    def getBet(self, hand_name):
        idx = self.HAND_TO_INDEX[hand_name]
        self.hand_index = idx
        strength = self._hand_strength(hand_name)
        
        # Apply risk aversion to the base strategy value
        base_bet = self.strategy[idx]
        adjusted_raise = self._risk_adjusted_raise(base_bet, strength)
        if adjusted_raise > MAX_BET: adjusted_raise = MAX_BET

        risk_multiplier = 1 + (1 - self.risk_aversion) # * (self.strategy[self.I_RISK] / MAX_BET)
        adjusted_call = adjusted_raise * risk_multiplier
        
        return adjusted_raise, adjusted_call
        # return adjusted_raise, adjusted_raise
    
    def _random_strat(self):
        return self.rng.uniform(0, MAX_BET, HANDS)
    
    def _upd_uniform_random(self, max_stepsize=10):
        """
        Choose a random hand value, then raise or lower it.
        Risk aversion influences step size.
        """
        statIndex = self.rng.choice(len(self.strategy))
        adjusted_stepsize = max_stepsize * (1 - 0.5 * self.risk_aversion)
        
        lo = max(0, self.strategy[statIndex] - adjusted_stepsize)
        hi = min(MAX_BET, self.strategy[statIndex] + adjusted_stepsize)
        newVal = self.rng.uniform(lo, hi)
        self.new_strat[statIndex] = newVal
 
    def _upd_normal(self, std = 10):
        statIndex = self.rng.choice(len(self.strategy))
        adjusted_std = std * (1 - 0.3 * self.risk_aversion)
        newVal = self.rng.normal(self.strategy[statIndex], adjusted_std)
        self.new_strat[statIndex] = self._clampValue(newVal) 

    def _foldRecombination(self, stratA, stratB):
        """
        Returns a recombined genotype for the agent.
        """
        parents = stratA, stratB
        coinflip = self.rng.choice(2,2,False)
        parStart, parEnd = parents[coinflip[0]], parents[coinflip[1]]

        splitpoint = self.rng.choice(len(stratA))
        parStart[splitpoint:] = parEnd[splitpoint:]

        return parStart
    
    def _randomRecombination(self, stratA: np.ndarray, stratB: np.ndarray):
        newStrat = stratA.copy()
        nB_genes = int(np.floor(len(self.strategy) / 2))
        B_geneChoices  = self.rng.choice(len(self.strategy), nB_genes, False)
        j = 0
        for i in B_geneChoices:
            newStrat[i] = stratB[i]
        return newStrat

    def neighborhood_adapt(self, recomb_target="self", recomb_method="rand", mut_method=None, mut_mod = 10):
        """ 
        Adapt strategy to moore neighborhood based on neighbor fitness
        """
        if  mut_method is None: mut_method = self.update_parms

        candidates, scores = self._neighbor_selection()

        if not candidates: return
        
        p_notChange = (self.balances.sum() / np.max(scores))
        if self.rng.random() < p_notChange: return
        
        prob_density = scores / scores.sum()
        if recomb_target == "self" or len(candidates) == 1:

            i_partner      = self.rng.choice(len(candidates), p = prob_density)
            partners       = [self.strategy.copy(), candidates[i_partner].getStrat()]
        elif recomb_target == "fittest":
            i_partners =  self.rng.choice(len(candidates), 2, False, p = prob_density)
            partners   = [candidates[i_partners[0]].getStrat(), candidates[i_partners[1]].getStrat()]

        if recomb_method == "rand":
            self.new_strat = self._randomRecombination(*partners)            
        if recomb_method == "fold":
            self.new_strat = self._foldRecombination(*partners)

        if mut_method   == "normal" : self._upd_normal (mut_mod)
        elif mut_method == "uniform": self._upd_uniform(mut_mod)
        
        return

    def updateStrat(self):
        self.strategy = self.new_strat.copy()

class EvoRiskAgent(fixedRiskAgent):
    I_RISK = HANDS
    # Index of risk factor in agent "genotype"

    def __init__(self, model, update_mode="null", update_parms=None, risk_aversion=0.5, *args, **kwargs):
        super().__init__(model, risk_aversion, update_mode, update_parms, *args, **kwargs)
        self.strategy[self.I_RISK] = MAX_BET * (1 - risk_aversion)

    def _random_strat(self):
        return self.rng.uniform(0, MAX_BET, HANDS + 1)
    
    def updateStrat(self):
        super().updateStrat()
        self.risk_aversion = 1 - self.strategy[self.I_RISK] / MAX_BET