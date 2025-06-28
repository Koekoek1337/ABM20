import numpy as np
from mesa import Agent
from mesa.space import MultiGrid

from typing import List, Tuple

RAISELIM   = 0
CALL_LIM   = 1
BLUFF_PROB = 2
BLUFF_SIZE = 3

MAX_BET = 400

HANDS = 10

class fixedRiskAgent(Agent):
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

    def __init__(self, model, risk_aversion, update_mode="null", *args, **kwargs):
        super().__init__(model)
        self.update_mode = update_mode
        self.risk_aversion = risk_aversion
        
        self.strategy    = self.random_strat()
        self.balances    = np.zeros(HANDS)
        self.hand_index  = 0
        self.new_strat = self.strategy.copy()

    def random_strat(self):
        """
        returns a random strategy consisting of private valuations of every hand.
        """
        return self.rng.uniform(0, MAX_BET, HANDS)
    
    def getData(self):
        """Return strategy, balances, and risk aversion for analysis"""
        return self.strategy.copy(), self.balances.copy(), self.risk_aversion
    
    def getStrat(self):
        """Returns a copy the currently active agent strategy"""
        return self.strategy.copy()
    
    
    def getBet(self, hand_name) -> Tuple[int, int]:
        """
        Returns the appropriate raise and call limits for an agent's strategy and risk aversion, coresponding 
        to its poker hand.
        """
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

    def handResult(self, delta) -> None:
        """
        Adds the winnings or loss to the balance of the previously played hand
        """
        self.balances[self.hand_index] += delta
    
    def challenge(self) -> None:
        """Initiate a Poker match with von neuemann neighbors"""
        players: List[fixedRiskAgent] = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=1)
        self.model.game.setup(players)
        self.model.game.playGame(self.model.n_rounds)

    def neighborhood_adapt(self, recomb_target="self", recomb_method="random", nRec=5, wRec=0.5, 
                           mut_method="normal", mut_mod = 10, nMut=1) -> None:
        """ 
        Adapt strategy to moore neighborhood based on neighbor fitness (total balance).

        Args:
            recomb_target: if "self" the recombination will always use the existing strategy genotype 
                            as a base and pick another parent based on fitness. 
                           if "fittest" the recombination will select two parents based on fitness.
            recomb_method: if "rand" the recombination will pick random genes to recombine
                           if "fold" the recombination will use 1-fold mutation.
                           otherwise no recombination occurs
            nRec: Number of recombined genes to recombine into base parent. (only applicable to "rand" method)
            wRec: Weight of the recombined genes. (only applicable to "rand" method)
            mut_method: if "normal" the agent genes will change by picking a gene and sampling a new 
                         value from a normal distribution centered on the current value and with a choisen std.
                        if "uniform" the mutation will occur by sampling a step between 0 and a maximum step size and
                         applying it to the gene.
            mut_mod: The standard deviation form "normal" mutation or "max stepsize" for uniform random mutation.
            nMut: The number of random gene mutations to perform 
        """
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

        if recomb_method == "random":
            self.new_strat = self._randomRecombination(*partners, nRec, wRec)            
        elif recomb_method == "fold":
            self.new_strat = self._foldRecombination(*partners)

        if mut_method   == "normal" : self._upd_normal (mut_mod)
        elif mut_method == "uniform": self._upd_uniform_random(mut_mod)
        
        return

    def updateStrat(self) -> None:
        self.strategy = self.new_strat.copy()  
        return
    
    def gameEnd(self, lost, update_mode=None, scores=None, strategies=None):
        if update_mode == None: update_mode = self.update_mode
        if lost:
            if update_mode == "uniform": self._upd_uniform_random(**self.update_parms)
            if update_mode == "normal" : self._upd_normal(**self.update_parms)

    def _hand_strength(self, hand_name):
        """Normalize hand strength to [0, 1]"""
        return self.HAND_TO_INDEX[hand_name] / (HANDS - 1)
    
    def _risk_adjusted_raise(self, base_raise, strength):
        """Adjust raise amount based on risk aversion and hand strength"""
        risk_factor = (1 - self.risk_aversion)
        strength_bonus = self._sigmoid(strength) * risk_factor
        return base_raise * (0.5 + 0.5 * strength_bonus)

    def _sigmoid(self, x, a=8):
        """Smooth function mapping [-1, 1] → [0, 1]"""
        return 1 / (1 + np.exp(-a * (x - 0.5)))

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

    def _neighbor_selection(self) -> Tuple[List[Agent], List[float]]:
        """
        Get neighbors with greater fitness (score) than self and return them and their scores.
        """
        fitness = self.balances.sum()
        # assert isinstance(self.model.grid, MultiGrid)
        neighbors = self.model.grid.get_neighbors(self.pos, False, False)
        neigh_fitnesses = [neighbor.balances.sum() for neighbor in neighbors]
        
        # indexes of neighbors with greater score(fitness) than self
        i_candidates = [i for i, val in enumerate(neigh_fitnesses) if val > fitness]
        
        if not i_candidates:
            return None, None
        return [neighbors[i] for i in i_candidates], np.asarray([neigh_fitnesses[i] - fitness for i in i_candidates])

    def _randomRecombination(self, stratA:np.ndarray, stratB:np.ndarray, nChange:int, weightNew:float) -> np.ndarray:
        """
        Returns a randomly recombined genotype from two parents. The amount of newly introduced genes 
        and their respective weights can be adjusted.

        Args:
            stratA:  Base strategic genotype, is seen as the "original" genotype with regards to the input parameters.
            stratB:  Base strategic genotype, is seen as the "new" genotype with regards to the input parameters.
            fChange: Amount of new genes to introduce into the genotype of stratA.
            wightNew:    Weight of the newly introduced genes of stratA w.r.t. stratB.
        """
        weightOld = 1 - weightNew
        newStrat = stratA.copy()
        B_geneChoices  = self.rng.choice(len(self.strategy), nChange, False)
        for i in B_geneChoices:
            newStrat[i] = stratB[i] * weightNew + stratA[i] * weightOld
            assert not np.isnan(newStrat[i])
        return newStrat    

    def _foldRecombination(self, stratA:np.ndarray, stratB:np.ndarray) -> np.ndarray:
        """
        Returns a 1-fold recombined genotype for the agent.
        """
        parents = stratA, stratB
        coinflip = self.rng.choice(2,2,False)
        parStart, parEnd = parents[coinflip[0]], parents[coinflip[1]]

        splitpoint = self.rng.choice(len(stratA))
        parStart[splitpoint:] = parEnd[splitpoint:]

        return parStart

    def _upd_normal(self, std=10, nVals=1):
        statIndexes = self.rng.choice(len(self.strategy), nVals, False)
        for statIndex in statIndexes:
            # adjusted_std = std * (1 - 0.3 * self.risk_aversion)
            newVal = self.rng.normal(self.strategy[statIndex], std)
            self.new_strat[statIndex] = self._clampValue(newVal)
    
    def _upd_uniform_random(self, max_stepsize=10, nVals=1):
        """
        Choose a random hand value, then raise or lower it.
        """
        statIndexes = self.rng.choice(len(self.strategy), nVals, False)
        # adjusted_stepsize = max_stepsize * (1 - 0.5 * self.risk_aversion)
        
        for statIndex in statIndexes:
            lo = self.strategy[statIndex] - max_stepsize
            hi = self.strategy[statIndex] + max_stepsize
            newVal = self.rng.uniform(lo, hi)
            self.new_strat[statIndex] = self._clampValue(newVal)

    def _clampValue(self, value: float) -> float:
        """
        Clamps a value between 0 and MAX_BET
        """
        if value < 0: return 0
        if value > MAX_BET: return MAX_BET
        return value

class EvoRiskAgent(fixedRiskAgent):
    I_RISK = HANDS
    # Index of risk factor in agent "genotype"

    def __init__(self, model, update_mode="null", update_parms=None, risk_aversion=0.5, *args, **kwargs):
        super().__init__(model, risk_aversion, update_mode, update_parms, *args, **kwargs)
        self.strategy[self.I_RISK] = MAX_BET * (1 - risk_aversion)
        return

    def random_strat(self):
        return self.rng.uniform(0, MAX_BET, HANDS + 1)
    
    def updateStrat(self) -> None:
        super().updateStrat()
        self.risk_aversion = 1 - self.strategy[self.I_RISK] / MAX_BET
        return