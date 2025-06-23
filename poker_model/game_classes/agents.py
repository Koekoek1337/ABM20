import numpy as np
from mesa import Agent, Model

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

    def __init__(self, model, update_mode="null", update_parms=None, risk_aversion=0.5):
        self.model = model
        self.update_mode = update_mode
        self.update_parms = update_parms or {}
        self.risk_aversion = risk_aversion
        self.pos = None

        self.n_hand_types = 10  # or however many your game defines (e.g. PokerPy.HAND_TYPES)
        self.balances = np.zeros(100)  # stores profits per round, or adjust if needed
        self.strategy = self._init_strategy()

    def _init_strategy(self):
        # Strategy: [aggressiveness, call threshold] per hand type
        strat = self.model.rng.uniform(0, 1, (2, self.n_hand_types))
        return self._enforce_order(strat)

    def _enforce_order(self, strat):
        # Ensure stronger hands have higher bet levels
        strat[0] = np.sort(strat[0])  # RAISE line increases with hand strength
        strat[1] = np.sort(strat[1])  # CALL line increases with hand strength
        return strat

    def _hand_strength(self, hand_name):
        """Normalize hand strength to [0, 1]"""
        return self.HAND_TO_INDEX[hand_name] / (HANDS - 1)

    def _sigmoid(self, x, a=8):
        """Smooth function mapping [-1, 1] → [0, 1]"""
        return 1 / (1 + np.exp(-a * (x - 0.5)))

    def _raise_amount(self, strength):
        """Higher hand strength & lower risk aversion → more aggressive raise"""
        base = self._sigmoid(strength)
        return MAX_BET * base * (1 - self.risk_aversion)

    def _call_amount(self, strength):
        """Always >= raise; moderately conservative"""
        return self._raise_amount(strength) + MAX_BET * 0.1 * (1 - self.risk_aversion)

    def _bluff_prob(self, strength):
        """Lower hand strength → more bluffing if risk-seeking"""
        weak = 1 - strength
        return (1 - self.risk_aversion) * self._sigmoid(weak)

    def _bluff_size(self, strength):
        """Small bluffs if risk-averse, bigger otherwise"""
        return MAX_BET * 0.2 + (MAX_BET * 0.5 * (1 - self.risk_aversion))

    def getBet(self, hand_name):
        strength = self._hand_strength(hand_name)
        self.hand_index = self.HAND_TO_INDEX[hand_name]

        if self.random.random() < self._bluff_prob(strength):
            bluff = self._bluff_size(strength)
            return bluff, bluff

        raise_amt = self._raise_amount(strength)
        call_amt  = self._call_amount(strength)
        return raise_amt, call_amt

    def handResult(self, delta):
        self.balances[self.hand_index] += delta

    def gameEnd(self, lost):
        self.balances[:] = 0
        # (optional) implement learning here

    def getData(self):
        """Return scalar data for analysis"""
        return {"risk_aversion": self.risk_aversion,
                "balances": self.balances.copy()}

    def update_strategy(self, model, lost=False):
        """Update strategy based on the configured mode and loss condition."""
        if self.update_mode == "null":
            return  # Do nothing
        
        if self.update_mode == "uniform":
            self._update_uniform(lost) #, **self.update_parms)
        elif self.update_mode == "neighbor_copy":
            self._update_neighbor_copy(model, lost) #, **self.update_parms)
        else:
            raise ValueError(f"Unknown update_mode: {self.update_mode}")

    def _update_uniform(self, lost: bool, max_stepsize=10):
        """Update strategy slightly by perturbing a random parameter, scaled by risk aversion."""
        if not lost:
            return  # Only update if the player lost
        
        # Pick a random hand
        i = self.model.rng.integers(0, HANDS)

        for row in range(self.strategy.shape[0]):
            current = self.strategy[row, i]

            # Determine range to explore based on risk_aversion
            step = self.model.rng.uniform(-1, 1) * max_stepsize * (1.0 - self.risk_aversion)
            new_val = np.clip(current + step, 0, MAX_BET)

            self.strategy[row, i] = new_val

        # If bet parameters are included, you might enforce internal constraints
        self.strategy = self._enforce_order(self.strategy)

    def _update_neighbor_copy(self, model: Model, lost: bool, copy_prob=0.3, blend=True):
        if not lost:
            return

        # Get neighbors via model
        neighbors = model.get_neighbors(self)
        if not neighbors:
            return

        # Filter for more successful neighbors
        my_score = self.balances.sum()
        better_neighbors = [n for n in neighbors if n.balances.sum() > my_score]

        if not better_neighbors:
            return

        # Select a neighbor to copy
        selected = self.model.rng.choice(better_neighbors)
        
        # Copy one hand's strategy (random column)
        i = self.model.rng.integers(0, self.strategy.shape[1])

        if blend:
            alpha = self.model.rng.uniform(0, copy_prob)  # blend weight
            self.strategy[:, i] = (
                (1 - alpha) * self.strategy[:, i] + alpha * selected.strategy[:, i]
            )
        else:
            self.strategy[:, i] = selected.strategy[:, i].copy()

        self.strategy = self._enforce_order(self.strategy)
