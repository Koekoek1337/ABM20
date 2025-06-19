import numpy as np
import scipy as sp
import PokerPy
import random
import math

from typing import Tuple, List, Dict, Any

RAISELIM = 0
CALL_LIM = 1
MAX_BET = 400
LAST_OPPONENT_ACTIONS = 20
NOT_BLUFF_THRESHOLD = 0.33
WINRATE_ESTIMATE_ITERATIONS = 100

class AuctionPlayer:
    HAND_TO_INDEX = {
        "Royal Flush": 9,
        "Straight Flush": 8,
        "Poker": 7,
        "Full House": 6,
        "Flush": 5,
        "Straight": 4,
        "Triples": 3,
        "Double Pairs": 2,
        "Pairs": 1,
        "High Card": 0
    }
    
    def __init__(self, rng: np.random.Generator, updateMode="uniform", updateParms={"maxStepsize": 10}):
        self.RNG = rng
        self.strategy = np.zeros((2, 10), float)
        self.strategy[RAISELIM, :] = self.randomStrat(50, 150)
        self.strategy[CALL_LIM, :] = self.randomStrat(150, 200)
        self.balances = np.zeros(10, float)
        self.handIndex = None
        self.updateMode = updateMode
        self.updateParms = updateParms
        self.n_opponents = None  # Will be set by Game
        self.recent_opponent_actions = []
        self.bluff_history = []
        self.aggression = 0.5
        self.risk_aversion = 0.5
        self.rationality = 2.0
        self.bluff_frequency = 0.1
        self.bluff_detection = 0.5

    def randomStrat(self, minB=50, maxB=200) -> np.ndarray:
        return self.RNG.uniform(minB, maxB, 10)

    def set_strategy(self, strategy: np.ndarray) -> None:
        self.strategy = strategy

    def getBet(self, hand: str, hole_cards, community_cards) -> np.ndarray[float]:
        self.handIndex = self.HAND_TO_INDEX[hand]
        
        # Get base strategy values
        base_raise = self.strategy[RAISELIM, self.handIndex]
        base_call = self.strategy[CALL_LIM, self.handIndex]
        
        # Heuristic adjustments
        win_prob = self.estimate_win_probability(hole_cards, community_cards)
        opponent_bluff_rate = self.estimate_opponent_bluff_rate()
        adjusted_win_prob = win_prob + (opponent_bluff_rate * self.bluff_detection * 0.2)
        adjusted_win_prob = min(1.0, max(0.0, adjusted_win_prob))
        
        # Adjust based on win probability and aggression
        win_factor = 1.0 + (adjusted_win_prob - 0.5) * self.aggression
        risk_factor = 1.0 - self.risk_aversion * (1 - adjusted_win_prob)
        
        adjusted_raise = base_raise * win_factor * risk_factor
        adjusted_call = base_call * win_factor * risk_factor
        
        # Apply bluffing logic
        if self.should_bluff(win_prob, base_raise + base_call):
            # Increase raise amount when bluffing
            adjusted_raise *= (1.0 + self.aggression)
        
        # Ensure values are within valid range
        adjusted_raise = max(0, min(MAX_BET, adjusted_raise))
        adjusted_call = max(0, min(MAX_BET, adjusted_call))
        
        return np.array([adjusted_raise, adjusted_call])

    def handResult(self, balanceChange) -> None:
        self.balances[self.handIndex] += balanceChange

    def getData(self):
        return self.strategy.copy(), self.balances.copy()

    def gameEnd(self, loss: bool):
        if loss:    
            if self.updateMode == "uniform":
                self._upd_uniformRandom(**self.updateParms)
            elif self.updateMode == "null":
                pass
        self.balances = self.balances * 0

    def _upd_uniformRandom(self, maxStepsize, **kwp):
        updateIndex = self.RNG.choice(self.strategy.shape[1])
        lowBet = self.strategy[0, updateIndex] - maxStepsize
        highBet = self.strategy[0, updateIndex] + maxStepsize
        lowRaise = self.strategy[0, updateIndex] - maxStepsize
        highRaise = self.strategy[0, updateIndex] + maxStepsize

        if lowBet < 0: lowBet = 0 
        if highBet > MAX_BET: highBet = MAX_BET
        if lowRaise < 0: lowRaise = 0
        if highRaise > MAX_BET: highRaise = MAX_BET

        newStrat = np.asarray((self.RNG.uniform(lowBet, highBet), self.RNG.uniform(lowRaise, highRaise)))
        self.strategy[RAISELIM, updateIndex] = np.max(newStrat)
        self.strategy[CALL_LIM, updateIndex] = np.min(newStrat)

    def _null(self, **kwp):
        return

    # Heuristic agent methods
    def update_opponent_history(self, opponent_actions):
        self.recent_opponent_actions.extend(opponent_actions)
        if len(self.recent_opponent_actions) > LAST_OPPONENT_ACTIONS:
            self.recent_opponent_actions = self.recent_opponent_actions[-LAST_OPPONENT_ACTIONS:]

    def estimate_opponent_bluff_rate(self):
        if len(self.recent_opponent_actions) < 5:
            return 0.5
        
        aggressive_plays = sum(1 for action in self.recent_opponent_actions if action == 'raise')
        total_plays = len(self.recent_opponent_actions)
        bluff_estimate = max(0.1, min(0.4, (aggressive_plays / total_plays - 0.3) * 2))
        return bluff_estimate

    def should_bluff(self, win_prob, pot_size):
        if win_prob > NOT_BLUFF_THRESHOLD:
            return False
        
        pot_factor = min(2.0, pot_size / 50.0)
        recent_bluff_success = 0.5
        if len(self.bluff_history) > 0:
            recent_bluff_success = sum(self.bluff_history[-5:]) / min(5, len(self.bluff_history))
        
        bluff_probability = self.bluff_frequency * pot_factor * (0.5 + recent_bluff_success)
        bluff_probability *= (0.5 + self.aggression)
        return random.random() < bluff_probability

    def estimate_win_probability(self, hole_cards, community_cards):
        # Simplified win probability estimation
        hand = PokerPy.get_best_hand(hole_cards + community_cards)
        base_strength = self.HAND_TO_INDEX[hand.hand_type] / 9.0
        
        # Adjust based on cards not yet revealed
        cards_visible = len(hole_cards) + len(community_cards)
        uncertainty_factor = 1.0 - (cards_visible / 52.0) * 0.3
        
        # Adjust for number of opponents
        opponent_factor = 1.0 / (1.0 + self.n_opponents * 0.2)
        
        return base_strength * uncertainty_factor * opponent_factor

    def logit_equilibrium(self, utilities):
        scaled_utilities = [u * self.rationality for u in utilities]
        max_util = max(scaled_utilities)
        exp_utils = [math.exp(u - max_util) for u in scaled_utilities]
        sum_exp = sum(exp_utils)
        probabilities = [exp_u / sum_exp for exp_u in exp_utils]
        rand = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand <= cumulative:
                return i
        return len(probabilities) - 1