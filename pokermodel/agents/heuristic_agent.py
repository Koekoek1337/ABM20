import random
import math

from agents.base_agent import BaseAgent
from utilities.utilities import *


LAST_OPPONENT_ACTIONS = 20
NOT_BLUFF_THRESHOLD = 0.33
WINRATE_ESTIMATE_ITERATIONS = 1000

class HeuristicAgent(BaseAgent):
    def __init__(self, name, wealth=1000, risk_aversion=0.5, aggression=0.5, 
                 rationality=2.0, bluff_frequency=0.1, bluff_detection=0.5):
        super().__init__(name, wealth, risk_aversion)
        self.aggression = aggression
        self.rationality = rationality
        self.bluff_frequency = bluff_frequency  # Probability of bluffing with weak hands
        self.bluff_detection = bluff_detection  # Ability to detect opponent bluffs
        self.recent_opponent_actions = []  # Track opponent behavior for bluff detection
        self.bluff_history = []  # Track own bluffing success
        
    def update_opponent_history(self, opponent_actions):
        """Update history of opponent actions for bluff detection"""
        self.recent_opponent_actions.extend(opponent_actions)
        if len(self.recent_opponent_actions) > LAST_OPPONENT_ACTIONS:
            self.recent_opponent_actions = self.recent_opponent_actions[-LAST_OPPONENT_ACTIONS:]
    
    def estimate_opponent_bluff_rate(self):
        """Estimate how often opponents are bluffing based on history"""
        if len(self.recent_opponent_actions) < 5:
            return 0.5
        
        # Simple heuristic: count aggressive plays vs conservative plays
        aggressive_plays = sum(1 for action in self.recent_opponent_actions if action == 'call')
        total_plays = len(self.recent_opponent_actions)
        
        bluff_estimate = max(0.1, min(0.4, (aggressive_plays / total_plays - 0.3) * 2))
        return bluff_estimate
    
    def should_bluff(self, win_prob, pot_size):
        """Decide whether to bluff based on various factors"""
        # Don't bluff if win probability is already good
        if win_prob > NOT_BLUFF_THRESHOLD:
            return False
        
        pot_factor = min(2.0, pot_size / 50.0)
        
        # Bluff more if we've been successful recently
        recent_bluff_success = 0.5  # Default
        if len(self.bluff_history) > 0:
            recent_bluff_success = sum(self.bluff_history[-5:]) / min(5, len(self.bluff_history))
        
        # Combine factors
        bluff_probability = self.bluff_frequency * pot_factor * (0.5 + recent_bluff_success)
        
        # Add some randomness based on personality
        bluff_probability *= (0.5 + self.aggression)
        
        return random.random() < bluff_probability
    
    def logit_equilibrium(self, utilities):
        """Implement logit equilibrium (LQRE) for bounded rationality"""
        # Apply rationality parameter to utilities
        scaled_utilities = [u * self.rationality for u in utilities]
        
        # Compute softmax probabilities
        max_util = max(scaled_utilities)
        exp_utils = [math.exp(u - max_util) for u in scaled_utilities]
        sum_exp = sum(exp_utils)
        
        probabilities = [exp_u / sum_exp for exp_u in exp_utils]
        
        # Choose action based on probabilities
        rand = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand <= cumulative:
                return i
        return len(probabilities) - 1
        
    def decide(self, hole_cards, pot_size, n_opponents, opponent_actions=None):
        """Enhanced decision making with bluffing and bounded rationality"""
        self.hands_played += 1
        
        if opponent_actions:
            self.update_opponent_history(opponent_actions)
        
        my_cards = [card_to_int(f"{card.value}{card.suit}") for card in hole_cards]
        win_prob, _ = simulate_win_probability(my_cards, iterations=WINRATE_ESTIMATE_ITERATIONS, n_opponents=n_opponents)
        
        opponent_bluff_rate = self.estimate_opponent_bluff_rate()
        adjusted_win_prob = win_prob + (opponent_bluff_rate * self.bluff_detection * 0.2)
        adjusted_win_prob = min(1.0, max(0.0, adjusted_win_prob))
        
        # Expected value calculations for both actions
        fold_utility = 0  # Folding gives 0 utility (no gain/loss from this decision)
        
        # Calculate call utility
        expected_return = adjusted_win_prob * pot_size - (1 - adjusted_win_prob) * pot_size
        
        # CRRA utility
        if expected_return != 0:
            if expected_return > 0:
                call_utility = (expected_return ** (1 - self.risk_aversion)) / (1 - self.risk_aversion)
            else:
                call_utility = -(abs(expected_return) ** (1 - self.risk_aversion)) / (1 - self.risk_aversion)
        else:
            call_utility = 0
        
        # Check for bluffing opportunity
        should_bluff = self.should_bluff(win_prob, pot_size)
        if should_bluff:
            # Bluffing adds utility based on fold equity
            fold_equity = 0.3 * self.aggression  # Estimate opponents fold to bluffs
            bluff_bonus = fold_equity * pot_size * 0.5
            call_utility += bluff_bonus
        
        # Use logit for decision making (bounded rationality)
        utilities = [fold_utility, call_utility]
        action = self.logit_equilibrium(utilities)
        
        # Track bluffing results for learning
        if should_bluff:
            self.bluff_history.append(action == 1)  # Will update with actual result later
        
        return action == 1
