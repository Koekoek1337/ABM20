from collections import defaultdict
import math
import random

from agents.base_agent import BaseAgent
from utilities.utilities import *

class RLAgent(BaseAgent):
    def __init__(self, name, wealth=1000, risk_aversion=0.5, alpha=0.1, gamma=0.9, epsilon=0.1, 
                 rationality=3.0, bluff_learning_rate=0.05):
        super().__init__(name, wealth, risk_aversion)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rationality = rationality
        self.bluff_learning_rate = bluff_learning_rate
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])  # [fold, call, bluff]
        self.bluff_success_rate = 0.3  # Initial estimate
        self.opponent_model = defaultdict(lambda: {'aggression': 0.5, 'bluff_rate': 0.2})
        self.last_state = None
        self.last_action = None
        self.last_was_bluff = False
        
    def encode_state(self, win_prob, pot_size, wealth_ratio, opponent_aggression):
        """Encode state for Q-learning including opponent modeling"""
        return (
            int(win_prob * 10),  # 0-10
            min(pot_size // 10, 20),  # pot size buckets
            int(wealth_ratio * 5),  # wealth relative to initial
            int(opponent_aggression * 10)  # opponent aggression estimate
        )
    
    def update_opponent_model(self, opponent_actions):
        """Update model of opponent behavior"""
        if not opponent_actions:
            return
        
        # Simple opponent modeling based on recent actions
        aggressive_actions = sum(1 for action in opponent_actions[-10:] if action == 'call')
        total_recent = len(opponent_actions[-10:])
        
        if total_recent > 0:
            observed_aggression = aggressive_actions / total_recent
            # Update with exponential smoothing
            current_estimate = self.opponent_model['default']['aggression']
            self.opponent_model['default']['aggression'] = \
                0.7 * current_estimate + 0.3 * observed_aggression
    
    def should_attempt_bluff(self, win_prob, pot_size, opponent_aggression):
        """Decide if this is a good bluffing situation"""
        # Don't bluff if hand is already strong
        if win_prob > 0.4:
            return False
        
        # Bluff more against tight opponents
        tight_opponent_bonus = (1 - opponent_aggression) * 0.3
        
        # Bluff more with larger pots
        pot_incentive = min(1.0, pot_size / 100.0)
        
        # Consider historical bluff success
        success_factor = self.bluff_success_rate
        
        bluff_value = tight_opponent_bonus + pot_incentive + success_factor
        return bluff_value > 0.6
    
    def choose_action(self, state, win_prob, pot_size, opponent_aggression):
        """Enhanced action selection with bluffing and bounded rationality"""
        # Epsilon-greedy with bluffing consideration
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])  # explore: fold, call, bluff
        
        q_values = self.q_table[state].copy()
        
        # Check if bluffing makes sense in this situation
        if not self.should_attempt_bluff(win_prob, pot_size, opponent_aggression):
            q_values[2] = -float('inf')  # Don't consider bluffing
        
        # Apply bounded rationality using quantal response
        scaled_q = [q * self.rationality for q in q_values]
        max_q = max(scaled_q)
        exp_q = [math.exp(q - max_q) for q in scaled_q]
        sum_exp = sum(exp_q)
        
        probabilities = [exp_q_val / sum_exp for exp_q_val in exp_q]
        
        # Sample action based on probabilities
        rand = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand <= cumulative:
                return i
        return 1  # Default to call
    
    def decide(self, hole_cards, pot_size, n_opponents, opponent_actions=None):
        """Enhanced decision making with bluffing and opponent modeling"""
        self.hands_played += 1
        
        # Update opponent model
        if opponent_actions:
            self.update_opponent_model(opponent_actions)
        
        # Convert cards and estimate win probability
        my_cards_int = [card_to_int(f"{card.value}{card.suit}") for card in hole_cards]
        win_prob, _ = simulate_win_probability(my_cards_int, iterations=1000, n_opponents=n_opponents)
        
        # Get opponent characteristics
        opponent_aggression = self.opponent_model['default']['aggression']
        
        # Create state
        wealth_ratio = self.wealth / self.initial_wealth
        state = self.encode_state(win_prob, pot_size, wealth_ratio, opponent_aggression)
        
        # Choose action (0=fold, 1=call, 2=bluff)
        action = self.choose_action(state, win_prob, pot_size, opponent_aggression)
        
        # Apply risk aversion override for very bad situations
        expected_value = win_prob * pot_size - (1 - win_prob) * pot_size
        risk_adjusted_ev = expected_value * (1 - self.risk_aversion)
        
        if risk_adjusted_ev < -pot_size * 0.7 and action != 0:
            action = 0  # Force fold in very bad situations
        
        self.last_state = state
        self.last_action = action
        self.last_was_bluff = (action == 2)
        
        # Convert to boolean (call/bluff both mean "stay in")
        return action != 0
    
    def update(self, reward, bluff_successful=None):
        """Update Q-values and bluff success rate"""
        if self.last_state is None:
            return
        
        # Update bluff success rate if we bluffed
        if self.last_was_bluff and bluff_successful is not None:
            self.bluff_success_rate = \
                (1 - self.bluff_learning_rate) * self.bluff_success_rate + \
                self.bluff_learning_rate * (1.0 if bluff_successful else 0.0)
        
        # Q-learning update
        next_state_value = 0
        target = reward + self.gamma * next_state_value
        current_q = self.q_table[self.last_state][self.last_action]
        
        self.q_table[self.last_state][self.last_action] = \
            current_q + self.alpha * (target - current_q)
