
import numpy as np
from numba import njit
from utilities.game import HOLE

VALUES = '23456789TJQKA'
SUITS = 'DCHS'
VALUE_MAP = {v: i for i, v in enumerate(VALUES)}
SUIT_MAP = {s: i for i, s in enumerate(SUITS)}

def card_to_int(card_str):
    """Convert card string like 'AS' to integer 0-51"""
    return VALUE_MAP[card_str[0]] * 4 + SUIT_MAP[card_str[1]]

def int_to_card(card_int):
    """Convert integer 0-51 to card string"""
    return VALUES[card_int // 4] + SUITS[card_int % 4]

@njit
def evaluate_hand_strength(cards):
    """
    Fast hand evaluation using high card + pairs heuristic
    Returns a score where higher = better
    """
    values = np.array([c // 4 for c in cards])
    values.sort()
    
    # Count occurrences of each value
    counts = np.zeros(13, dtype=np.int32)
    for v in values:
        counts[v] += 1
    
    # Check for pairs, trips, etc.
    max_count = np.max(counts)
    if max_count >= 4:
        return 8000 + np.argmax(counts)  # Four of a kind
    elif max_count == 3:
        if np.sum(counts >= 2) >= 2:
            return 7000 + np.argmax(counts)  # Full house
        else:
            return 4000 + np.argmax(counts)  # Three of a kind
    elif max_count == 2:
        pair_count = np.sum(counts == 2)
        if pair_count >= 2:
            pairs = np.where(counts == 2)[0]
            return 3000 + pairs[-1] * 100 + pairs[-2]  # Two pair
        else:
            return 2000 + np.argmax(counts)  # One pair
    else:
        return values[-1]  # High card

@njit
def simulate_win_probability(cards, iterations=5000, n_opponents=3):
    """
    Monte Carlo simulation to estimate win probability for a given hand
    """
    wins = 0
    ties = 0
    
    # Convert to numpy array for Numba compatibility
    my_cards = np.array(cards, dtype=np.int32)
    
    for _ in range(iterations):
        # Create deck and mark used cards
        used = np.zeros(52, dtype=np.bool_)
        for card in my_cards:
            used[card] = True
        
        available = np.where(~used)[0]
        np.random.shuffle(available)
        cards_needed = n_opponents * HOLE + 5
        if len(available) < cards_needed:
            continue
            
        dealt = available[:cards_needed]
        board = dealt[-5:]
        
        # Create opponent hands
        my_best = evaluate_hand_strength(np.concatenate((my_cards, board)))
        
        opponent_scores = []
        for i in range(n_opponents):
            opp_cards = dealt[i*2:(i+1)*2]
            opp_hand = np.concatenate((opp_cards, board))
            opponent_scores.append(evaluate_hand_strength(opp_hand))
        
        max_opponent = max(opponent_scores) if opponent_scores else 0
        
        if my_best > max_opponent:
            wins += 1
        elif my_best == max_opponent:
            ties += 1
    
    return wins / iterations, ties / iterations
