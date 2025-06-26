import random
from tqdm import tqdm

from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent
from utilities.card import Card
from utilities.utilities import *

BLUFF_THRESHOLD = 0.3
BLUFF_ESTIMATION_ITERATIONS = 500
HOLE = 2

class PokerGame:
    def __init__(self, agents, ante=10, rounds=1000):
        self.agents = agents
        self.ante = ante
        self.rounds = rounds
        self.round_history = []
        
    def deal_hand(self):
        """Shuffles and deals HOLE cards to each player"""
        deck = [(v, s) for v in VALUES for s in SUITS]
        random.shuffle(deck)
        
        hands = []
        for i in range(len(self.agents)):
            hand = [Card(deck.pop()[0], deck.pop()[1]) for _ in range(HOLE)]
            hands.append(hand)
        
        return hands
    
    def play_round(self):
        """Play a single round with enhanced bluffing and opponent modeling"""
        n_players = len(self.agents)
        
        # Collect ante from all players
        pot = 0
        for agent in self.agents:
            if agent.wealth >= self.ante:
                agent.wealth -= self.ante
                agent.total_bet += self.ante
                pot += self.ante
            else:
                # Player is out of money
                return None
        
        hands = self.deal_hand() # Deal cards      
        round_actions = [] # Track actions for opponent modeling
        active_players = [] # For betting phase
        bluffing_players = set() # To track bluffs
        for i, agent in enumerate(self.agents):
            
            # Get opponent actions and calculate decision based on them
            opponent_actions = [action for j, action in enumerate(round_actions) if j != i]
            decision = agent.decide(hands[i], pot, n_players - 1, opponent_actions)
            
            action_name = 'call' if decision else 'fold'
            round_actions.append(action_name)
            
            if decision:
                active_players.append(i)
                
                # Check if this was a bluff (weak hand but still calling)
                agent_cards = [card_to_int(f"{card.value}{card.suit}") for card in hands[i]]
                win_prob, _ = simulate_win_probability(agent_cards, iterations=BLUFF_ESTIMATION_ITERATIONS, n_opponents=n_players-1)
                if win_prob < BLUFF_THRESHOLD:
                    bluffing_players.add(i)
        
        # Everyone folded, return ante to all
        if len(active_players) == 0:
            for agent in self.agents:
                agent.wealth += self.ante
            winner_idx = None
            bluff_successful = False
        
        # Only one player left
        elif len(active_players) == 1: 
            winner_idx = active_players[0]
            self.agents[winner_idx].wealth += pot
            self.agents[winner_idx].hands_won += 1
            bluff_successful = winner_idx in bluffing_players
        
        # Showdown - find best hand
        else:
            best_score = -1
            winner_idx = active_players[0]
            
            for player_idx in active_players:
                agents_hands = [card_to_int(f"{card.value}{card.suit}") for card in hands[player_idx]]
                score = evaluate_hand_strength(np.array(agents_hands))
                if score > best_score:
                    best_score = score
                    winner_idx = player_idx
            
            self.agents[winner_idx].wealth += pot
            self.agents[winner_idx].hands_won += 1

            bluff_successful = winner_idx in bluffing_players
        
        # Update agents
        for i, agent in enumerate(self.agents):
            # if isinstance(agent, RLAgent):
            #     reward = pot if i == winner_idx else -self.ante
            #     was_bluffing = i in bluffing_players
            #     agent.update(reward, bluff_successful if was_bluffing else None)
            # elif isinstance(agent, HeuristicAgent):
            if hasattr(agent, 'bluff_history') and len(agent.bluff_history) > 0:
                if i in bluffing_players:
                    agent.bluff_history[-1] = bluff_successful
        
        return winner_idx
    
    def simulate(self):
        """Run the full simulation"""
        wealth_history = {agent.name: [] for agent in self.agents}
        winner_idxs = []
        
        for round_num in tqdm(range(self.rounds), desc="Playing rounds"):
            winner_idx = self.play_round()
            
            if winner_idx is not None:
                winner_idxs.append(winner_idx)

            for agent in self.agents:
                wealth_history[agent.name].append(agent.wealth)
            
            if any(agent.wealth <= 0 for agent in self.agents):
                print(f"Game ended early at round {round_num + 1} - player went broke")
                break
        
        return wealth_history
    
    def get_results(self):
        """Get final results with enhanced statistics"""
        results = {}
        for agent in self.agents:
            win_rate = agent.hands_won / max(agent.hands_played, 1)
            profit = agent.wealth - agent.initial_wealth
            
            # Calculate additional statistics
            bluff_stats = {}
            if hasattr(agent, 'bluff_history') and agent.bluff_history:
                bluff_stats['total_bluffs'] = len(agent.bluff_history)
                bluff_stats['successful_bluffs'] = sum(agent.bluff_history)
                bluff_stats['bluff_success_rate'] = sum(agent.bluff_history) / len(agent.bluff_history)
            elif hasattr(agent, 'bluff_success_rate'):
                bluff_stats['estimated_bluff_success_rate'] = agent.bluff_success_rate
            
            results[agent.name] = {
                'final_wealth': agent.wealth,
                'profit': profit,
                'hands_played': agent.hands_played,
                'hands_won': agent.hands_won,
                'win_rate': win_rate,
                'total_bet': agent.total_bet,
                'risk_aversion': agent.risk_aversion,
                'rationality': getattr(agent, 'rationality', 'N/A'),
                'bluff_stats': bluff_stats
            }
        return results
