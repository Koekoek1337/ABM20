import numpy as np

from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent
from utilities.game import PokerGame
from utilities.utilities import *
from utilities.plots import *


def analyze_risk_aversion_impact():
    """Analyze how different risk aversion levels affect performance"""
    risk_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for risk in risk_levels:
        print(f"\nTesting risk aversion: {risk}")
        
        agents = [
            HeuristicAgent(f"Conservative_{risk}", risk_aversion=risk, aggression=0.3),
            HeuristicAgent(f"Moderate_{risk}", risk_aversion=risk, aggression=0.5),
            HeuristicAgent(f"Aggressive_{risk}", risk_aversion=risk, aggression=0.7),
        ]
        
        game = PokerGame(agents, rounds=500)
        wealth_history = game.simulate()
        game_results = game.get_results()
        
        for agent_name, stats in game_results.items():
            results.append({
                'risk_aversion': risk,
                'agent_type': agent_name.split('_')[0],
                'final_wealth': stats['final_wealth'],
                'profit': stats['profit'],
                'win_rate': stats['win_rate']
            })
    
    return results


if __name__ == "__main__":
    print("=== Poker Agent Simulation ===\n")
    
    # Create agents with different risk aversion levels
    agents = [
        HeuristicAgent("Conservative", risk_aversion=0.8, aggression=0.2),
        HeuristicAgent("Moderate", risk_aversion=0.5, aggression=0.5),
        HeuristicAgent("Aggressive", risk_aversion=0.2, aggression=0.8),
    ]
    
    # Run simulation
    game = PokerGame(agents, ante=10, rounds=1000)
    wealth_history = game.simulate()
    
    # Display results
    print("\n=== Final Results ===")
    results = game.get_results()
    
    for agent_name, stats in results.items():
        print(f"\n{agent_name}:")
        print(f"  Final Wealth: ${stats['final_wealth']}")
        print(f"  Profit/Loss: ${stats['profit']:+d}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Hands Played: {stats['hands_played']}")
        print(f"  Risk Aversion: {stats['risk_aversion']:.2f}")
    
    # Plot results
    plot_wealth_evolution(wealth_history)

    print("\n=== Risk Aversion Analysis ===")
    risk_analysis = analyze_risk_aversion_impact()
    
    print("\nRisk vs Performance Summary:")
    for risk in [0.1, 0.3, 0.5, 0.7, 0.9]:
        risk_data = [r for r in risk_analysis if r['risk_aversion'] == risk]
        avg_profit = np.mean([r['profit'] for r in risk_data])
        avg_winrate = np.mean([r['win_rate'] for r in risk_data])
        print(f"Risk {risk}: Avg Profit = ${avg_profit:+.0f}, Avg Win Rate = {avg_winrate:.2%}")