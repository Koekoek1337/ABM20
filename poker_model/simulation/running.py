import numpy as np
import pandas as pd

from copy import copy

from game_classes.model import SpatialModel
from game_classes.agents import fixedRiskAgent, EvoRiskAgent
from simulation.analysis import analyze_step

from typing import List, Dict, Tuple, Any

def run_multiple_simulations(scenarios: List[Dict[str, Any]], sharedParameters: Dict[str,Any], replications=3, seed=42):
    """Run multiple simulations with different risk aversion bounds"""
    SEED_GENERATOR = np.random.default_rng(seed)

    all_results = []
    final_results = []
    
    for scenario in scenarios:
        scenario["parameters"] = scenario["parameters"] | sharedParameters
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario['name']} (RA: {scenario["parameters"]['ra_bounds']})")
        print(f"{'='*50}")
        
        scenario_results = []

        
        
        for rep in range(replications):
            print(f"\nReplication {rep+1}/{replications}")
            
            # Use different seeds for each replication
            seed = SEED_GENERATOR.choice(2**32)
            scenario["parameters"]["seed"] = seed
            
            model, evolution_data = run_single_simulation(
                scenario["parameters"] 
            )
            
            evolution_df = extractEvoData(scenario,evolution_data, rep)
            all_results.append(evolution_df)
            scenario_results.append(evolution_df)

            final_agents = extractAgentFinState(scenario, model, rep)
            final_results.extend(final_agents)
    
    # Combine all results
    combined_evolution = pd.concat(all_results, ignore_index=True)
    final_agents_df = pd.DataFrame(final_results)
    
    return combined_evolution, final_agents_df

def run_single_simulation(modelParameters: Dict[str, Any]):
    """Run a single simulation with given risk aversion bounds"""
    ra_bounds = modelParameters["ra_bounds"]
    print(f"Running simulation with RA bounds: {ra_bounds}")
    games = modelParameters["games"]
    modelParameters = copy(modelParameters)
    modelParameters.pop("games")
    
    M = SpatialModel(**modelParameters)
    
    # Track evolution metrics
    evolution_data = {
        'step': [],
        'avg_risk_aversion': [],
        'std_risk_aversion': [],
        'total_wealth': [],
        'wealth_gini': [],
        'max_wealth': [],
        'min_wealth': [],
        'ra_wealth_correlation': []
    }
    
    for game in range(games):
        print(f"  Game {game+1}/{games}")
        M.step()
        
        # Calculate metrics after each step
        step_data = analyze_step(M, game)
        for key, value in step_data.items():
            evolution_data[key].append(value)
    
    return M, evolution_data


def extractEvoData(scenario:Dict[str,Any], evolution_data, rep:int) -> pd.DataFrame:
    for key in evolution_data:
        if isinstance(evolution_data[key], list):
            evolution_data[key] = np.array(evolution_data[key])
    
    evolution_df = pd.DataFrame(evolution_data)
    evolution_df['scenario'] = scenario['name']
    evolution_df['replication'] = rep
    evolution_df['ra_bounds'] = str(scenario["parameters"]["ra_bounds"])

    return evolution_df


def extractAgentFinState(scenario:Dict[str,Any], model:SpatialModel, rep:int) -> List[Dict[str, Any]]:
    final_agents = []
    for agent in model.agents:
        if hasattr(agent, 'pos') and agent.pos is not None:
            x, y = agent.pos
            wealth = model.profit_grids[-1][x, y] if model.profit_grids else 0
            final_agents.append({
                'scenario': scenario['name'],
                'replication': rep,
                'risk_aversion': agent.risk_aversion,
                'wealth': wealth,
                'ra_bounds': str(scenario["parameters"]["ra_bounds"])
            })
    return final_agents