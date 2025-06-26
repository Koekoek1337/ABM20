from game_classes.model import SpatialModel
from game_classes.agents import fixedRiskAgent, EvoRiskAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns

def run_multiple_simulations():
    """Run multiple simulations with different risk aversion bounds"""
    
    # Define different risk aversion scenarios
    scenarios = [
        {'name': 'Risk Seeking', 'bounds': (0.0, 0.3), 'color': 'red'},
        {'name': 'Mixed Low', 'bounds': (0.0, 0.5), 'color': 'orange'},  
        {'name': 'Balanced', 'bounds': (0.2, 0.8), 'color': 'green'},
        {'name': 'Mixed High', 'bounds': (0.5, 1.0), 'color': 'blue'},
        {'name': 'Risk Averse', 'bounds': (0.7, 1.0), 'color': 'purple'},
        {'name': 'Full Range', 'bounds': (0.0, 1.0), 'color': 'black'}
    ]
    
    # Simulation parameters
    GAMES = 25
    GRID_SIZE = (8, 8)  # Smaller grid for faster execution
    N_ROUNDS = 800
    N_REPLICATIONS = 3  # Run each scenario multiple times
    
    all_results = []
    final_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario['name']} (RA: {scenario['bounds']})")
        print(f"{'='*50}")
        
        scenario_results = []
        
        for rep in range(N_REPLICATIONS):
            print(f"\nReplication {rep+1}/{N_REPLICATIONS}")
            
            # Use different seeds for each replication
            seed = 42 + rep * 100
            
            model, evolution_data = run_single_simulation(
                ra_bounds=scenario['bounds'],
                gridDim=GRID_SIZE,
                n_rounds=N_ROUNDS,
                games=GAMES,
                seed=seed
            )
            
            # Add scenario info to evolution data
            for key in evolution_data:
                if isinstance(evolution_data[key], list):
                    evolution_data[key] = np.array(evolution_data[key])
            
            evolution_df = pd.DataFrame(evolution_data)
            evolution_df['scenario'] = scenario['name']
            evolution_df['replication'] = rep
            evolution_df['ra_bounds'] = str(scenario['bounds'])
            
            all_results.append(evolution_df)
            scenario_results.append(evolution_df)
            
            # Store final state data
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
                        'ra_bounds': str(scenario['bounds'])
                    })
            
            final_results.extend(final_agents)
    
    # Combine all results
    combined_evolution = pd.concat(all_results, ignore_index=True)
    final_agents_df = pd.DataFrame(final_results)
    
    return scenarios, combined_evolution, final_agents_df


def run_single_simulation(ra_bounds, gridDim=(10, 10), n_rounds=1000, games=20, seed=None):
    """Run a single simulation with given risk aversion bounds"""
    print(f"Running simulation with RA bounds: {ra_bounds}")
    
    M = SpatialModel(gridDim=gridDim, n_rounds=n_rounds, seed=seed, ra_bounds=ra_bounds)
    
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

def analyze_step(model, step):
    """Analyze a single simulation step"""
    # Get agent data
    agents_data = []
    for agent in model.agents:
        if hasattr(agent, 'pos') and agent.pos is not None:
            x, y = agent.pos
            wealth = model.profit_grids[-1][x, y] if model.profit_grids else 0
            agents_data.append({
                'risk_aversion': agent.risk_aversion,
                'wealth': wealth,
                'position': agent.pos
            })
    
    if not agents_data:
        return {
            'step': step,
            'avg_risk_aversion': 0,
            'std_risk_aversion': 0,
            'total_wealth': 0,
            'wealth_gini': 0,
            'max_wealth': 0,
            'min_wealth': 0,
            'ra_wealth_correlation': 0
        }
    
    df = pd.DataFrame(agents_data)
    
    # Calculate metrics
    avg_ra = df['risk_aversion'].mean()
    std_ra = df['risk_aversion'].std()
    total_wealth = df['wealth'].sum()
    max_wealth = df['wealth'].max()
    min_wealth = df['wealth'].min()
    
    # Calculate Gini coefficient for wealth inequality
    wealth_gini = calculate_gini(df['wealth'].values)
    
    # Calculate correlation between risk aversion and wealth
    if len(df) > 1 and df['wealth'].std() > 0 and df['risk_aversion'].std() > 0:
        ra_wealth_corr = df['risk_aversion'].corr(df['wealth'])
    else:
        ra_wealth_corr = 0
    
    return {
        'step': step,
        'avg_risk_aversion': avg_ra,
        'std_risk_aversion': std_ra,
        'total_wealth': total_wealth,
        'wealth_gini': wealth_gini,
        'max_wealth': max_wealth,
        'min_wealth': min_wealth,
        'ra_wealth_correlation': ra_wealth_corr
    }

def calculate_gini(wealth_array):
    """Calculate Gini coefficient for wealth inequality"""
    if len(wealth_array) == 0:
        return 0
    
    # Ensure non-negative values
    wealth_array = np.maximum(wealth_array, 0)
    
    if np.sum(wealth_array) == 0:
        return 0
    
    # Sort wealth array
    sorted_wealth = np.sort(wealth_array)
    n = len(sorted_wealth)
    cumsum = np.cumsum(sorted_wealth)
    
    # Calculate Gini coefficient
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_wealth))) / (n * np.sum(sorted_wealth)) - (n + 1) / n
    return max(0, gini)  # Ensure non-negative

def create_comprehensive_analysis(scenarios, evolution_df, final_agents_df):
    """Create comprehensive analysis plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Evolution of Average Risk Aversion
    ax1 = plt.subplot(3, 4, 1)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            # Group by step and calculate mean/std across replications
            grouped = scenario_data.groupby('step')['avg_risk_aversion']
            mean_ra = grouped.mean()
            std_ra = grouped.std()
            
            steps = mean_ra.index
            ax1.plot(steps, mean_ra.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax1.fill_between(steps, mean_ra - std_ra, mean_ra + std_ra, 
                           alpha=0.2, color=scenario['color'])
    
    ax1.set_title('Evolution of Average Risk Aversion')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Average Risk Aversion')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolution of Total Wealth
    """
    ax2 = plt.subplot(3, 4, 2)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            grouped = scenario_data.groupby('step')['total_wealth']
            mean_wealth = grouped.mean()
            std_wealth = grouped.std()
            
            steps = mean_wealth.index
            ax2.plot(steps, mean_wealth.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax2.fill_between(steps, mean_wealth - std_wealth, mean_wealth + std_wealth, 
                           alpha=0.2, color=scenario['color'])
    
    ax2.set_title('Evolution of Total Wealth')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Total Wealth')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    """
    
    # 3. Wealth Inequality (Gini Coefficient)
    ax3 = plt.subplot(3, 4, 3)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            grouped = scenario_data.groupby('step')['wealth_gini']
            mean_gini = grouped.mean()
            std_gini = grouped.std()
            
            steps = mean_gini.index
            ax3.plot(steps, mean_gini.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax3.fill_between(steps, mean_gini - std_gini, mean_gini + std_gini, 
                           alpha=0.2, color=scenario['color'])
    
    ax3.set_title('Wealth Inequality Evolution (Gini)')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Gini Coefficient')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk Aversion vs Wealth Correlation
    ax4 = plt.subplot(3, 4, 4)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            grouped = scenario_data.groupby('step')['ra_wealth_correlation']
            mean_corr = grouped.mean()
            std_corr = grouped.std()
            
            steps = mean_corr.index
            ax4.plot(steps, mean_corr.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax4.fill_between(steps, mean_corr - std_corr, mean_corr + std_corr, 
                           alpha=0.2, color=scenario['color'])
    
    ax4.set_title('Risk Aversion vs Wealth Correlation')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Correlation Coefficient')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Final Wealth Distribution by Scenario
    ax5 = plt.subplot(3, 4, 5)
    scenario_names = [s['name'] for s in scenarios]
    wealth_by_scenario = [final_agents_df[final_agents_df['scenario'] == name]['wealth'].values 
                         for name in scenario_names]
    
    bp = ax5.boxplot(wealth_by_scenario, labels=scenario_names, patch_artist=True)
    for patch, scenario in zip(bp['boxes'], scenarios):
        patch.set_facecolor(scenario['color'])
        patch.set_alpha(0.7)
    
    ax5.set_title('Final Wealth Distribution by Scenario')
    ax5.set_xlabel('Scenario')
    ax5.set_ylabel('Final Wealth')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. Risk Aversion vs Final Wealth Scatter
    ax6 = plt.subplot(3, 4, 6)
    for scenario in scenarios:
        scenario_data = final_agents_df[final_agents_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            ax6.scatter(scenario_data['risk_aversion'], scenario_data['wealth'], 
                       alpha=0.6, label=scenario['name'], color=scenario['color'], s=30)
    
    ax6.set_title('Risk Aversion vs Final Wealth')
    ax6.set_xlabel('Risk Aversion')
    ax6.set_ylabel('Final Wealth')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # 7. Average Final Metrics by Scenario
    ax7 = plt.subplot(3, 4, 7)
    final_summary = final_agents_df.groupby('scenario').agg({
        'wealth': ['mean', 'std'],
        'risk_aversion': 'mean'
    }).round(2)
    
    scenarios_order = [s['name'] for s in scenarios]
    mean_wealth = [final_summary.loc[name, ('wealth', 'mean')] if name in final_summary.index else 0 
                   for name in scenarios_order]
    std_wealth = [final_summary.loc[name, ('wealth', 'std')] if name in final_summary.index else 0 
                  for name in scenarios_order]
    colors = [s['color'] for s in scenarios]
    
    bars = ax7.bar(scenarios_order, mean_wealth, yerr=std_wealth, 
                   color=colors, alpha=0.7, capsize=5)
    ax7.set_title('Average Final Wealth by Scenario')
    ax7.set_xlabel('Scenario')
    ax7.set_ylabel('Average Final Wealth')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3)
    
    # 8. Wealth vs Risk Aversion Density Plot
    ax8 = plt.subplot(3, 4, 8)
    if not final_agents_df.empty:
        scatter = ax8.scatter(final_agents_df['risk_aversion'], final_agents_df['wealth'], 
                             c=final_agents_df['wealth'], cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax8, label='Wealth')
    
    ax8.set_title('Wealth Density by Risk Aversion')
    ax8.set_xlabel('Risk Aversion')
    ax8.set_ylabel('Final Wealth')
    ax8.grid(True, alpha=0.3)
    
    # 9-12. Individual scenario evolution plots
    for i, scenario in enumerate(scenarios[:4]):
        ax = plt.subplot(3, 4, 9 + i)
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        
        if not scenario_data.empty:
            # Plot multiple metrics for this scenario
            grouped_ra = scenario_data.groupby('step')['avg_risk_aversion'].mean()
            grouped_wealth = scenario_data.groupby('step')['total_wealth'].mean()
            
            # Normalize wealth for comparison
            if grouped_wealth.max() != 0:
                normalized_wealth = grouped_wealth / grouped_wealth.max()
            else:
                normalized_wealth = grouped_wealth
            
            ax.plot(grouped_ra.index, grouped_ra.values, 
                   label='Avg Risk Aversion', color=scenario['color'], linewidth=2)
            ax.plot(normalized_wealth.index, normalized_wealth.values, 
                   label='Normalized Wealth', color=scenario['color'], linestyle='--', linewidth=2)
        
        ax.set_title(f'{scenario["name"]} Evolution')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for scenario in scenarios:
        scenario_data = final_agents_df[final_agents_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            print(f"\n{scenario['name']} (RA bounds: {scenario['bounds']}):")
            print(f"  Number of agents: {len(scenario_data)}")
            print(f"  Average Risk Aversion: {scenario_data['risk_aversion'].mean():.3f} ± {scenario_data['risk_aversion'].std():.3f}")
            print(f"  Average Final Wealth: {scenario_data['wealth'].mean():.2f} ± {scenario_data['wealth'].std():.2f}")
            print(f"  Wealth Range: [{scenario_data['wealth'].min():.2f}, {scenario_data['wealth'].max():.2f}]")
            
            # Calculate correlation within scenario
            if len(scenario_data) > 1:
                corr = scenario_data['risk_aversion'].corr(scenario_data['wealth'])
                print(f"  RA-Wealth Correlation: {corr:.3f}")

if __name__ == "__main__":
    print("Starting comprehensive poker simulation analysis...")
    print("This will run multiple scenarios with different risk aversion bounds.")
    
    # Run the comprehensive analysis
    scenarios, evolution_df, final_agents_df = run_multiple_simulations()
    
    # Create all the analysis plots
    create_comprehensive_analysis(scenarios, evolution_df, final_agents_df)
    
    print("\nAnalysis complete! Check the plots for insights into risk aversion effects.")