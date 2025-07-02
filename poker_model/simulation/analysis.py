import pandas as pd
import numpy as np

from simulation.analysis_plots import plot_comprehensive_analysis

from game_classes.model import SpatialModel

def analyze_step(model:SpatialModel, step:int):
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
        nullDict = {
            'step': step,
            'avg_risk_aversion': 0,
            'std_risk_aversion': 0,
            'total_wealth': 0,
            'wealth_gini': 0,
            'max_wealth': 0,
            'min_wealth': 0,
            'ra_wealth_correlation': 0
        }
        return nullDict
    
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

    outDict = {
        'step': step,
        'avg_risk_aversion': avg_ra,
        'std_risk_aversion': std_ra,
        'total_wealth': total_wealth,
        'wealth_gini': wealth_gini,
        'max_wealth': max_wealth,
        'min_wealth': min_wealth,
        'ra_wealth_correlation': ra_wealth_corr
    }
    return outDict


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
    
    plot_comprehensive_analysis(scenarios, evolution_df, final_agents_df)
    
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