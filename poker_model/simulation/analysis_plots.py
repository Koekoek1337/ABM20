import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Tuple, Any
from pandas import DataFrame

def plot_comprehensive_analysis(scenarios: List[Dict[str, Any]], evolution_df: DataFrame, final_agents_df: DataFrame):
    """
    Plots a comprehensive analysis of multiple model scenarios.
    Will throw error if more scenarios are present than fit on the plot

    Args:
        scenarios: Scenario dictionaries used for simulation initialization.
        evolution_df: TODO
        final final_agents_df: TODO

    At time of writing: [5 plots]
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))

    subplots = [(3, 4, i) for i in range(1, 13)]
    subplot_id = 0

    # 1. Evolution of Average Risk Aversion
    p_avgRiskAversion(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 2. Evolution of Total Wealth
    # p_totalWealth(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    # subplot_id += 1
    # 3. Wealth Inequality (Gini Coefficient)
    p_gini(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 4. Risk Aversion vs Wealth Correlation
    p_riskWealthCorrelation(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 5. Final Wealth Distribution by Scenario
    p_wealthDistribution(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 6. Risk Aversion vs Final Wealth Scatter
    p_riskWealthScatter(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 7. Average Final Metrics by Scenario
    p_finMetric(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 8. Wealth vs Risk Aversion Density Plot
    p_wealthDensity(fig, scenarios, evolution_df, final_agents_df, subplots[subplot_id])
    subplot_id += 1
    # 9-12. Individual scenario evolution plots
    for scenario in scenarios:
        p_individualScenario(fig, scenario, evolution_df, subplots[subplot_id])
        subplot_id += 1

    plt.tight_layout()
    plt.show()

###{INDIVIDUAL PLOTTING FUNCTIONS}###
def p_avgRiskAversion(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
                      final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """
    Evolution of Average Risk Aversion
    """
    plt.style.use('default')
    sns.set_palette("husl")
    ax = fig.add_subplot(*subplot)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            # Group by step and calculate mean/std across replications
            grouped = scenario_data.groupby('step')['avg_risk_aversion']
            mean_ra = grouped.mean()
            std_ra = grouped.std()
            
            steps = mean_ra.index
            ax.plot(steps, mean_ra.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax.fill_between(steps, mean_ra - std_ra, mean_ra + std_ra, 
                           alpha=0.2, color=scenario['color'])
    
    ax.set_title('Evolution of Average Risk Aversion')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Average Risk Aversion')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def p_totalWealth(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
                  final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    ax = fig.add_subplot(*subplot)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            grouped = scenario_data.groupby('step')['total_wealth']
            mean_wealth = grouped.mean()
            std_wealth = grouped.std()
            
            steps = mean_wealth.index
            ax.plot(steps, mean_wealth.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax.fill_between(steps, mean_wealth - std_wealth, mean_wealth + std_wealth, 
                           alpha=0.2, color=scenario['color'])
    
    ax.set_title('Evolution of Total Wealth')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Total Wealth')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def p_gini(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
           final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """Wealth Inequality (Gini Coefficient)"""
    ax = fig.add_subplot(*subplot)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            grouped = scenario_data.groupby('step')['wealth_gini']
            mean_gini = grouped.mean()
            std_gini = grouped.std()
            
            steps = mean_gini.index
            ax.plot(steps, mean_gini.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax.fill_between(steps, mean_gini - std_gini, mean_gini + std_gini, 
                           alpha=0.2, color=scenario['color'])
    
    ax.set_title('Wealth Inequality Evolution (Gini)')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Gini Coefficient')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def p_riskWealthCorrelation(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
                            final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """Risk Aversion vs Wealth Correlation"""
    ax = fig.add_subplot(*subplot)
    for scenario in scenarios:
        scenario_data = evolution_df[evolution_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            grouped = scenario_data.groupby('step')['ra_wealth_correlation']
            mean_corr = grouped.mean()
            std_corr = grouped.std()
            
            steps = mean_corr.index
            ax.plot(steps, mean_corr.values, label=scenario['name'], 
                    color=scenario['color'], linewidth=2)
            ax.fill_between(steps, mean_corr - std_corr, mean_corr + std_corr, 
                           alpha=0.2, color=scenario['color'])
    
    ax.set_title('Risk Aversion vs Wealth Correlation')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Correlation Coefficient')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def p_wealthDistribution(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
                         final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """Final Wealth Distribution by Scenario"""
    ax = fig.add_subplot(*subplot)
    scenario_names = [s['name'] for s in scenarios]
    wealth_by_scenario = [final_agents_df[final_agents_df['scenario'] == name]['wealth'].values 
                         for name in scenario_names]
    
    bp = ax.boxplot(wealth_by_scenario, labels=scenario_names, patch_artist=True)
    for patch, scenario in zip(bp['boxes'], scenarios):
        patch.set_facecolor(scenario['color'])
        patch.set_alpha(0.7)
    
    ax.set_title('Final Wealth Distribution by Scenario')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Final Wealth')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def p_riskWealthScatter(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
                        final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """Risk Aversion vs Final Wealth Scatter"""
    ax = fig.add_subplot(*subplot)
    for scenario in scenarios:
        scenario_data = final_agents_df[final_agents_df['scenario'] == scenario['name']]
        if not scenario_data.empty:
            ax.scatter(scenario_data['risk_aversion'], scenario_data['wealth'], 
                       alpha=0.6, label=scenario['name'], color=scenario['color'], s=30)
    
    ax.set_title('Risk Aversion vs Final Wealth')
    ax.set_xlabel('Risk Aversion')
    ax.set_ylabel('Final Wealth')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def p_finMetric(fig: plt.Figure, scenarios: List[Dict[str, Any]], evolution_df: DataFrame, 
                final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """Average Final Metrics by Scenario"""
    ax = fig.add_subplot(*subplot)
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
    
    bars = ax.bar(scenarios_order, mean_wealth, yerr=std_wealth, 
                   color=colors, alpha=0.7, capsize=5)
    ax.set_title('Average Final Wealth by Scenario')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Average Final Wealth')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def p_wealthDensity(fig: plt.Figure, scenario: Dict[str, Any], evolution_df: DataFrame, 
                    final_agents_df: DataFrame, subplot: Tuple[int, int, int]):
    """Wealth vs Risk Aversion Density Plot"""
    ax = fig.add_subplot(*subplot)
    if not final_agents_df.empty:
        scatter = ax.scatter(final_agents_df['risk_aversion'], final_agents_df['wealth'], 
                             c=final_agents_df['wealth'], cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Wealth')
    
    ax.set_title('Wealth Density by Risk Aversion')
    ax.set_xlabel('Risk Aversion')
    ax.set_ylabel('Final Wealth')
    ax.grid(True, alpha=0.3)


def p_individualScenario(fig: plt.Figure, scenario: Dict[str, Any], evolution_df: DataFrame, 
                         subplot: Tuple[int, int, int]):
    """Individual scenario evolution plot. Takes single scenario rather than a list of scenarios."""
    ax = fig.add_subplot(*subplot)
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