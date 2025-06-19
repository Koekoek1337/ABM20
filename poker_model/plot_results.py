import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd
import os

def plot_simulation_results(results: List[Dict[str, Any]], output_dir="plots"):
    """
    Plot comprehensive results from poker simulations
    Input: List of simulation results, each containing player statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all results into a single DataFrame
    all_data = []
    for sim_idx, sim_result in enumerate(results):
        for player_name, stats in sim_result.items():
            # Add simulation index and player name
            player_data = stats.copy()
            player_data["simulation"] = sim_idx
            player_data["player"] = player_name
            
            # Flatten bluff stats
            if "bluff_stats" in player_data:
                for key, value in player_data["bluff_stats"].items():
                    player_data[f"bluff_{key}"] = value
                del player_data["bluff_stats"]
            
            all_data.append(player_data)
    
    df = pd.DataFrame(all_data)
    
    # Set plot style
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    
    # 1. Wealth Distribution
    plt.figure()
    sns.boxplot(data=df, x="player", y="final_wealth", hue="player")
    plt.title("Final Wealth Distribution by Player")
    plt.xlabel("Player")
    plt.ylabel("Final Wealth")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wealth_distribution.png"))
    plt.close()
    
    # 2. Win Rate Comparison
    plt.figure()
    sns.barplot(data=df, x="player", y="win_rate", ci="sd", capsize=0.1)
    plt.title("Average Win Rate by Player")
    plt.xlabel("Player")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "win_rate.png"))
    plt.close()
    
    # 3. Profit vs Risk Aversion
    plt.figure()
    sns.scatterplot(data=df, x="risk_aversion", y="profit", hue="player", 
                   size="aggression", sizes=(50, 200), alpha=0.7)
    plt.title("Profit vs Risk Aversion")
    plt.xlabel("Risk Aversion")
    plt.ylabel("Profit")
    plt.axhline(0, color="gray", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_vs_risk.png"))
    plt.close()
    
    # 4. Bluffing Statistics
    if "bluff_total_bluffs" in df.columns:
        # Filter players who actually bluffed
        bluff_df = df[df["bluff_total_bluffs"] > 0]
        
        if not bluff_df.empty:
            plt.figure(figsize=(14, 6))
            
            # Bluff Success Rate
            plt.subplot(1, 2, 1)
            sns.barplot(data=bluff_df, x="player", y="bluff_bluff_success_rate", 
                        hue="player", ci="sd", capsize=0.1)
            plt.title("Bluff Success Rate")
            plt.xlabel("Player")
            plt.ylabel("Success Rate")
            plt.ylim(0, 1)
            
            # Bluff Frequency
            plt.subplot(1, 2, 2)
            sns.barplot(data=bluff_df, x="player", y="bluff_frequency", 
                        hue="player", ci="sd", capsize=0.1)
            plt.title("Bluff Frequency")
            plt.xlabel("Player")
            plt.ylabel("Frequency")
            plt.ylim(0, 0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bluff_stats.png"))
            plt.close()
    
    # 5. Wealth Evolution (if per-game data is available)
    if "wealth_history" in df.columns and any(df["wealth_history"].notna()):
        # Create wealth evolution plot
        plt.figure()
        
        for player in df["player"].unique():
            player_data = df[df["player"] == player]
            for idx, row in player_data.iterrows():
                if isinstance(row["wealth_history"], list):
                    plt.plot(row["wealth_history"], label=f"{player} - Sim {row['simulation']}")
        
        plt.title("Wealth Evolution Across Games")
        plt.xlabel("Game Number")
        plt.ylabel("Wealth")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wealth_evolution.png"))
        plt.close()
    
    # 6. Personality Traits Comparison
    traits = ["aggression", "risk_aversion", "rationality", "bluff_frequency", "bluff_detection"]
    if all(trait in df.columns for trait in traits):
        plt.figure(figsize=(14, 8))
        
        for i, trait in enumerate(traits):
            plt.subplot(2, 3, i+1)
            sns.barplot(data=df, x="player", y=trait, ci="sd", capsize=0.1)
            plt.title(trait.replace("_", " ").title())
            plt.xticks(rotation=15)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "personality_traits.png"))
        plt.close()
    
    # 7. Betting Behavior
    plt.figure()
    sns.scatterplot(data=df, x="hands_played", y="total_bet", hue="player", 
                   size="win_rate", sizes=(50, 200), alpha=0.7)
    plt.title("Betting Behavior")
    plt.xlabel("Hands Played")
    plt.ylabel("Total Bet")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "betting_behavior.png"))
    plt.close()
    
    # 8. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 2:
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   cbar_kws={"shrink": .8}, vmin=-1, vmax=1)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
    
    print(f"Saved all plots to {output_dir}/ directory")

if __name__ == "__main__":
    # Example usage
    sample_results = [
        {
            "Player1": {
                "final_wealth": 1200,
                "profit": 200,
                "hands_played": 100,
                "hands_won": 35,
                "win_rate": 0.35,
                "total_bet": 5000,
                "risk_aversion": 0.3,
                "aggression": 0.7,
                "rationality": 2.0,
                "bluff_frequency": 0.15,
                "bluff_detection": 0.6,
                "bluff_stats": {
                    "total_bluffs": 15,
                    "successful_bluffs": 8,
                    "bluff_success_rate": 0.53
                },
                "wealth_history": [1000, 950, 1100, 1200]  # Optional
            },
            "Player2": {
                "final_wealth": 850,
                "profit": -150,
                "hands_played": 100,
                "hands_won": 25,
                "win_rate": 0.25,
                "total_bet": 4800,
                "risk_aversion": 0.6,
                "aggression": 0.4,
                "rationality": 1.8,
                "bluff_frequency": 0.05,
                "bluff_detection": 0.4,
                "bluff_stats": {
                    "total_bluffs": 5,
                    "successful_bluffs": 1,
                    "bluff_success_rate": 0.2
                },
                "wealth_history": [1000, 1050, 900, 850]  # Optional
            }
        }
    ]
    
    plot_simulation_results(sample_results)