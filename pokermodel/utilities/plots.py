
import matplotlib.pyplot as plt

def plot_wealth_evolution(wealth_history):
    """Plot wealth evolution over time"""
    plt.figure(figsize=(12, 8))
    
    for agent_name, wealth_list in wealth_history.items():
        plt.plot(wealth_list, label=agent_name, linewidth=2)
    
    plt.xlabel('Round')
    plt.ylabel('Wealth')
    plt.title('Wealth Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
