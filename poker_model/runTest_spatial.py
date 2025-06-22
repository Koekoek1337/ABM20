from game_classes.model import SpatialModel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

if __name__ == "__main__":
    M = SpatialModel(n_players=9, n_rounds=1000, gridDim=(3,3), seed=42)
    GAMES = 10  # Run more games to see evolution
    
    for game in range(GAMES):
        print(f"Running game {game+1}/{GAMES}")
        M.step()

    # Create enhanced Matplotlib animation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update_plot(frame_idx):
        ax.clear()
        
        if frame_idx >= len(M.profit_grids):
            frame_idx = len(M.profit_grids) - 1
            
        grid = M.profit_grids[frame_idx]
        
        # Create the heatmap
        im = ax.imshow(grid, cmap='RdYlBu_r', interpolation='nearest', 
                      vmin=np.min(M.profit_grids), vmax=np.max(M.profit_grids))
        
        # Add colorbar with values
        if not hasattr(ax, '_colorbar'):
            ax._colorbar = plt.colorbar(im, ax=ax, label='Profit/Loss')
        else:
            ax._colorbar.update_normal(im)
        
        # Get agent positions and strategies at this frame
        agents_info = {}
        for agent in M.agents:
            if hasattr(agent, 'pos') and agent.pos is not None:
                x, y = agent.pos
                agents_info[(x, y)] = {
                    'profit': grid[x, y],
                    'strategy': agent.strategy,
                    'agent': agent
                }
        
        # Add text annotations for each cell
        for x in range(M.gridDim[0]):
            for y in range(M.gridDim[1]):
                # Display profit value
                profit_val = grid[x, y]
                
                if (x, y) in agents_info:
                    agent_info = agents_info[(x, y)]
                    strategy = agent_info['strategy']
                    
                    # Show key strategy parameters (averaged across hands)
                    avg_raise = np.mean(strategy[0, :])  # RAISELIM
                    avg_call = np.mean(strategy[1, :])   # CALL_LIM
                    avg_bluff_prob = np.mean(strategy[2, :]) / 400 * 100  # BLUFF_PROB as %
                    avg_bluff_size = np.mean(strategy[3, :])  # BLUFF_SIZE
                    
                    # Create text to display
                    text = f"${profit_val:.0f}\n"
                    text += f"R:{avg_raise:.0f}\n"
                    text += f"C:{avg_call:.0f}\n" 
                    text += f"Bl:{avg_bluff_prob:.0f}%\n"
                    text += f"Bs:{avg_bluff_size:.0f}"
                    
                    # Choose text color based on background
                    text_color = 'white' if profit_val < 0 else 'black'
                    
                    ax.text(y, x, text, ha='center', va='center', 
                           fontsize=8, color=text_color, weight='bold')
                else:
                    # Empty cell
                    ax.text(y, x, f"${profit_val:.0f}\nEmpty", 
                           ha='center', va='center', fontsize=8, color='gray')
        
        # Set labels and title
        ax.set_title(f"Step {frame_idx + 1} - Agent Strategies & Profits\n"
                    f"R=Raise Limit, C=Call Limit, Bl=Bluff %, Bs=Bluff Size", 
                    fontsize=12, pad=20)
        ax.set_xlabel("Y Coordinate")
        ax.set_ylabel("X Coordinate")
        ax.set_xticks(range(M.gridDim[1]))
        ax.set_yticks(range(M.gridDim[0]))
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, M.gridDim[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, M.gridDim[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        
        return [im]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_plot, frames=len(M.profit_grids), 
                                 interval=10, repeat=True, blit=False)
    
    # Add overall statistics
    total_profits = [np.sum(grid) for grid in M.profit_grids]
    print(f"\nTotal profit evolution: {total_profits}")
    print(f"Final agent positions and strategies:")
    
    for i, agent in enumerate(M.agents):
        if hasattr(agent, 'pos'):
            print(f"Agent {i} at {agent.pos}:")
            strategy = agent.strategy
            print(f"  Avg Raise Limit: {np.mean(strategy[0, :]):.1f}")
            print(f"  Avg Call Limit: {np.mean(strategy[1, :]):.1f}")
            print(f"  Avg Bluff Prob: {np.mean(strategy[2, :]/400*100):.1f}%")
            print(f"  Avg Bluff Size: {np.mean(strategy[3, :]):.1f}")
            print(f"  Final Profit: {M.profit_grids[-1][agent.pos]:.1f}")
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Save the animation
    # ani.save('poker_evolution.gif', writer='pillow', fps=1)