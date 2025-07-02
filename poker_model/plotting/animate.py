import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from game_classes.model import SpatialModel

DEFAULT_FILELOCATION = "./out_animation.gif"

def createAnimation(Model: SpatialModel, showAnimation = False, savePath=DEFAULT_FILELOCATION):
    animator = PokerAnimator(Model)
    ani = animator.makeAnimation()
    
    if showAnimation:
        plt.show()

    if savePath is None:
        return
    ani.save(savePath, writer='pillow', fps=1)

class PokerAnimator:
    def __init__(self, M: SpatialModel):
        self.fig, self.ax = plt.subplots()
        self.M = M
        self.grid = None
        self.im   = None

        self.cell_annotations = []

        self.initGrid()

    def initGrid(self):
        """
        Initializes gridlines and labels.
        """
        # Add ticks and labels
        self.ax.set_xlabel("Y Coordinate")
        self.ax.set_ylabel("X Coordinate")
        self.ax.set_xticks(range(self.M.gridDim[1]))
        self.ax.set_yticks(range(self.M.gridDim[0]))
        # Add grid lines
        self.ax.set_xticks(np.arange(-0.5, self.M.gridDim[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.M.gridDim[0], 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    def updateGrid(self, frame_idx):
        """
        Update internal profit grid for convenience
        """
        self.grid = self.M.profit_grids[frame_idx]
    
    def update_plot(self, frame_idx):
        """
        Animation plot updater.
        """
        #self.ax.clear()
        
        if frame_idx >= len(self.M.profit_grids):
            frame_idx = len(self.M.profit_grids) - 1
            
        self.updateGrid(frame_idx)
        im = self.updateHeatmap()

        agents_info = self.agentInfo(frame_idx)
        self.updateCellAnnotations(frame_idx, agents_info)
        
        title = self.updateTitle(frame_idx)

        return [im, title] + self.cell_annotations
    
    def updateHeatmap(self):
        """
        Create the plot heatmap if it does not exists, else update it
        """
        if self.im is None:
            # Create the heatmap
            self.im = self.ax.imshow(self.grid, cmap='RdYlBu_r', interpolation='nearest', 
                        vmin=np.min(self.M.profit_grids), vmax=np.max(self.M.profit_grids))
        else:
            self.im.set_data(self.grid)

        # Add colorbar with values
        if not hasattr(self.ax, '_colorbar'):
            self.ax._colorbar = plt.colorbar(self.im, ax=self.ax, label='Profit/Loss')
        else:
            self.ax._colorbar.update_normal(self.im)
        
        return self.im

    def agentInfo(self, frame_idx):
        # Get agent positions and strategies at this frame
        agents_info = {}
        for agent in self.M.agents:
            if hasattr(agent, 'pos') and agent.pos is not None:
                x, y = agent.pos
                agents_info[(x, y)] = {
                    'profit': self.grid[x, y],
                    'strategy': agent.strategy,
                    'agent': agent,
                    'risk_aversion': agent.risk_aversion
                }
        return agents_info
    
    def updateCellAnnotations(self, frame_idx, agents_info):
        """
        Add text annotations for each cell
        TODO: See if datapoints can be made relevant per cell.
        """
        firstRun = False

        if not self.cell_annotations:
            firstRun = True

        xLim = self.M.gridDim[0]
        yLim = self.M.gridDim[1]
        for x in range(xLim):
            for y in range(yLim):
                # Display profit value
                textIndex = x * yLim + y
                profit_val = self.grid[x, y]
                
                if (x, y) in agents_info:
                    agent_info = agents_info[(x, y)]
                    strategy = agent_info['strategy']
                    
                    # Create text to display
                    text  = f"${profit_val:.0f}\n"
                    text += f"RA: {agent_info['risk_aversion']:.2f}\n"

                    # Choose text color based on background
                    text_color = 'white' if profit_val < 0 else 'black'
                    
                    if firstRun: 
                        self.cell_annotations.append(self.ax.text(y, x, text, ha='center', va='center', 
                                                                  fontsize=8, color=text_color, weight='bold')) 
                    else: 
                        self.cell_annotations[textIndex].set_text(text)
                else:
                    # Empty cell
                    if firstRun:
                        self.cell_annotations.append(self.ax.text(y, x, f"${profit_val:.0f}\nEmpty", 
                                                                  ha='center', va='center', fontsize=8, color='gray'))

    def updateTitle(self, frame_idx):
        """Update title with relevant frame number."""
        title = self.ax.set_title(f"Step {frame_idx + 1} - Agent Strategies & Profits\n"
                          f"R=Raise Limit, C=Call Limit, Bl=Bluff %, Bs=Bluff Size", 
                          fontsize=12, pad=20)  

        return title
    
    def makeAnimation(self) -> animation.FuncAnimation:
        ani = animation.FuncAnimation(self.fig, self.update_plot, frames=len(self.M.profit_grids), 
                                 interval=20, repeat=True, blit=True)
        plt.tight_layout()
        return ani
        
    