import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from .agents import AuctionPlayer, RiskFreeAgent, SingleRiskAgent
from .game   import Game


from typing import List

class SpatialModel(Model):
    def __init__(self, gridDim=(10,10), n_rounds=1, games_per_step=1, seed=None, agentType = RiskFreeAgent):
        super().__init__(seed=seed)
        self.rng          = np.random.default_rng(seed)
        self.gridDim      = gridDim
        self.n_rounds     = n_rounds  # rounds per game
        self.games_per_step = games_per_step  # games per step
        
        # Create grid - using MultiGrid for better compatibility
        self.grid = MultiGrid(gridDim[0], gridDim[1], torus=True)
        
        self.game         = Game([], self.rng)
        self.datacollect  = []               # net pot per round
        self.profit_grids = []               # snapshots of profit heatmaps
        self.step_count   = 0

        # create & place players
        for x in range(gridDim[0]):
            for y in range(gridDim[1]):
                a = agentType(self, update_mode="uniform", update_parms={"max_stepsize": 1})

                # Place agent on grid - agents are automatically added to self.agents
                self.grid.place_agent(a, (x, y))

        print(f"Created {len(self.agents)} agents on {gridDim[0]}x{gridDim[1]} grid")

    def step(self):
        """
        One step of the model: Agents will find their neighbors and play a match of poker against them
        """
        self.step_count += 1
        
        # Reset all agent balances before the step
        for agent in self.agents:
            agent.balances[:] = 0
        self.agents.do("challenge")

        # After all rounds, determine who lost (minimum total balance)
        yields = [agent.balances.sum() for agent in self.agents]
            
        # Call gameEnd for all agents, marking losers
        self.agents.shuffle_do("neighborhood_adapt")
        
        # Record current profit state
        self.profit_grids.append(self._get_profit_grid())
        print(f"Step {self.step_count}: Profit grid recorded.")
        
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: Yields = {yields}")

    def _get_profit_grid(self):
        """
        Returns a 2D numpy array of shape gridDim where each cell contains
        the sum of balances of the agent occupying that cell (or 0 if empty).
        """
        matrix = np.zeros(self.gridDim)
        
        # Iterate through all agents and their positions
        for agent in self.agents:
            if hasattr(agent, 'pos') and agent.pos is not None:
                x, y = agent.pos
                matrix[x][y] = agent.balances.sum()
                
        return matrix

    def get_grid_contents(self):
        """Helper method to see what's in each cell"""
        contents = {}
        for x in range(self.gridDim[0]):
            for y in range(self.gridDim[1]):
                cell_contents = self.grid.get_cell_list_contents([(x, y)])
                if cell_contents:
                    contents[(x, y)] = len(cell_contents)
        return contents
    
    def get_agent_strategies(self):
        """Helper method to inspect agent strategies"""
        strategies = {}
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'pos'):
                strategies[agent.pos] = {
                    'agent_id': i,
                    'strategy': agent.strategy.copy(),
                    'total_balance': agent.balances.sum()
                }
        return strategies

class IterativeCompetition(Model):
    def __init__(self, nPlayers, agentUpdateMode="null", updateParms={"maxStepsize":0.1}, *args, 
                 seed = None, rng = None, **kwargs):
        super().__init__(*args, seed=seed, rng=rng, **kwargs)

        AuctionPlayer.create_agents(self, nPlayers, update_mode=agentUpdateMode, updateParms=updateParms)
        self.game = Game([player for player in self.agents], self.rng)