import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from .agents import AuctionPlayer
from .game   import Game

class SpatialModel(Model):
    def __init__(self, n_players=9, n_rounds=1, games_per_step=1, gridDim=(3,3), seed=None):
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
        for i in range(n_players):
            a = AuctionPlayer(self, update_mode="uniform", update_parms={"max_stepsize": 1})
            
            # Find empty cell and place agent
            x = self.rng.integers(0, gridDim[0])
            y = self.rng.integers(0, gridDim[1])
            
            # Try to find an empty cell (since capacity=1 conceptually)
            attempts = 0
            while len(self.grid.get_cell_list_contents([(x, y)])) > 0 and attempts < 100:
                x = self.rng.integers(0, gridDim[0])
                y = self.rng.integers(0, gridDim[1])
                attempts += 1
            
            # Place agent on grid - agents are automatically added to self.agents
            self.grid.place_agent(a, (x, y))

        print(f"Created {len(self.agents)} agents on {gridDim[0]}x{gridDim[1]} grid")

    def step(self):
        """
        One step of the model: All agents play together in the same tournament.
        This is equivalent to calling game.round() with all agents.
        """
        self.step_count += 1
        
        if len(self.agents) < 2:
            print("Not enough agents for a tournament")
            return
        
        # Reset all agent balances before the step
        for agent in self.agents:
            agent.balances[:] = 0
        
        # Setup the game with ALL agents
        all_agents = list(self.agents)
        self.game.setup(all_agents)
        
        for game_num in range(self.games_per_step):
            for round_num in range(self.n_rounds):
                self.game.round()
                self.profit_grids.append(self._get_profit_grid())
                ## mutate agents' strategies after each round
                for agent in all_agents:
                    if agent.update_mode == "uniform":
                        agent._upd_uniform(**agent.update_parms)
        
        # After all rounds, determine who lost (minimum total balance)
        yields = [agent.balances.sum() for agent in all_agents]
        if yields:
            min_yield = min(yields)
            total_pot = sum(yields)
            self.datacollect.append(total_pot)
            

        # Call gameEnd for all agents, marking losers
        for agent, yield_val in zip(all_agents, yields):
            is_loser = (yield_val == min_yield)
            agent.gameEnd(is_loser)
        
        # Record current profit state
        print(f"Step {self.step_count}: Profit grid recorded.")
        
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: Yields = {yields}")

    def get_neighbors(self, agent):
        """Get Von Neumann neighbors of an agent (for future spatial interactions)"""
        neighbors = self.grid.get_neighbors(
            agent.pos, 
            moore=False,  # Von Neumann (not Moore)
            include_center=False,
            radius=1
        )
        return neighbors

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