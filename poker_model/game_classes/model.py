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
        self.n_rounds     = n_rounds
        self.games_per_step = games_per_step
        
        self.grid = MultiGrid(gridDim[0], gridDim[1], torus=True)
        self.game = Game([], self.rng)
        self.datacollect = []
        self.profit_grids = []
        self.step_count = 0

        self.custom_agents = []  # manual tracking

        for _ in range(n_players):
            # Random risk aversion for now
            # TODO: decide how to act on this
            risk_aversion = self.rng.uniform(0, 1)
            a = AuctionPlayer(
                self,
                update_mode="neighbor_copy",
                update_parms={"max_stepsize": 1},
                risk_aversion=risk_aversion
            )
            for _ in range(100):  # max 100 placement attempts
                x = self.rng.integers(0, gridDim[0])
                y = self.rng.integers(0, gridDim[1])
                if len(self.grid.get_cell_list_contents([(x, y)])) == 0:
                    self.grid.place_agent(a, (x, y))
                    self.custom_agents.append(a)  # track agent
                    # a.pos = (x, y)
                    break

        print(f"Created {len(self.custom_agents)} agents on {gridDim[0]}x{gridDim[1]} grid")


    def step(self):
        self.step_count += 1
        
        if len(self.custom_agents) < 2:
            print("Not enough agents for a tournament")
            return

        for agent in self.custom_agents:
            agent.balances[:] = 0

        all_agents = list(self.custom_agents)
        self.game.setup(all_agents)

        for _ in range(self.games_per_step):
            for _ in range(self.n_rounds):
                self.game.round()
                self.profit_grids.append(self._get_profit_grid())

                # Strategy update
                for agent in all_agents:

                    agent.update_strategy(self)  # unified strategy update interface

        yields = [agent.balances.sum() for agent in all_agents]
        if yields:
            min_yield = min(yields)
            self.datacollect.append(sum(yields))

        for agent, y in zip(all_agents, yields):
            agent.gameEnd(y == min_yield)

        print(f"Step {self.step_count}: Profit grid recorded.")
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: Yields = {yields}")

    def get_neighbors(self, agent):
        return self.grid.get_neighbors(
            agent.pos,
            moore=False,
            include_center=False,
            radius=1
        )

    def _get_profit_grid(self):
        matrix = np.zeros(self.gridDim)
        for agent in self.custom_agents:
            if hasattr(agent, 'pos') and agent.pos is not None:
                x, y = agent.pos
                matrix[x][y] = agent.balances.sum()
        return matrix

    def get_grid_contents(self):
        contents = {}
        for x in range(self.gridDim[0]):
            for y in range(self.gridDim[1]):
                cell = self.grid.get_cell_list_contents([(x, y)])
                if cell:
                    contents[(x, y)] = len(cell)
        return contents

    def get_agent_strategies(self):
        strategies = {}
        for i, agent in enumerate(self.custom_agents):
            if hasattr(agent, 'pos'):
                strategies[agent.pos] = {
                    'agent_id': i,
                    'strategy': agent.strategy.copy(),
                    'total_balance': agent.balances.sum()
                }
        return strategies
