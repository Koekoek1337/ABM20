
class BaseAgent:
    def __init__(self, name, wealth=1000, risk_aversion=0.5):
        self.name = name
        self.initial_wealth = wealth
        self.wealth = wealth
        self.risk_aversion = risk_aversion
        self.hands_played = 0
        self.hands_won = 0
        self.total_bet = 0
        
    def reset_stats(self):
        self.wealth = self.initial_wealth
        self.hands_played = 0
        self.hands_won = 0
        self.total_bet = 0
