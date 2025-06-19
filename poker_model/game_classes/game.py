import numpy as np
import pandas as pd
from operator import itemgetter
import PokerPy
from .deck import Deck
from .agents import AuctionPlayer, RAISELIM, CALL_LIM
from typing import List, Tuple, Type

Card = Type[PokerPy.Card]

class Game:
    def __init__(self, players: List[AuctionPlayer], rng:np.random.Generator):
        self.RNG = rng
        self.players = players
        self._nPlayers = len(players)
        
        # Set number of opponents for each player
        for player in players:
            player.n_opponents = self._nPlayers - 1
            
        self.deck = Deck(nPlayers=self._nPlayers)
        self.deck.shuffle(self.RNG)

    def round(self):
        self.deck.shuffle(self.RNG)
        holes = [[self.deck.deal() for _ in range(Deck.HOLE)] for _ in range(self._nPlayers)]
        community = [self.deck.deal() for _ in range(Deck.COMMUNITY)]

        strats = np.zeros((2, self._nPlayers), float)
        handScores = []
        opponent_actions = ['call'] * self._nPlayers  # Placeholder for actual actions
        
        for i, player in enumerate(self.players):
            hand = PokerPy.get_best_hand(holes[i] + community)
            handName = hand.hand_type
            handScores.append(hand.hand_heuristic())
            
            # Update opponent history
            player.update_opponent_history(opponent_actions)
            
            # Pass hole cards and community cards to getBet
            strats[:, i] = player.getBet(handName, holes[i], community)
            
            # Record action for opponent modeling
            if strats[RAISELIM, i] > 0:
                opponent_actions[i] = 'raise'
            else:
                opponent_actions[i] = 'call'

        pot = np.ones_like(strats[0]) * np.min(strats[RAISELIM])
        strats = strats - pot
        maxBet = max(strats[RAISELIM])
        folds = strats[CALL_LIM] < maxBet

        pot += folds * strats[CALL_LIM]
        pot += np.invert(folds) * maxBet
        handScores = handScores * np.invert(folds)
        winners = handScores == np.max(handScores)
        payOuts = winners * (np.sum(pot) / np.sum(winners)) - pot

        for i, player in enumerate(self.players):
            player.handResult(payOuts[i])

    def play(self, games: int, rounds: int, verbose=True):
        data = []
        for game in range(games):
            for round in range(rounds):
                self.round()
            data.append([player.getData() for player in self.players])
            yields = [np.sum(handYields) for strat, handYields in data[-1]]
            loss = yields == np.min(yields)

            if verbose: 
                print(f"Game {game+1} results:")
                for i, y in enumerate(yields):
                    print(f"Player {i}: {y:.2f}")
                print(f"Eliminated players: {loss}")

            for i, player in enumerate(self.players): 
                player.gameEnd(loss[i])

        return data