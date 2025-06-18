import numpy as np
import pandas as pd
from operator import itemgetter

import PokerPy

from .deck import Deck
from .agents import AuctionPlayer, RAISELIM, CALL_LIM

from typing import List, Tuple

type Card = PokerPy.Card

class Game:
    def __init__(self, players: List[AuctionPlayer], rng:np.random.Generator):
        self.RNG=rng
        self.players = players
        self._nPlayers = len(players)

        self.deck = Deck(nPlayers=self._nPlayers)
        self.deck.shuffle(self.RNG)

    
    def round(self):
        """
        A round of texas hold 'em poker, resolving at the river stage (eg. 5 community cards on table)
        """
        self.deck.shuffle(self.RNG)

        holes    = [[self.deck.deal() for j in range(Deck.HOLE)] for i in range(self._nPlayers)]
        community = [self.deck.deal() for i in range(Deck.COMMUNITY)]

        strats = np.zeros((2, self._nPlayers), float)
        handScores = []
        for i, player in enumerate(self.players):
            hand = PokerPy.get_best_hand(holes[i] + community)
            handName = hand.hand_type
            handScores.append(hand.hand_heuristic())
            
            strats[:, i] = player.getBet(handName)

        # Game will always raise up to the lowest bet. Strats can be adjusted as to start where the lowest bet ends.
        pot    = np.ones_like(strats[0,:]) * np.min(strats[RAISELIM])
        strats = strats - pot

        maxBet = max(strats[RAISELIM,:])
        folds = strats[CALL_LIM, :] < maxBet

        # Bets increase up to CALL_LIM for folding players
        # Otherwise, up to the highest maxbet
        pot += folds * strats[CALL_LIM, :]
        pot += np.invert(folds) * maxBet

        handScores = handScores * np.invert(folds) # Multiply all folded hands by 0
        winners    = handScores == np.max(handScores)

        payOuts    = winners * (np.sum(pot) / np.sum(winners)) - pot

        for i, player in enumerate(self.players):
            player.handResult(payOuts[i])
        

    def play(self, games: int, rounds: int, verbose = True):
        """
        Play an amount of games consisting of an amount of rounds. Players will adjust strategies between
        rounds depending on whether they ended up last. 

        For testing purposes.
        """
        data = []
        for game in range(games):
            for round in range(rounds):
                self.round()
            data.append([player.getData() for player in self.players])
            yields = [np.sum(handYields) for strat, handYields in data[-1]]
            loss   = yields == np.min(yields)

            print(loss)
            for i, player in enumerate(self.players): player.gameEnd(loss[i])
            
            if verbose: print(*yields)

        return data
    


