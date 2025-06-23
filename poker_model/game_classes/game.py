import numpy as np
import pandas as pd
from operator import itemgetter
import PokerPy

from .deck import Deck
from .agents import AuctionPlayer

from typing import List

class Game:
    def __init__(self, players: List[AuctionPlayer], rng: np.random.Generator):
        self.RNG = rng
        self.deck = Deck()
        self.setup(players)

    def setup(self, players: List[AuctionPlayer]):
        self.players = players
        self._nPlayers = len(players)
        self.deck.shuffle(self.RNG)

    def round(self):
        """
        A round of Texas Hold 'Em resolved at the river.
        """
        self.deck.shuffle(self.RNG)

        holes = [[self.deck.deal() for _ in range(Deck.HOLE)] for _ in range(self._nPlayers)]
        community = [self.deck.deal() for _ in range(Deck.COMMUNITY)]

        bets = []
        handScores = []
        for i, player in enumerate(self.players):
            hand = PokerPy.get_best_hand(holes[i] + community)
            handName = hand.hand_type
            handScores.append(hand.hand_heuristic())

            raise_amt, call_amt = player.getBet(handName)
            bets.append((raise_amt, call_amt))

        raise_amts = np.array([b[0] for b in bets])
        call_amts  = np.array([b[1] for b in bets])

        base_bet = np.min(raise_amts)
        pot = np.ones(self._nPlayers) * base_bet

        adjusted_raises = raise_amts - base_bet
        max_bet = np.max(adjusted_raises)

        folds = call_amts < max_bet
        pot += np.where(folds, call_amts, max_bet)

        # Folded players can't win
        handScores = np.where(folds, 0, handScores)
        winners = handScores == np.max(handScores)

        payOuts = winners * (np.sum(pot) / np.sum(winners)) - pot

        for i, player in enumerate(self.players):
            player.handResult(payOuts[i])

    def play(self, games: int, rounds: int, verbose=True):
        """
        Play several games, each made up of multiple rounds. After each game,
        players may update strategies based on outcomes.
        """
        data = []
        for game in range(games):
            for _ in range(rounds):
                self.round()
            data.append([player.getData() for player in self.players])

            yields = [np.sum(info["balances"]) for info in data[-1]]
            lowest = yields == np.min(yields)

            for i, player in enumerate(self.players):
                player.gameEnd(lowest[i])

            if verbose:
                print(*yields)

        return data
