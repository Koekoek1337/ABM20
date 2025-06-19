import numpy as np
import PokerPy

from typing import Optional, List, Tuple, Union, Type

Card = Type[PokerPy.Card]

class Deck:
    """
    Deck class for poker game

    Properties
        _basedeck: Storage of Suit
    """
    HOLE  = 2

    FLOP  = 3
    TURN  = 1
    RIVER = 1
    COMMUNITY = FLOP + TURN + RIVER
    
    def __init__(self, suits=4, ranks=13, nPlayers: Optional[int] = None) -> None:
        self._basedeck = self.buildDeck(suits, ranks)
        """Tuple of two ints representing the suit and rank of a card"""
        self.suits  = suits
        """Amount of suits in deck"""
        self.ranks  = ranks
        """Amount or ranks in deck"""
        self.nCards = suits * ranks
        """Total amount of cards in the deck"""

        dealBufferSize = self.nCards
        if nPlayers is not None: # As a memory optimization, only the required amount of playing cards can be buffered.
            dealBufferSize = self.COMMUNITY + self.HOLE * nPlayers

        self._dealbuffer = np.arange(dealBufferSize)
        """The indexes of shuffled cards that can be dealt from the deck"""
        self.dealIndex  = 0
        """"The index of the shuffled card to be dealt"""


    def buildDeck(self, suits:int, ranks:int) -> List[Card]:
        """
        Builds a deck of an arbitrary number of suits and ranks.
        """
        SUITS = ["D", "H", "C", "S"]
        RANKS = ["A"] + [str(x) for x in range(2, 11)] + ["J", "Q", "K"]
        deck: List[Tuple[int,int]] = []
        for suit in SUITS[0:ranks]:
            for rank in RANKS[0:ranks]:
                deck.append(PokerPy.Card(rank+suit))

        return deck
    

    def shuffle(self, rng: np.random.Generator) -> None:
        """
        Randomly sample indexes from the deck until the dealbuffer is replaced. 
        In-place operation on dealbuffer.
        """
        np.copyto(self._dealbuffer, rng.choice(self.nCards, len(self._dealbuffer), False))
        self.dealIndex = 0


    def deal(self) -> Union[Card, None]:
        """
        Returns a single card from the dealbuffer and advances the bufferindex by 1.
        Returns None if a card is dealt at the end of the index.
        """
        card = None
        if self.dealIndex < len(self._dealbuffer):
            card = self._basedeck[self._dealbuffer[self.dealIndex]]
            self.dealIndex += 1
        return card
        