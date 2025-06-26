from poker_model.game_classes.deck import Deck

import pandas as pd

def test_cardProperties():
    """
    Test deck integrity
    """    
    deck = Deck()

    cards = [deck.deal() for _ in range(deck.nCards)]
    suits = [card[0] for card in cards]
    ranks = [card[1] for card in cards]

    assert deck.nCards == 52, "Default settings does not give list of 52 standard playing cards"
    assert len(cards) == deck.nCards, "Amount of cards is different than expected"
    assert len(pd.unique(cards)) == deck.nCards, "Duplicate cards in deck"
    assert len(pd.unique(suits)) == deck.suits, "Amount of suits different than expected"
    assert len(pd.unique(ranks)) == deck.ranks, "Amount of ranks different than expected"

def test_overflowDeal():
    """
    Test wheter dealing more cards than expected is caught as expected
    """
    deck = Deck()

    cards = [deck.deal() for _ in range(deck.nCards + 1)]
    assert cards[-1] is None, "Unexpected behavior in overflow deal"
