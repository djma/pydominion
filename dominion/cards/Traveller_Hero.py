#!/usr/bin/env python

import unittest

from dominion import Game, Card, Piles


###############################################################################
class Card_Hero(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.TRAVELLER]
        self.base = Card.CardExpansion.ADVENTURE
        self.desc = "+2 Coin, Gain a Treasure; Discard to replace with Champion"
        self.name = "Hero"
        self.purchasable = False
        self.insupply = False
        self.coin = 2
        self.cost = 5
        self.numcards = 5

    def special(self, game, player) -> None:
        """Gain a treasure"""
        player.plr_gain_card(cost=None, types={Card.CardType.TREASURE: True})

    def hook_discard_this_card(self, game, player, source):
        """Replace with Champion"""
        if source != Piles.PLAYED and source is not player.piles[Piles.PLAYED]:
            return
        player.replace_traveller(self, "Champion")


###############################################################################
class Test_Hero(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(
            quiet=True, numplayers=1, initcards=["Page"], badcards=["Fool's Gold"]
        )
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Hero")
        self.plr.add_card(self.card, Piles.HAND)

    def test_hero(self) -> None:
        """Play a hero"""
        self.plr.test_input = ["get gold"]
        self.plr.play_card(self.card)
        try:
            self.assertEqual(self.plr.coins.get(), 2)
            self.assertIn("Gold", self.plr.piles[Piles.DISCARD])
        except AssertionError:  # pragma: no cover
            self.g.print_state()
            raise

    def test_exchange_only_from_play(self):
        from unittest.mock import patch

        with patch.object(self.plr, "replace_traveller") as exchange:
            for source in (Piles.HAND, Piles.EXILE, self.plr.piles[Piles.HAND], self.plr.piles[Piles.EXILE]):
                self.card.hook_discard_this_card(self.g, self.plr, source)
            exchange.assert_not_called()
            for source in (Piles.PLAYED, self.plr.piles[Piles.PLAYED]):
                self.card.hook_discard_this_card(self.g, self.plr, source)
            self.assertEqual(exchange.call_count, 2)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
