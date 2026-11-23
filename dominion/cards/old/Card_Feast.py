#!/usr/bin/env python

import unittest

from dominion import Card, Game, Piles, Player


###############################################################################
class Card_Feast(Card.Card):
    def __init__(self):
        Card.Card.__init__(self)
        self.cardtype = Card.CardType.ACTION
        self.base = Card.CardExpansion.DOMINION
        self.desc = "Trash this card, Gain a card costing up to 5"
        self.name = "Feast"
        self.cost = 4

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Trash this card. Gain a card costing up to 5"""
        if self.location == Piles.PLAYED and self in player.piles[Piles.PLAYED]:
            player.trash_card(self)
        candidates = player.cards_under(5)
        if candidates:
            player.plr_gain_card(5, cardsrc=candidates, force=True)


###############################################################################
class TestFeast(unittest.TestCase):
    def setUp(self):
        self.g = Game.TestGame(
            numplayers=1,
            initcards=["Feast"],
            oldcards=True,
            badcards=["Den of Sin", "Ghost Town", "Duchess"],
        )
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Feast")
        self.plr.add_card(self.card, Piles.HAND)

    def test_trash_and_gain(self):
        self.plr.test_input = ["Get Duchy"]
        self.plr.play_card(self.card)
        self.assertIn(self.card, self.g.trash_pile)
        self.assertTrue(self.plr.piles[Piles.PLAYED].is_empty())
        self.assertIn("Duchy", self.plr.piles[Piles.DISCARD])

    def test_replay_still_gains(self):
        self.plr.test_input = ["Get Duchy", "Get Silver"]
        self.plr.play_card(self.card)
        self.plr.play_card(self.card, cost_action=False, move_card=False)
        self.assertEqual(sum(c is self.card for c in self.g.trash_pile), 1)
        self.assertIn("Duchy", self.plr.piles[Piles.DISCARD])
        self.assertIn("Silver", self.plr.piles[Piles.DISCARD])

    def test_no_eligible_cards(self):
        from unittest.mock import patch

        with patch.object(self.plr, "cards_under", return_value=[]):
            self.plr.play_card(self.card)
        self.assertIn(self.card, self.g.trash_pile)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
