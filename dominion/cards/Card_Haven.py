#!/usr/bin/env python

import unittest

from dominion import Game, Card, Piles, Player, OptionKeys


###############################################################################
class Card_Haven(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.DURATION]
        self.base = Card.CardExpansion.SEASIDE
        self.desc = """+1 Card; +1 Action; Set aside a card from your hand face down (under this).
        At the start of your next turn, put it into your hand."""
        self.name = "Haven"
        self.cards = 1
        self.actions = 1
        self.cost = 2
        self._play_batches: list[list[Card.Card]] = []

    def _mat_key(self) -> str:
        return f"haven:{self.uuid}"

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Set aside a card from your hand face down."""
        if not player.piles[Piles.HAND]:
            self.stay_in_play = bool(self._play_batches)
            return
        area = player.get_set_aside_area(self._mat_key())
        if card := player.plr_pick_card(force=True, prompt="Pick card to put into hand next turn"):
            player.remove_card(card)
            area.add(card)
            card.location = Piles.SPECIAL
            self._play_batches.append([card])
        self.stay_in_play = bool(self._play_batches)

    def duration(self, game: Game.Game, player: Player.Player) -> dict[OptionKeys, str]:
        """At the start of your next turn, put it into your hand."""
        area = player.get_set_aside_area(self._mat_key())
        batch = self._play_batches.pop(0) if self._play_batches else []
        for card in batch:
            if card not in area:
                continue
            area.remove(card)
            player.add_card(card, Piles.HAND)
            player.output(f"Pulling {card} out of from haven")
        self.stay_in_play = bool(self._play_batches)
        return {}


###############################################################################
class TestHaven(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Haven", "Throne Room"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Haven")
        self.plr.piles[Piles.DISCARD].set("Copper", "Copper", "Copper", "Copper", "Copper")
        self.plr.piles[Piles.DECK].set("Estate", "Estate", "Estate", "Estate", "Gold")
        self.plr.add_card(self.card, Piles.HAND)

    def test_play_card(self) -> None:
        """Play a haven"""
        self.plr.test_input = ["select gold"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 5)
        self.assertEqual(self.plr.actions.get(), 1)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(self.plr.piles[Piles.PLAYED].size(), 1)
        self.assertIn("Gold", self.plr.piles[Piles.HAND])
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 6)
        self.assertEqual(self.plr.actions.get(), 1)

    def test_lost_throne_returns_only_first_batch(self):
        throne = self.g.get_card_from_pile("Throne Room")
        self.plr.piles[Piles.HAND].set("Gold", "Silver")
        self.plr.piles[Piles.DECK].set("Copper", "Copper")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.add_card(throne, Piles.HAND)
        self.plr.test_input = ["Haven", "Gold", "Silver"]
        self.plr.play_card(throne)
        area = self.plr.get_set_aside_area(self.card._mat_key())
        self.assertEqual(len(area), 2)
        self.plr.move_card(throne, Piles.DISCARD)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertIn("Gold", self.plr.piles[Piles.HAND])
        self.assertIn("Silver", area)
        self.assertNotIn("Silver", self.plr.piles[Piles.HAND])
        self.assertIn(area[0], self.plr.all_cards())

    def test_empty_hand_skips_prompt(self):
        from unittest.mock import patch

        self.plr.piles[Piles.HAND].empty()
        with patch.object(self.plr, "plr_pick_card", side_effect=AssertionError("Empty prompt")):
            self.card.special(self.g, self.plr)
        self.assertFalse(self.card.stay_in_play)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
