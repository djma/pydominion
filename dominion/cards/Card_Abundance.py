#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Abundance """
import unittest
from typing import Any

from dominion import Game, Card, Piles, Player, OptionKeys


###############################################################################
class Card_Abundance(Card.Card):
    """Abundance"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.TREASURE, Card.CardType.DURATION]
        self.base = Card.CardExpansion.PLUNDER
        self.desc = """The next time you gain an Action card: +1 Buy and +$3."""
        self.name = "Abundance"
        self.cost = 4
        self.permanent = True
        self._pending = 0

    def special(self, game: Game.Game, player: Player.Player) -> None:
        self._pending += 1
        self.permanent = True
        self.stay_in_play = True

    def hook_gain_card(
        self, game: Game.Game, player: Player.Player, card: Card.Card
    ) -> dict[OptionKeys, Any]:
        """+1 Buy and +$3."""
        if card.isAction() and self._pending:
            while self._pending:
                self._pending -= 1
                player.buys.add(1)
                player.coins.add(3)
            self.permanent = False
            self.stay_in_play = False
            player.output(f"Abundance triggered due to gaining {card}")
            if self in player.in_play_cards():
                player.move_card(self, Piles.DISCARD)
        return {}


###############################################################################
class TestAbundance(unittest.TestCase):
    """Test Abundance"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Abundance", "Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Abundance")

    def test_gain_action(self) -> None:
        """Play an Abundance"""
        self.plr.piles[Piles.HAND].empty()
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        buys = self.plr.buys.get()
        coins = self.plr.coins.get()
        self.plr.gain_card("Moat")
        self.assertEqual(self.plr.buys.get(), buys + 1)
        self.assertEqual(self.plr.coins.get(), coins + 3)
        self.assertIn("Abundance", self.plr.piles[Piles.DISCARD])

    def test_same_turn_gain_after_leaving_play(self):
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        self.plr.move_card(self.card, Piles.HAND)
        coins = self.plr.coins.get()
        self.plr.gain_card("Moat")
        self.assertEqual(self.plr.coins.get(), coins + 3)
        self.assertIn(self.card, self.plr.piles[Piles.HAND])
        self.plr.gain_card("Moat")
        self.assertEqual(self.plr.coins.get(), coins + 3)

    def test_multiple_plays_trigger_on_one_gain(self):
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        self.plr.play_card(self.card, cost_action=False, move_card=False)
        self.plr.gain_card("Moat")
        self.assertEqual(self.plr.coins.get(), 6)
        self.assertEqual(self.card._pending, 0)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
