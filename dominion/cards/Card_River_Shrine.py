#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/River_Shrine"""
import unittest
from typing import Any

from dominion import Game, Card, Piles, Player, OptionKeys


###############################################################################
class Card_River_Shrine(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.OMEN]
        self.base = Card.CardExpansion.RISING_SUN
        self.desc = """+1 Sun; Trash up to 2 cards from your hand.
            At the start of Clean-up, if you didn't gain any cards in your Buy phase this turn,
            gain a card costing up to $4."""
        self.name = "River Shrine"
        self.cost = 4
        self._played_normally = False

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Trash up to 2 cards from your hand."""
        self._played_normally = True
        player.plr_trash_card(num=2)

    def hook_cleanup(self, game: "Game.Game", player: "Player.Player") -> dict[OptionKeys, Any]:
        """At the start of Clean-up, if you didn't gain any cards in your
        Buy phase this turn, gain a card costing up to $4."""
        played_normally = self._played_normally
        self._played_normally = False
        if played_normally and not (player.stats.get("gained_in_buy", []) or player.stats["bought"]):
            player.plr_gain_card(cost=4)
        return {}


###############################################################################
class Test_River_Shrine(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["River Shrine"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("River Shrine")

    def test_play(self) -> None:
        """Play card - buy nothing"""
        self.plr.piles[Piles.HAND].set("Copper", "Gold", "Estate", "Duchy", "River Shrine")
        self.plr.test_input = [
            "Play River Shrine",
            "Trash Copper",
            "Trash Estate",
            "Finish",
            "End Phase",  # Action Phase
            "End Phase",  # Buy Phase
            "Get Silver",
        ]
        self.plr.turn()
        self.assertIn("Copper", self.g.trash_pile)
        self.assertIn("Silver", self.plr.piles[Piles.DISCARD])

    def test_play_buy(self) -> None:
        """Play card - buy something"""
        self.plr.piles[Piles.HAND].set("Copper", "Silver", "Gold", "Estate", "Duchy", "River Shrine")
        self.plr.test_input = [
            "Play River Shrine",
            "Trash Copper",
            "Trash Estate",
            "Finish",
            "End Phase",  # Action
            "Buy Copper",
            "End Phase",  # Buy
        ]
        self.plr.turn()
        self.assertIn("Copper", self.g.trash_pile)

    def test_buy_phase_gain_suppresses_cleanup_gain(self):
        from unittest.mock import patch
        from dominion import Phase

        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["Finish"]
        self.plr.play_card(self.card)
        self.plr.phase = Phase.BUY
        self.plr.gain_card("Silver")
        with patch.object(self.plr, "plr_gain_card", side_effect=AssertionError("Extra gain")):
            self.card.hook_cleanup(self.g, self.plr)
        self.assertFalse(self.card._played_normally)

    def test_unplayed_text_has_no_cleanup_gain(self):
        from unittest.mock import patch

        with patch.object(self.plr, "plr_gain_card", side_effect=AssertionError("Way cleanup gain")):
            self.card.hook_cleanup(self.g, self.plr)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
