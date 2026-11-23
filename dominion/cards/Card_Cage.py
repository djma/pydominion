#!/usr/bin/env python
"""http://wiki.dominionstrategy.com/index.php/Cage"""

import unittest
from typing import Any

from dominion import Game, Card, Piles, Player, PlayArea, OptionKeys

CAGE = "cage"


###############################################################################
class Card_Cage(Card.Card):
    """Cage"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.TREASURE, Card.CardType.DURATION]
        self.base = Card.CardExpansion.PLUNDER
        self.desc = """Set aside up to 4 cards from your hand face down.
            The next time you gain a Victory card, trash this, and put the set aside cards into
            your hand at end of turn."""
        self.name = "Cage"
        self.cost = 2
        self.permanent = True
        self._active = False

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Set aside up to 4 cards from your hand face down"""
        area = player.get_set_aside_area(CAGE)
        player.specials[CAGE] = area
        player.output("Set aside up to 4 cards on Cage")
        for _ in range(4):
            if not self._set_aside(player, area):
                break

    def _set_aside(self, player: Player.Player, area: PlayArea.PlayArea) -> bool:
        card = player.plr_pick_card()
        if not card:
            return False
        player.output(f"Adding {card} to the {self}")
        player.remove_card(card)
        area.add(card)
        card.location = Piles.SPECIAL
        return True

    def hook_gain_card(self, game: "Game.Game", player: "Player.Player", card: "Card.Card") -> dict[OptionKeys, Any]:
        """The next time you gain a Victory card, trash this"""
        if self.location not in (Piles.PLAYED, Piles.DURATION):
            return {}
        if not card.isVictory():
            return {}
        if CAGE not in player.set_aside:
            return {}
        self._active = True
        return {}

    def hook_end_turn(self, game: "Game.Game", player: "Player.Player") -> None:
        """and put the set aside cards into your hand at end of turn."""
        if not self._active:
            return
        area = player.get_set_aside_area(CAGE)
        for card in list(area):
            player.output(f"Pulling {card} out of {self}")
            area.remove(card)
            player.add_card(card, Piles.HAND)
        # We have to trash here rather than the gain card hook otherwise this card won't be considered for running hooks
        player.trash_card(self)
        self._active = False


###############################################################################
class TestCage(unittest.TestCase):
    """Test Cage"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Cage"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Cage")

    def test_play_card(self) -> None:
        """Play Card"""
        self.plr.piles[Piles.HAND].set("Copper", "Silver", "Gold", "Estate", "Duchy", "Province")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["Select Province", "Select Duchy", "Select Gold", "Finish"]
        self.plr.play_card(self.card)
        self.assertNotIn("Gold", self.plr.piles[Piles.HAND])
        self.plr.end_turn()
        self.plr.start_turn()
        self.plr.gain_card("Estate")
        self.assertNotIn("Gold", self.plr.piles[Piles.HAND])
        self.plr.end_turn()
        self.assertIn("Cage", self.g.trash_pile)
        self.assertIn("Province", self.plr.piles[Piles.HAND])
        self.assertIn("Gold", self.plr.piles[Piles.HAND])


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
