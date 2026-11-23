#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Island"""
import unittest

from dominion import Card
from dominion import Game, Piles
from dominion import PlayArea, Player

ISLAND = "island"


###############################################################################
class Card_Island(Card.Card):
    """Island"""

    def __init__(self):
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.VICTORY]
        self.base = Card.CardExpansion.SEASIDE
        self.desc = """Set aside this and another card from your hand.
            Return them to your deck at the end of the game. 2VP"""
        self.name = "Island"
        self.cost = 4
        self.victory = 2

    def special(self, game: Game.Game, player: Player.Player) -> None:
        area = player.get_set_aside_area(ISLAND)
        player.specials[ISLAND] = area
        if c := player.card_sel(prompt="Select a card to set aside for the rest of the game"):
            card = c[0]
            player.remove_card(card)
            area.add(card)
            card.location = Piles.SPECIAL
        if self in player.piles[Piles.PLAYED]:  # If Island is in play
            player.remove_card(self)
            area.add(self)
            self.location = Piles.SPECIAL

    def hook_end_of_game(self, game: Game.Game, player: Player.Player) -> None:
        area = player.get_set_aside_area(ISLAND)
        for card in list(area):
            player.output(f"Returning {card} from Island")
            area.remove(card)
            player.add_card(card)

    def debug_dump(self, player: Player.Player) -> None:
        if ISLAND in player.set_aside:
            player.output(f"Island Reserve: {self}: {player.set_aside[ISLAND]}")


###############################################################################
class TestIsland(unittest.TestCase):
    """Test Island"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Island"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.plr.get_card_from_pile("Island")

    def test_play_province(self) -> None:
        """Play an island on a province"""
        self.plr.piles[Piles.HAND].set("Silver", "Province")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["province"]
        self.plr.play_card(self.card)
        self.assertNotIn("Island", self.plr.piles[Piles.PLAYED])
        self.assertNotIn("Island", self.plr.piles[Piles.HAND])
        self.assertNotIn("Island", self.plr.piles[Piles.DISCARD])
        self.assertNotIn("Province", self.plr.piles[Piles.HAND])
        self.assertNotIn("Province", self.plr.piles[Piles.DISCARD])
        self.assertEqual(len(self.plr.get_set_aside_area(ISLAND)), 2)
        score = self.plr.get_score_details()
        self.assertEqual(score["Island"], 2)
        self.assertEqual(score["Province"], 6)
        self.plr.game_over()
        self.assertIn("Island", self.plr.piles[Piles.DISCARD])
        self.assertIn("Province", self.plr.piles[Piles.DISCARD])

    def test_play_alone(self) -> None:
        """Play a island but don't pick another card"""
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["finish"]
        self.plr.play_card(self.card)
        self.assertNotIn("Island", self.plr.piles[Piles.PLAYED])
        self.assertNotIn("Island", self.plr.piles[Piles.HAND])
        self.assertNotIn("Island", self.plr.piles[Piles.DISCARD])
        self.assertEqual(len(self.plr.get_set_aside_area(ISLAND)), 1)
        self.plr.game_over()
        self.assertIn("Island", self.plr.piles[Piles.DISCARD])
        score = self.plr.get_score_details()
        self.assertEqual(score["Island"], 2)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
