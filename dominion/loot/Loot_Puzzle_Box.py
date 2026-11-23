#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Puzzle_Box"""
import unittest

from dominion import Loot, Card, Game, Piles, Player, OptionKeys


###############################################################################
class Loot_PuzzleBox(Loot.Loot):
    """Puzzle Box"""

    def __init__(self) -> None:
        Loot.Loot.__init__(self)
        self.cardtype = [Card.CardType.LOOT, Card.CardType.TREASURE]
        self.base = Card.CardExpansion.PLUNDER
        self.name = "Puzzle Box"
        self.desc = "$3; +1 Buy; You may set aside a card from your hand face down. Put it into your hand at end of turn."
        self.cost = 7
        self.pile = "Loot"
        self.coin = 3
        self.buys = 1

    def special(self, game: Game.Game, player: Player.Player) -> None:
        area = player.get_set_aside_area(f"puzzle box:{self.uuid}")
        if card := player.plr_pick_card(prompt="Puzzle Box: Set aside a card"):
            player.remove_card(card)
            player.add_card(card, area)

    def hook_end_turn(self, game: Game.Game, player: Player.Player) -> None:
        area = player.get_set_aside_area(f"puzzle box:{self.uuid}")
        for card in list(area):
            area.remove(card)
            player.add_card(card, Piles.HAND)


###############################################################################
class TestPuzzleBox(unittest.TestCase):
    """Test Puzzle Box."""

    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, traits=["Cursed"], initcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Loot", "Puzzle Box")
        self.plr.add_card(self.card, Piles.HAND)

    def test_return_at_end_turn(self) -> None:
        self.plr.piles[Piles.HAND].set("Gold")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["Select Gold"]
        self.plr.play_card(self.card)
        self.assertNotIn("Gold", self.plr.piles[Piles.HAND])
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 2)
        self.plr.end_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 6)
        self.assertIn("Gold", self.plr.piles[Piles.HAND])
        self.assertIn(self.card, self.plr.piles[Piles.DISCARD])

    def test_empty_hand(self) -> None:
        self.plr.piles[Piles.HAND].empty()
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        self.plr.end_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 5)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
