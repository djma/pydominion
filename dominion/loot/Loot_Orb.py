#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Orb"""
import unittest

from dominion import Loot, Card, Game, Piles, Player, OptionKeys


###############################################################################
class Loot_Orb(Loot.Loot):
    """Orb"""

    def __init__(self) -> None:
        Loot.Loot.__init__(self)
        self.cardtype = [Card.CardType.LOOT, Card.CardType.TREASURE]
        self.base = Card.CardExpansion.PLUNDER
        self.name = "Orb"
        self.desc = "Look through your discard pile. Choose one: Play an Action or Treasure from it; or +1 Buy and +$3."
        self.cost = 7
        self.pile = "Loot"

    def special(self, game: Game.Game, player: Player.Player) -> None:
        choices = [("+1 Buy and +$3", None)]
        choices.extend(
            (f"Play {card}", card) for card in player.piles[Piles.DISCARD]
            if card.isAction() or card.isTreasure()
        )
        card = player.plr_choose_options("Orb: Choose one", *choices)
        if card is None:
            player.buys.add(1)
            player.coins.add(3)
        else:
            player.play_card(card, cost_action=False)


###############################################################################
class TestOrb(unittest.TestCase):
    """Test Orb."""

    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, traits=["Cursed"], initcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Loot", "Orb")
        self.plr.add_card(self.card, Piles.HAND)

    def test_coins(self) -> None:
        self.plr.test_input = ["+1 Buy"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 2)

    def test_play_action(self) -> None:
        self.plr.piles[Piles.DISCARD].set("Moat", "Estate")
        self.plr.piles[Piles.DECK].set("Copper", "Copper")
        cards = len(self.plr.piles[Piles.HAND])
        self.plr.test_input = ["Play Moat"]
        self.plr.play_card(self.card)
        self.assertIn("Moat", self.plr.piles[Piles.PLAYED])
        self.assertEqual(len(self.plr.piles[Piles.HAND]), cards - 1 + 2)
        self.assertEqual(self.plr.coins.get(), 0)

    def test_play_treasure(self) -> None:
        self.plr.piles[Piles.DISCARD].set("Gold")
        self.plr.test_input = ["Play Gold"]
        self.plr.play_card(self.card)
        self.assertIn("Gold", self.plr.piles[Piles.PLAYED])
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 1)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
