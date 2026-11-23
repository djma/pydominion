#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Jewels"""
import unittest

from dominion import Loot, Card, Game, Piles, Player, OptionKeys


###############################################################################
class Loot_Jewels(Loot.Loot):
    """Jewels"""

    def __init__(self) -> None:
        Loot.Loot.__init__(self)
        self.cardtype = [Card.CardType.LOOT, Card.CardType.TREASURE, Card.CardType.DURATION]
        self.base = Card.CardExpansion.PLUNDER
        self.name = "Jewels"
        self.desc = "$3; +1 Buy; At the start of your next turn, put this on the bottom of your deck."
        self.cost = 7
        self.pile = "Loot"
        self.coin = 3
        self.buys = 1

    def duration(self, game: Game.Game, player: Player.Player) -> dict[OptionKeys, Piles]:
        if self.location in (Piles.PLAYED, Piles.DURATION):
            player.move_card(self, Piles.DECK)
        return {OptionKeys.DESTINATION: Piles.DECK}


###############################################################################
class TestJewels(unittest.TestCase):
    """Test Jewels."""

    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, traits=["Cursed"], initcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Loot", "Jewels")
        self.plr.add_card(self.card, Piles.HAND)

    def test_bottomdeck(self) -> None:
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 2)
        self.plr.end_turn()
        self.plr.piles[Piles.DECK].set("Copper", "Silver", "Gold")
        self.plr.start_turn()
        self.assertIs(self.plr.piles[Piles.DECK][0], self.card)
        self.assertEqual(self.card.location, Piles.DECK)
        self.assertNotIn(self.card, self.plr.piles[Piles.PLAYED])


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
