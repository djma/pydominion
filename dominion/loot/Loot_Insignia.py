#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Insignia"""
import unittest

from dominion import Loot, Card, Game, Piles, Player, OptionKeys


###############################################################################
class Loot_Insignia(Loot.Loot):
    """Insignia"""

    def __init__(self) -> None:
        Loot.Loot.__init__(self)
        self.cardtype = [Card.CardType.LOOT, Card.CardType.TREASURE]
        self.base = Card.CardExpansion.PLUNDER
        self.name = "Insignia"
        self.desc = "$3; This turn, when you gain a card, you may put it onto your deck."
        self.cost = 7
        self.pile = "Loot"
        self.coin = 3

    def hook_gain_card(
        self, game: Game.Game, player: Player.Player, card: Card.Card
    ) -> dict[OptionKeys, Piles]:
        if player.plr_choose_options(
            f"Insignia: Put {card} onto your deck?",
            ("Put onto deck", True), ("Leave it", False),
        ):
            return {OptionKeys.DESTINATION: Piles.TOPDECK}
        return {}


###############################################################################
class TestInsignia(unittest.TestCase):
    """Test Insignia."""

    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, traits=["Cursed"], initcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Loot", "Insignia")
        self.plr.add_card(self.card, Piles.HAND)

    def test_topdeck(self) -> None:
        self.plr.play_card(self.card)
        self.plr.test_input = ["Put onto deck", "Put onto deck"]
        self.plr.gain_card("Silver")
        self.plr.gain_card("Gold")
        self.assertEqual(self.plr.piles[Piles.DECK][-1].name, "Gold")
        self.assertIn("Silver", self.plr.piles[Piles.DECK])
        self.assertEqual(self.plr.coins.get(), 3)

    def test_leave_card(self) -> None:
        self.plr.play_card(self.card)
        self.plr.test_input = ["Leave it"]
        self.plr.gain_card("Gold")
        self.assertIn("Gold", self.plr.piles[Piles.DISCARD])


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
