#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Curse"""
import unittest

from dominion import Game, Card, Piles, Player


###############################################################################
class Card_Curse(Card.Card):
    """Curse"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = Card.CardType.CURSE
        self.base = Card.CardExpansion.DOMINION
        self.desc = "-1 VP"
        self.basecard = True
        self.playable = False
        self.purchasable = True
        self.name = "Curse"
        self.cost = 0
        self.victory = -1

    @classmethod
    def calc_numcards(cls, game: Game.Game) -> int:
        if game.numplayers == 1:
            return 10
        return 10 * (game.numplayers - 1)

    def hook_coinvalue(self, game: Game.Game, player: Player.Player) -> int:
        if "Charlatan" in game.card_piles:
            return 1
        return 0

    def special(self, game: Game.Game, player: Player.Player) -> None:
        pass

    def setup(self, game: Game.Game) -> None:
        if "Charlatan" in game.card_piles:
            for c in game.card_piles["Curse"]:
                c.desc = "-1 VP; +$1"
                c.playable = True
            game.card_instances["Curse"].desc = "-1 VP; +$1"
            game.card_instances["Curse"].playable = True


###############################################################################
class TestCurse(unittest.TestCase):
    """Test Curse"""

    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, initcards=["Witch"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Curse")

    def test_play(self) -> None:
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)

    def test_have(self) -> None:
        self.plr.add_card(self.card)
        sc = self.plr.get_score_details()
        self.assertEqual(sc["Curse"], -1)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
