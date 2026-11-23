#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Encampment
https://wiki.dominionstrategy.com/index.php/Plunder"""
import unittest

from dominion import Card, Game, CardPile, game_setup, Keys


###############################################################################
class Card_EncampmentSplit(Card.Card):
    """Encampment / Plunder Card"""

    def __init__(self):
        Card.Card.__init__(self)
        self.name = "Encampment"
        self.base = Card.CardExpansion.EMPIRES
        self.cardtype = [Card.CardType.ACTION]
        self.cost = 2

    @classmethod
    def cardpile_setup(cls, game):
        card_pile = EncampmentCardPile(game)
        return card_pile


###############################################################################
class EncampmentCardPile(CardPile.CardPile):
    """Encampment / Plunder Pile"""

    def __init__(self, game):
        # pylint: disable=import-outside-toplevel
        from dominion.cards.Split_Encampment import Card_Encampment
        from dominion.cards.Split_Plunder import Card_Plunder

        for klass in (Card_Encampment, Card_Plunder):
            inst = klass()
            inst.game = game
            game.card_instances[inst.name] = inst
        super().__init__(game=game)

    def init_cards(self, num_cards=0, card_class=None):
        # pylint: disable=import-outside-toplevel
        from dominion.cards.Split_Encampment import Card_Encampment
        from dominion.cards.Split_Plunder import Card_Plunder

        for card_klass in (Card_Encampment, Card_Plunder):
            for _ in range(5):
                self.cards.insert(0, card_klass())


###############################################################################
class TestEncampmentPile(unittest.TestCase):
    """Test Encampment Pile"""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, initcards=["Encampment"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def test_pile(self):
        """Test Pile"""
        self.assertEqual(len(self.g.card_piles["Encampment"]), 10)
        card = self.g.get_card_from_pile("Encampment")
        self.assertEqual(card.name, "Encampment")
        self.assertEqual(len(self.g.card_piles["Encampment"]), 9)
        for _ in range(5):
            card = self.g.get_card_from_pile("Encampment")
        self.assertEqual(card.name, "Plunder")


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
