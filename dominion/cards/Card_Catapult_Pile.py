#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Catapult
https://wiki.dominionstrategy.com/index.php/Rocks"""
import unittest

from dominion import Card, Game, CardPile, game_setup, Keys


###############################################################################
class Card_CatapultSplit(Card.Card):
    """Catapult Split"""

    def __init__(self):
        Card.Card.__init__(self)
        self.name = "Catapult"
        self.base = Card.CardExpansion.EMPIRES
        self.cardtype = [Card.CardType.ACTION, Card.CardType.ATTACK]
        self.cost = 3

    @classmethod
    def cardpile_setup(cls, game):
        card_pile = CatapultCardPile(game)
        return card_pile


###############################################################################
class CatapultCardPile(CardPile.CardPile):
    """Catapult/Rocks Pile"""

    def __init__(self, game):
        # pylint: disable=import-outside-toplevel
        from dominion.cards.Split_Catapult import Card_Catapult
        from dominion.cards.Split_Rocks import Card_Rocks

        for klass in (Card_Catapult, Card_Rocks):
            inst = klass()
            inst.game = game
            game.card_instances[inst.name] = inst
        super().__init__(game=game)

    def init_cards(self, num_cards=0, card_class=None):
        # pylint: disable=import-outside-toplevel
        from dominion.cards.Split_Catapult import Card_Catapult
        from dominion.cards.Split_Rocks import Card_Rocks

        for card_klass in (Card_Catapult, Card_Rocks):
            for _ in range(5):
                self.cards.insert(0, card_klass())


###############################################################################
class TestCatapultPile(unittest.TestCase):
    """Test Catapult Pile"""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, initcards=["Catapult"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def test_pile(self):
        """Test pile"""
        self.assertEqual(len(self.g.card_piles["Catapult"]), 10)
        card = self.g.get_card_from_pile("Catapult")
        self.assertEqual(card.name, "Catapult")
        self.assertEqual(len(self.g.card_piles["Catapult"]), 9)
        for _ in range(5):
            card = self.g.get_card_from_pile("Catapult")
        self.assertEqual(card.name, "Rocks")


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
