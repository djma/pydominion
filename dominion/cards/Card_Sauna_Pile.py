#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Sauna
https://wiki.dominionstrategy.com/index.php/Avanto"""
import unittest
from typing import Optional

from dominion import Card, Game, CardPile, Keys, game_setup


###############################################################################
class Card_SaunaSplit(Card.Card):
    """Split Pile"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.name = "Sauna"
        self.base = Card.CardExpansion.PROMO
        self.cardtype = [Card.CardType.ACTION]
        self.cost = 4

    @classmethod
    def cardpile_setup(cls, game: Game.Game) -> CardPile.CardPile:
        return SaunaCardPile(game)


###############################################################################
class SaunaCardPile(CardPile.CardPile):
    """Sauna / Avanto Pile"""

    def __init__(self, game: Game.Game) -> None:
        # pylint: disable=import-outside-toplevel
        from dominion.cards.Split_Sauna import Card_Sauna
        from dominion.cards.Split_Avanto import Card_Avanto

        for klass in (Card_Sauna, Card_Avanto):
            inst = klass()
            inst.game = game
            game.card_instances[inst.name] = inst
        super().__init__(game=game)

    def init_cards(self, num_cards: int = 0, card_class: Optional[Card.Card] = None) -> None:
        # pylint: disable=import-outside-toplevel
        from dominion.cards.Split_Sauna import Card_Sauna
        from dominion.cards.Split_Avanto import Card_Avanto

        for card_klass in (Card_Sauna, Card_Avanto):
            for _ in range(5):
                self.cards.insert(0, card_klass())


###############################################################################
class TestSaunaPile(unittest.TestCase):
    """Test Sauna Pile"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Sauna"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def test_pile(self) -> None:
        """Test Pile"""
        self.assertEqual(len(self.g.card_piles["Sauna"]), 10)
        card = self.g.get_card_from_pile("Sauna")
        self.assertEqual(card.name, "Sauna")
        self.assertEqual(len(self.g.card_piles["Sauna"]), 9)
        for _ in range(5):
            card = self.g.get_card_from_pile("Sauna")
        self.assertEqual(card.name, "Avanto")


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
