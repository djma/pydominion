"""https://wiki.dominionstrategy.com/index.php/Loot"""

import random
from typing import TYPE_CHECKING, Type

from dominion import Card, CardPile, Piles

if TYPE_CHECKING:
    from dominion import Game


###############################################################################
class LootPile(CardPile.CardPile):
    """Pile of Loot (non-Supply, face-down shuffled)."""

    def __init__(self, game: "Game.Game", card_mapping: dict[str, Type[Card.Card]]) -> None:
        self.mapping = card_mapping
        for name, class_ in self.mapping.items():
            inst = class_()
            inst.game = game
            game.card_instances[name] = inst
        super().__init__(game=game, in_supply=False)

    def init_cards(self, num_cards: int = 0, card_class=None) -> None:
        """2 of each loot card"""
        for loot_card in self.mapping.values():
            for _ in range(2):
                card = loot_card()
                card.pile = "Loot"
                card.game = self.game
                card.location = Piles.CARDPILE
                self.cards.append(card)
        random.shuffle(self.cards)


###############################################################################
class Loot(Card.Card):
    """Loot"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.purchasable = False
        self.insupply = False


# EOF
