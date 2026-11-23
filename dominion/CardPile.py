"""https://wiki.dominionstrategy.com/index.php/Pile"""

from typing import Optional, TYPE_CHECKING, Self

from dominion import NoCardException
from dominion.Card import CardType

if TYPE_CHECKING:
    from dominion import Game
    from dominion.Card import Card
    from dominion.Cost import Cost


###############################################################################
class CardPile:
    """Card Pile with explicit Supply membership and Randomizer metadata."""

    def __init__(self, game: Optional["Game.Game"] = None, in_supply: bool = True) -> None:
        self.cards: list[Card] = []
        self.game = game
        self.in_supply: bool = in_supply
        self.is_kingdom: bool = False
        self.embargo_level = 0
        self.gatheredvp = 0
        self.trait: Optional[str] = None
        self.randomizer_cost: Optional["Cost"] = None
        self.randomizer_types: list[CardType] = []

    ###########################################################################
    def init_cards(self, num_cards: int = 0, card_class: Optional[type["Card"]] = None) -> None:
        """Can be overwritten for the more unusual piles"""
        if num_cards == 0 or not card_class:
            return
        for _ in range(num_cards):
            card = card_class()
            self.cards.append(card)

    ##########################################################################
    def set_debug(self) -> None:
        """Set debug for all cards in a pile"""
        for card in self.cards:
            card.debug = True

    ##########################################################################
    def addVP(self, num: int = 1) -> None:
        """Add VP to a pile"""
        self.gatheredvp += num

    ##########################################################################
    def getVP(self) -> int:
        """Get the VP on a pile"""
        return self.gatheredvp

    ##########################################################################
    def drainVP(self) -> int:
        """Remove all VP from a pile"""
        num = self.gatheredvp
        self.gatheredvp = 0
        return num

    ###########################################################################
    def __iter__(self):
        return CardPileIterator(self)

    ###########################################################################
    def __len__(self) -> int:
        return len(self.cards)

    ###########################################################################
    def __bool__(self) -> bool:
        return not self.is_empty()

    ###########################################################################
    def embargo(self) -> None:
        """Embargo a pile"""
        self.embargo_level += 1

    ###########################################################################
    def __lt__(self, other_card_pile: Self) -> bool:
        self_name = self.cards[-1].name if self.cards else "ZZZZ"
        other_name = other_card_pile.cards[-1].name if other_card_pile.cards else "ZZZZ"
        return self_name < other_name

    ###########################################################################
    def setup(self, game: "Game.Game") -> None:
        """Setup card pile (runs once on the top/representative card)."""
        if self.cards:
            self.cards[-1].setup(game)

    ###########################################################################
    def is_empty(self) -> bool:
        """Is the card pile empty"""
        return not self.cards

    ###########################################################################
    def remove(self, name: str = "", top_only: bool = False) -> Optional["Card"]:
        """Remove a card from the card pile.

        - Without `name`, pops the top card (index -1) or raises NoCardException.
        - With `name` and `top_only=True`, only removes if top card matches `name`.
        - With `name` and `top_only=False`, removes the topmost matching card (legacy/setup).
        """
        if not name:
            try:
                return self.cards.pop()
            except IndexError:
                raise NoCardException from None
        if top_only:
            if self.cards and self.cards[-1].name == name:
                return self.cards.pop()
            return None
        for idx in range(len(self.cards) - 1, -1, -1):
            if self.cards[idx].name == name:
                return self.cards.pop(idx)
        return None

    ###########################################################################
    def add(self, card: "Card") -> None:
        """Add a card to the bottom of the pile (used by rotate/init)."""
        self.cards.insert(0, card)

    ###########################################################################
    def add_to_top(self, card: "Card") -> None:
        """Add a card to the top of the pile (used when returning cards to their pile)."""
        self.cards.append(card)

    return_card = add_to_top

    ###########################################################################
    def top_card(self) -> str:
        """What is the name of the top card of the card pile"""
        if self.is_empty():
            raise NoCardException
        return self.cards[-1].name

    ###########################################################################
    def get_top_card(self) -> "Card":
        """What is the top card of the card pile"""
        if self.is_empty():
            raise NoCardException
        return self.cards[-1]

    ###########################################################################
    def rotate(self) -> None:
        """Rotate a pile of cards - only works with split decks
        http://wiki.dominionstrategy.com/index.php/Rotate
        """
        if self.is_empty():
            return
        top_card_name = self.top_card()
        count = 0
        while True:
            count += 1
            if self.top_card() != top_card_name:
                break
            if next_card := self.remove():
                self.add(next_card)
            if count > len(self.cards):  # Only one sort of card in pile
                break

    ###########################################################################
    def dump(self, name: str = "") -> None:
        """Print out all the pile - for debugging purposes only"""
        if name:
            print(f"{name}: ", end="")
        for card in self.cards:
            print(f"{card} ", end="")
        print()

    ###########################################################################
    def __repr__(self) -> str:
        trait_str = ""
        if self.trait:
            trait_str = f" Trait: {self.trait}"
        return f"<CardPile {len(self.cards)}{trait_str}>"


###############################################################################
class CardPileIterator:
    """Iterate across the Card Pile"""

    def __init__(self, card_pile: CardPile) -> None:
        self.card_pile = card_pile
        self.index = 0

    def __next__(self) -> "Card":
        try:
            result = self.card_pile.cards[self.index]
        except IndexError:
            raise StopIteration  # pylint: disable=raise-missing-from
        self.index += 1
        return result


# EOF
