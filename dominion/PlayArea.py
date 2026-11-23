"""Class defining a PlayArea - such as a deck of cards, a player's hand of cards, etc"""

import random
import sys
from typing import Any, Optional, Iterator, TYPE_CHECKING

from dominion import NoCardException
from dominion.Card import Card

if TYPE_CHECKING:
    from dominion import Game


###############################################################################
class PlayArea:
    """Area of play - such as a deck of cards.

    Index conventions:
      - index 0 is the bottom of the stack (`add()` / `add_to_bottom()`)
      - index -1 is the top of the stack (`add_to_top()` / `addToTop()` / `top_card()` / `next_card()`)
    """

    def __init__(
        self,
        name: str = "",
        game: Optional["Game.Game"] = None,
        initial: Optional[list[Card]] = None,
    ) -> None:
        self.name: str = name
        self.game: Optional[Game.Game] = game
        self._cards: list[Card] = list(initial) if initial is not None else []

    ###########################################################################
    def __repr__(self) -> str:
        return f"<PlayArea {self.name}: {', '.join([str(_) for _ in self._cards])}>"

    ###########################################################################
    def __contains__(self, obj: Any) -> bool:
        """Return True if a card {obj} or a card with name {obj} is in play area"""
        for card in self._cards:
            if isinstance(obj, str):
                if card.name == obj:
                    return True
            elif card == obj:
                return True
        return False

    ###########################################################################
    def __getitem__(self, name_or_idx: str | int) -> Optional[Card]:
        """Return a card with the name {name_or_idx} or None from playarea.
        If {name_or_idx} is an integer, return that card from that index.
        """
        if isinstance(name_or_idx, int):
            return self._cards[name_or_idx]
        for card in self._cards:
            if card.name == name_or_idx:
                return card
        return None

    ###########################################################################
    def set(self, *cards: str) -> None:
        """Used for testing to set contents"""
        self.empty()
        assert self.game is not None
        for card_name in cards:
            if card_name in self.game.card_piles:
                card = self.game.card_piles[card_name].remove()
            else:
                card = self.game.card_instances[card_name]
            if card is None:
                print(f"Card Pile {card_name} is empty")
                return
            if card.pile == "":
                card.pile = card_name
            card.location = self.name
            self.addToTop(card)

    ###########################################################################
    def add(self, card: Card) -> None:
        """Add a card to the bottom of the area (index 0)."""
        self._cards.insert(0, card)

    ###########################################################################
    def add_to_bottom(self, card: Card) -> None:
        """Explicitly add a card to the bottom of the area (index 0)."""
        self._cards.insert(0, card)

    ###########################################################################
    def add_bottom_batch(self, cards: list[Card]) -> None:
        """Place a sequence of cards on the bottom of this area, preserving order (last element nearest top)."""
        self._cards = list(cards) + self._cards

    ###########################################################################
    def remove(self, card: Card) -> None:
        """Remove a card from the area"""
        try:
            self._cards.remove(card)
        except ValueError:
            sys.stderr.write(
                f"Trying to remove a card ({card}) that doesn't exist."
                f" Contents of {self.name} are: {', '.join([str(_) for _ in self._cards])}\n"
            )
            raise

    ###########################################################################
    def addToTop(self, card: Card) -> None:
        """Add a card to the top of the area (index -1)."""
        self._cards.append(card)

    add_to_top = addToTop

    ###########################################################################
    def shuffle(self) -> None:
        """Shuffle the cards"""
        for card in self._cards:
            card.bump_move_epoch()
        if self.game is not None:
            self.game.rng.shuffle(self._cards)
        else:
            random.shuffle(self._cards)

    ###########################################################################
    def size(self) -> int:
        """How many cards are in the play area"""
        return len(self)

    ###########################################################################
    def __len__(self) -> int:
        return len(self._cards)

    ###########################################################################
    def next_card(self) -> Card:
        """Take the top card of the play area"""
        return self._cards.pop()

    ###########################################################################
    def top_card(self) -> Card:
        """Return the top card without moving it"""
        try:
            return self._cards[-1]
        except IndexError as e:
            raise NoCardException from e

    ###########################################################################
    def empty(self) -> None:
        self._cards = []

    ###########################################################################
    def count(self, card: Card | str) -> int:
        cname = card.name if hasattr(card, "name") else card
        return [_.name for _ in self._cards].count(cname)

    ###########################################################################
    def is_empty(self) -> bool:
        return self._cards == []

    ###########################################################################
    def __eq__(self, a: Any) -> bool:
        if isinstance(a, PlayArea):
            return self._cards == a._cards
        if hasattr(a, "_cards"):
            return self._cards == a._cards
        return self._cards == a

    ###########################################################################
    def __add__(self, a: Any) -> "PlayArea":
        x = self._cards[:]
        if a is None:
            pass
        elif hasattr(a, "values"):
            x.extend(a.values())
        elif isinstance(a, PlayArea):
            x.extend(a._cards[:])
        elif isinstance(a, list):
            x.extend(a[:])
        elif isinstance(a, Card):
            x.append(a)
        else:
            sys.stderr.write(f"Unhandled __add__ operand: {type(a)}\n")
        return PlayArea("", self.game, x)

    ###########################################################################
    def __iter__(self) -> Iterator[Card]:
        for card in self._cards[:]:
            yield card

    ###########################################################################
    def sort(self, *args: Any, **kwargs: Any) -> None:
        self._cards.sort(*args, **kwargs)

    ###########################################################################
    def dump(self, name: Optional[str] = None) -> None:
        """Print out all the playarea - for debugging purposes only"""
        if name is None:
            name = self.name
        print(f"{name}: ", end="")
        for card in self._cards:
            print(f"{card} ", end="")
        print()


# EOF
