"""Player using the RandoBotController."""

import sys
from typing import Any, TYPE_CHECKING

from dominion import Piles
from dominion.Controller import RandoBotController
from dominion.Player import Player

if TYPE_CHECKING:
    from dominion.Card import Card
    from dominion.Game import Game


###############################################################################
class RandobotPlayer(Player):
    """Random-choice bot player backed by RandoBotController."""

    def __init__(self, game: "Game", name: str = "", quiet: bool = False, **kwargs: Any) -> None:
        kwargs.pop("number", None)
        controller = RandoBotController()
        super().__init__(game, name, quiet=quiet, controller=controller, **kwargs)

    def card_selSource(self, **kwargs: Any) -> Any:
        return self.controller.select_source(self, **kwargs)

    def selectorLine(self, o: Any) -> str:
        return RandoBotController.selector_line(o)

    def pick_to_discard(self, numtodiscard: int, keepvic: bool = False) -> list["Card"]:
        """Return `numtodiscard` cards from hand."""
        if numtodiscard <= 0:
            return []
        to_discard: list[Card] = []

        for card in self.piles[Piles.HAND]:
            if card.isTreasure():
                continue
            if keepvic and card.isVictory():
                continue
            to_discard.append(card)
        if len(to_discard) >= numtodiscard:
            return to_discard[:numtodiscard]

        for treas in ("Copper", "Silver", "Gold", "Platinum"):
            for card in self.piles[Piles.HAND]:
                if card.name == treas and card not in to_discard:
                    to_discard.append(card)
                    if len(to_discard) >= numtodiscard:
                        return to_discard[:numtodiscard]

        for card in self.piles[Piles.HAND]:
            if card not in to_discard:
                to_discard.append(card)
                if len(to_discard) >= numtodiscard:
                    return to_discard[:numtodiscard]

        hand = ", ".join([_.name for _ in self.piles[Piles.HAND]])
        sys.stderr.write(f"Couldn't find cards to discard {numtodiscard} from {hand}")
        return to_discard


# EOF
