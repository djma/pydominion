"""Player using the BigMoneyBotController."""

import sys
from types import ModuleType
from typing import Any, Optional, TYPE_CHECKING

from dominion import Piles
from dominion.Controller import BigMoneyBotController
from dominion.Player import Player

if TYPE_CHECKING:
    from dominion.Card import Card
    from dominion.Game import Game


###############################################################################
class BotPlayer(Player):
    """Big-money bot player backed by BigMoneyBotController."""

    def __init__(self, game: "Game", name: str = "", quiet: bool = False, **kwargs: Any) -> None:
        kwargs.pop("number", None)
        controller = BigMoneyBotController()
        self.colour = controller.colour
        self.console = controller.console
        super().__init__(game, name, quiet=quiet, controller=controller, **kwargs)

    @classmethod
    def get_options(cls, options: list[Any]) -> dict[str, Any]:
        return BigMoneyBotController.get_options(options)

    @classmethod
    def get_calling_card(cls) -> Optional[ModuleType]:
        return BigMoneyBotController.get_calling_card()

    def pick_to_discard(self, num_to_discard: int, keepvic: bool = False) -> list["Card"]:
        """Return `num_to_discard` cards from hand (non-treasures first, then cheapest treasures)."""
        if num_to_discard <= 0:
            return []
        to_discard: list[Card] = []

        for card in self.piles[Piles.HAND]:
            if card.isTreasure():
                continue
            if keepvic and card.isVictory():
                continue
            to_discard.append(card)
        if len(to_discard) >= num_to_discard:
            return to_discard[:num_to_discard]

        for treas in ("Copper", "Silver", "Gold", "Platinum"):
            for card in self.piles[Piles.HAND]:
                if card.name == treas and card not in to_discard:
                    to_discard.append(card)
                    if len(to_discard) >= num_to_discard:
                        return to_discard[:num_to_discard]

        # Fall back to any remaining cards in hand
        for card in self.piles[Piles.HAND]:
            if card not in to_discard:
                to_discard.append(card)
                if len(to_discard) >= num_to_discard:
                    return to_discard[:num_to_discard]

        hand_contents = ", ".join([_.name for _ in self.piles[Piles.HAND]])
        sys.stderr.write(f"Couldn't find cards to discard {num_to_discard} from {hand_contents}")
        return to_discard


# EOF
