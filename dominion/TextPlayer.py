"""Dominion player using a text interface (backed by TextController)."""

from typing import Any, TYPE_CHECKING

from dominion.Controller import TextController
from dominion.Option import Option
from dominion.Player import Player

if TYPE_CHECKING:
    from dominion.Game import Game


###############################################################################
class TextPlayer(Player):
    """Player wired to a TextController (terminal / test_input)."""

    def __init__(self, game: "Game", name: str = "", quiet: bool = False, **kwargs: Any) -> None:
        player_num = kwargs.pop("number", 0)
        controller = TextController(player_number=player_num)
        self.colour = controller.colour
        self.console = controller.console
        super().__init__(game, name, quiet=quiet, controller=controller, **kwargs)

    def selector_line(self, o: Option) -> str:
        """Delegate formatting to TextController."""
        assert isinstance(self.controller, TextController)
        return self.controller.selector_line(self, o)

    def get_user_input(self) -> str:
        """Delegate raw input retrieval to TextController."""
        assert isinstance(self.controller, TextController)
        return self.controller.get_raw_input(self)

    def select_source(self, **kwargs: Any) -> Any:
        return self.controller.select_source(self, **kwargs)

    def can_finish(self, any_num: bool, force: bool, num: int, num_selected: int) -> bool:
        return self.controller.can_finish(any_num, force, num, num_selected)

    def card_pile_sel_options(self, selected: list[Any], piles: list[Any], flags: dict[str, Any]) -> list[Option]:
        assert isinstance(self.controller, TextController)
        return self.controller.card_pile_sel_options(self, selected, piles, flags)


# EOF
