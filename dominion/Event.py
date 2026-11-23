""" http://wiki.dominionstrategy.com/index.php/Event """
from dominion import Card
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dominion import Game, Player


###############################################################################
class Event(Card.Card):
    """Class representing events - mostly just card code"""

    def can_buy(self, game: "Game.Game", player: "Player.Player") -> bool:
        """Check whether this Event can be bought."""
        return True

    def hook_allowed_to_buy(self, game: "Game.Game", player: "Player.Player") -> bool:
        return self.can_buy(game, player)


# EOF
