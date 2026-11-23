""" https://wiki.dominionstrategy.com/index.php/Hex"""
from typing import Any

from dominion import Card


###############################################################################
class HexPile:
    def __init__(self, cardname: str, klass) -> None:
        self.cardname = cardname
        self.hx = klass()

    ###########################################################################
    def __getattr__(self, name: str) -> Any:
        if name == "hx" or "hx" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["hx"], name)

    ###########################################################################
    def __repr__(self) -> str:
        return self.name


###############################################################################
class Hex(Card.Card):
    pass


# EOF
