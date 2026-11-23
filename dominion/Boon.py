""" https://wiki.dominionstrategy.com/index.php/Boon"""
from dominion import Card


###############################################################################
class BoonPile:
    def __init__(self, cardname, klass):
        self.cardname = cardname
        self.boon = klass()

    ###########################################################################
    def __getattr__(self, name):
        if name == "boon" or "boon" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["boon"], name)

    ###########################################################################
    def __repr__(self):
        return self.name


###############################################################################
class Boon(Card.Card):
    pass


# EOF
