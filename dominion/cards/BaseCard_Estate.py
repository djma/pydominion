from dominion import Card


class Card_Estate(Card.Card):
    def __init__(self):
        Card.Card.__init__(self)
        self.cardtype = Card.CardType.VICTORY
        self.base = Card.CardExpansion.DOMINION
        self.desc = "1 VP"
        self.playable = False
        self.basecard = True
        self.name = "Estate"
        self.cost = 2
        self.victory = 1

    @classmethod
    def calc_numcards(cls, game):
        supply = 8 if game.numplayers <= 2 else 12
        if getattr(game, "use_shelters", False):
            return supply
        return supply + game.numplayers * 3


# EOF
