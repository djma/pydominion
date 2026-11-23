#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Archive"""
import unittest

from dominion import Card, Game, PlayArea, Piles, NoCardException, Player, OptionKeys

ARCHIVE = "archive"


###############################################################################
class Card_Archive(Card.Card):
    """Archive"""

    def __init__(self):
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.DURATION]
        self.base = Card.CardExpansion.EMPIRES
        self.desc = """+1 Action; Set aside the top 3 cards of your deck face
            down (you may look at them). Now and at the start of your next two turns,
            put one into your hand."""
        self.name = "Archive"
        self.actions = 1
        self.cost = 5

    def special(self, game: Game.Game, player: Player.Player):
        area = player.get_set_aside_area(ARCHIVE)
        player.specials[ARCHIVE] = area
        for _ in range(3):
            try:
                card = player.next_card()
            except NoCardException:
                continue
            player.output(f"Putting {card} in the archive")
            area.add(card)
            card.location = Piles.SPECIAL
        self.permanent = not area.is_empty()
        self.stay_in_play = not area.is_empty()

    def duration(self, game: "Game.Game", player: "Player.Player") -> dict[OptionKeys, str]:
        area = player.get_set_aside_area(ARCHIVE)
        options = [(f"Bring back {_}", _) for _ in area]
        if not options:
            self.permanent = False
            return {}
        if card := player.plr_choose_options("What card to bring back from the Archive?", *options):
            area.remove(card)
            player.add_card(card, Piles.HAND)
        if area.is_empty():
            self.permanent = False
        return {}


###############################################################################
class TestArchive(unittest.TestCase):
    """Test Archive"""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, initcards=["Archive"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Archive")

    def test_play(self):
        """Play an Archive"""
        self.plr.piles[Piles.DECK].set("Gold", "Silver", "Province")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.actions.get(), 1)
        self.plr.end_turn()
        self.plr.test_input = ["Bring back Gold"]
        self.plr.start_turn()
        self.assertIn("Gold", self.plr.piles[Piles.HAND])
        self.assertEqual(len(self.plr.specials[ARCHIVE]), 2)
        self.plr.end_turn()
        self.plr.test_input = ["Bring back Silver"]
        self.plr.start_turn()
        self.assertIn("Silver", self.plr.piles[Piles.HAND])
        self.plr.end_turn()
        self.plr.test_input = ["Bring back Province"]
        self.plr.start_turn()
        self.assertFalse(self.card.permanent)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
