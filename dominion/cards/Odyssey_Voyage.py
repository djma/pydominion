#!/usr/bin/env python
""" http://wiki.dominionstrategy.com/index.php/Voyage"""

import unittest

from dominion import Game, Card, Piles, Limits


###############################################################################
class Card_Voyage(Card.Card):
    def __init__(self):
        Card.Card.__init__(self)
        self.cardtype = [
            Card.CardType.ACTION,
            Card.CardType.DURATION,
            Card.CardType.ODYSSEY,
        ]
        self.base = Card.CardExpansion.ALLIES
        self.cost = 4
        self.actions = 1
        self.name = "Voyage"
        self.desc = """+1 Action; If the previous turn wasn't yours, take an
            extra turn after this one, during which you can only play 3 cards
            from your hand."""
        self.pile = "Odysseys"

    def special(self, game, player):
        if not game.last_turn(player) and game.schedule_extra_turn(
            player, source="Voyage", modifiers={"play_limit": 3}
        ):
            self.stay_in_play = True
            player.output("Having a second turn due to Voyage")
        else:
            self.stay_in_play = False
            player.output("You had the previous turn")


###############################################################################
class TestVoyage(unittest.TestCase):
    def setUp(self):
        self.g = Game.TestGame(numplayers=2, initcards=["Odysseys"])
        self.g.start_game()
        self.plr, self.other = self.g.player_list()
        self.card = self.g.get_card_from_pile("Odysseys", "Voyage")

    def test_play(self):
        """Play Voyage and schedule an extra turn with play_limit=3."""
        self.plr.piles[Piles.HAND].set("Estate", "Copper")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        self.assertEqual(len(self.g.turn_queue.extra_queue), 1)
        entry = self.g.turn_queue.extra_queue[0]
        self.assertEqual(entry.player_uuid, self.plr.uuid)
        self.assertEqual(entry.modifiers.get("play_limit"), 3)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
