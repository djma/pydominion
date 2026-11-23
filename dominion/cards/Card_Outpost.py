#!/usr/bin/env python

import unittest
from typing import Any

from dominion import Game, Card, Piles, OptionKeys, Player


###############################################################################
class Card_Outpost(Card.Card):
    def __init__(self):
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.DURATION]
        self.base = Card.CardExpansion.SEASIDE
        self.desc = """You only draw 3 cards (instead of 5) in this turn's Clean-up phase.
        Take an extra turn after this one.
        This can't cause you to take more than two consecutive turns."""
        self.name = "Outpost"
        self.cost = 5
        self._triggered = False

    def special(self, game: "Game.Game", player: "Player.Player") -> None:
        self._triggered = True
        self.stay_in_play = True

    def hook_cleanup(self, game: "Game.Game", player: "Player.Player") -> dict[OptionKeys, Any]:
        if self._triggered:
            self._triggered = False
            player.newhandsize = 3
            if "Outpost" not in player.once:
                player.once["Outpost"] = True
                if game.schedule_extra_turn(player, source="Outpost"):
                    player.output("Having a second turn due to Outpost")
        return {}


###############################################################################
class Test_Outpost(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=2, initcards=["Outpost"])
        self.g.start_game()
        self.plr, self.other = self.g.player_list()
        self.card = self.g.get_card_from_pile("Outpost")
        self.plr.add_card(self.card, Piles.HAND)

    def test_play(self) -> None:
        """Play Outpost and take a 3-card extra turn."""
        self.plr.play_card(self.card)
        self.plr.end_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 3)
        self.assertEqual(len(self.g.turn_queue.extra_queue), 1)
        self.assertEqual(self.g.turn_queue.extra_queue[0].player_uuid, self.plr.uuid)
        self.assertEqual(self.g.turn_queue.extra_queue[0].source, "Outpost")

    def test_replay_keeps_cleanup_effect(self):
        self.plr.play_card(self.card)
        self.plr.play_card(self.card, cost_action=False, move_card=False)
        self.plr.end_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 3)
        self.assertEqual(len(self.g.turn_queue.extra_queue), 1)
        self.assertIn(self.card, self.plr.piles[Piles.DURATION])

    def test_extra_turn_still_draws_three(self):
        from unittest.mock import patch

        with patch.object(self.g.turn_queue, "can_take_extra_turn", return_value=False):
            self.plr.play_card(self.card)
            self.plr.end_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 3)
        self.assertEqual(len(self.g.turn_queue.extra_queue), 0)
        self.assertIn(self.card, self.plr.piles[Piles.DURATION])


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
