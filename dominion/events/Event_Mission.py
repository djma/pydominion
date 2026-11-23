#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Mission"""
import unittest

from dominion import Card, Game, Event, Limits, Player


###############################################################################
class Event_Mission(Event.Event):
    def __init__(self):
        Event.Event.__init__(self)
        self.base = Card.CardExpansion.ADVENTURE
        self.desc = """Once per turn: If the previous turn wasn't yours, take another turn after this one,
        during which you can't buy cards."""
        self.name = "Mission"
        self.cost = 4

    def can_buy(self, game: "Game.Game", player: "Player.Player") -> bool:
        return "Mission" not in player.once

    def special(self, game: "Game.Game", player: "Player.Player") -> None:
        if "Mission" in player.once:
            player.output("Mission can only be used once per turn")
            return
        player.once["Mission"] = True
        if game.last_turn(player) or not game.turn_queue.can_take_extra_turn(player.uuid):
            player.output("You had the previous turn")
            return
        game.schedule_extra_turn(player, source="Mission", modifiers={"no_buy_cards": True})


###############################################################################
class TestMission(unittest.TestCase):
    def setUp(self):
        self.g = Game.TestGame(
            numplayers=2,
            events=["Mission"],
        )
        self.g.start_game()
        self.plr, self.other = self.g.player_list()
        self.card = self.g.events["Mission"]

    def test_Mission(self):
        """Use Mission"""
        self.plr.coins.add(4)
        self.plr.perform_event(self.card)
        self.assertEqual(len(self.g.turn_queue.extra_queue), 1)
        entry = self.g.turn_queue.extra_queue[0]
        self.assertEqual(entry.player_uuid, self.plr.uuid)
        self.assertTrue(entry.modifiers.get("no_buy_cards"))
        self.plr.end_turn()
        self.plr.test_input = ["End Phase", "End Phase"]
        self.g.turn()
        self.assertEqual(self.g.current_player, self.plr)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
