#!/usr/bin/env python

import unittest

from dominion import Card, Game, Event, Player, Piles

SEIZE = "seize the day"


###############################################################################
class Event_Seize_the_Day(Event.Event):
    def __init__(self) -> None:
        Event.Event.__init__(self)
        self.base = Card.CardExpansion.ADVENTURE
        self.desc = "Once per game: Take an extra turn after this one."
        self.name = "Seize the Day"
        self.cost = 4

    def can_buy(self, game: Game.Game, player: Player.Player) -> bool:
        return SEIZE not in player.specials

    def special(self, game: Game.Game, player: Player.Player) -> None:
        if SEIZE not in player.specials:
            player.specials[SEIZE] = True
            game.schedule_extra_turn(player, source="Seize the Day", enforce_consecutive_cap=False)
        else:
            player.output("Can only play this once per game")


###############################################################################
class TestSeize_the_Day(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(
            numplayers=1,
            events=["Seize the Day"],
            initcards=["Moat", "Lurker"],
        )
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.events["Seize the Day"]

    def test_play(self) -> None:
        """Use Seize the Day"""
        self.plr.coins.set(4)
        self.plr.buys.set(5)
        self.plr.actions.set(5)
        self.plr.perform_event(self.card)
        self.plr.test_input = ["End Phase", "End Phase"]
        self.g.turn()
        self.g.print_state()
        self.assertEqual(self.plr.coins.get(), 0)
        self.assertEqual(self.plr.buys.get(), 1)
        self.assertEqual(self.plr.actions.get(), 1)
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 5)
        self.assertIn(SEIZE, self.plr.specials)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
