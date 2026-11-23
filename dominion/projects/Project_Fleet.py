#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Fleet"""
import unittest

from dominion import Card, Game, Project, Player, Piles


###############################################################################
class Project_Fleet(Project.Project):
    def __init__(self):
        Project.Project.__init__(self)
        self.base = Card.CardExpansion.RENAISSANCE
        self.desc = "After the game ends, there's an extra round of turns just for players with this."
        self.name = "Fleet"
        self.cost = 5

    # Fleet extra-round orchestration is handled by Game.turn() via TurnQueue.start_fleet_round()
    # immediately after game-end conditions are met and BEFORE end-of-game scoring.


###############################################################################
class Test_Fleet(unittest.TestCase):
    def setUp(self):
        self.g = Game.TestGame(numplayers=1, projects=["Fleet"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def test_play(self):
        """Fleet grants an extra round of turns after the game ends."""
        self.plr.assign_project("Fleet")
        # Empty Provinces so game-end triggers after the regular turn
        while not self.g.card_piles["Province"].is_empty():
            card = self.g.get_card_from_pile("Province")
            self.g.trash_pile.add(card)
            card.location = Piles.TRASH
        self.plr.test_input = ["End Phase", "End Phase"]
        self.g.turn()
        self.assertFalse(self.g.game_over)
        self.assertTrue(self.g.turn_queue.has_pending_fleet_turns())
        # Execute the Fleet turn
        self.plr.test_input = ["End Phase", "Buy Copper", "End Phase"]
        before_coppers = self.plr.get_cards().get("Copper", 0)
        self.g.turn()
        self.assertTrue(self.g.game_over)
        self.assertEqual(self.plr.get_cards().get("Copper", 0), before_coppers + 1)
        self.assertTrue(self.plr.piles[Piles.PLAYED].is_empty())
        self.assertEqual(self.plr.turn_number, 1)  # Fleet turn does not count as a regular turn


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
