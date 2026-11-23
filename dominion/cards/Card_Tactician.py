#!/usr/bin/env python

import unittest

from dominion import Game, Card, Piles, Player, OptionKeys


###############################################################################
class Card_Tactician(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.DURATION]
        self.base = Card.CardExpansion.SEASIDE
        self.desc = "Discard hand; +5 cards, +1 buy and +1 action next turn"
        self.name = "Tactician"
        self.cost = 5
        self._pending_triggers = 0

    def special(self, game: Game.Game, player: Player.Player) -> None:
        discard = player.plr_choose_options(
            "Discard hand for good stuff next turn?", ("Keep", False), ("Discard", True)
        )
        if discard and player.piles[Piles.HAND].size():
            self._pending_triggers += 1
            player.discard_hand({})
        self.stay_in_play = self._pending_triggers > 0

    def duration(self, game: Game.Game, player: Player.Player) -> dict[OptionKeys, str]:
        """+5 Cards, +1 Buy, +1 Action"""
        if self._pending_triggers > 0:
            player.pickup_cards(5)
            player.buys.add(1)
            player.add_actions(1)
            self._pending_triggers -= 1
        self.stay_in_play = self._pending_triggers > 0
        return {}


###############################################################################
class TestTactician(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Tactician"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Tactician")
        self.plr.add_card(self.card, Piles.HAND)

    def test_play_discard(self) -> None:
        """Play a tactician and discard hand"""
        self.plr.test_input = ["discard"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 0)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 10)
        self.assertEqual(self.plr.actions.get(), 2)
        self.assertEqual(self.plr.buys.get(), 2)

    def test_play_keep(self) -> None:
        """Play a tactician and discard hand"""
        self.plr.test_input = ["keep"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 5)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 5)
        self.assertEqual(self.plr.actions.get(), 1)
        self.assertEqual(self.plr.buys.get(), 1)

    def test_empty_replay_keeps_pending_trigger(self):
        self.plr.test_input = ["Discard", "Discard"]
        self.plr.play_card(self.card)
        self.plr.play_card(self.card, cost_action=False, move_card=False)
        self.assertEqual(self.card._pending_triggers, 1)
        self.assertTrue(self.card.stay_in_play)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 10)
        self.assertEqual(self.plr.buys.get(), 2)
        self.assertEqual(self.card._pending_triggers, 0)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
