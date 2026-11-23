#!/usr/bin/env python
"""http://wiki.dominionstrategy.com/index.php/Royal_Carriage"""

import unittest

from dominion import Game, Card, Piles, Player, Whens


###############################################################################
class Card_RoyalCarriage(Card.Card):
    """Royal Carriage"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [
            Card.CardType.ACTION,
            Card.CardType.RESERVE,
        ]
        self.base = Card.CardExpansion.ADVENTURE
        self.desc = """+1 Action; Put this on your Tavern mat. After you play an Action card, if it's still in play,
        you may call this, to replay that Action."""
        self.name = "Royal Carriage"
        self.cost = 5
        self.actions = 1
        self.when = Whens.POSTACTION

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Put this on your Tavern mat."""

    def hook_call_reserve(self, game: Game.Game, player: Player.Player) -> None:
        """After you play an Action card, if it's still in play,
        you may call this, to replay that Action."""
        track = getattr(player, "_last_played_action_track", None)
        if track is not None:
            if not track.is_valid():
                return
            card = track.card
        elif player.played_this_turn:
            card = player.played_this_turn[-1]
        else:
            return
        if not card.isAction() or card.location != Piles.PLAYED or card not in player.piles[Piles.PLAYED]:
            return
        player.play_card(card, cost_action=False, discard=True, move_card=False, throne_card=self)
        return


###############################################################################
class TestRoyalCarriage(unittest.TestCase):
    """Test Royal Carriage"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Royal Carriage", "Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Royal Carriage")

    def test_play_card(self) -> None:
        """Play the card"""
        self.plr.add_card(self.card, Piles.HAND)
        actions = self.plr.actions.get()
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.actions.get(), actions + 1 - 1)
        self.assertIn("Royal Carriage", self.plr.piles[Piles.RESERVE])

    def test_replay(self) -> None:
        """Replay the card"""
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        moat = self.g.get_card_from_pile("Moat")
        self.plr.add_card(moat, Piles.HAND)
        self.plr.play_card(moat, cost_action=False, post_action_hook=False)
        hand_size = self.plr.piles[Piles.HAND].size()
        self.plr.call_reserve(self.card)
        self.g.print_state()
        self.assertEqual(self.plr.piles[Piles.HAND].size(), hand_size + 2)  # Calling Moat
        self.assertIn("Royal Carriage", self.plr.piles[Piles.PLAYED])
        self.assertIn("Moat", self.plr.piles[Piles.PLAYED])

    def test_triggering_action_left_play(self):
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.play_card(self.card)
        moat = self.g.get_card_from_pile("Moat")
        self.plr.add_card(moat, Piles.HAND)
        self.plr.play_card(moat, cost_action=False, post_action_hook=False)
        self.plr.move_card(moat, Piles.DISCARD)
        self.plr.piles[Piles.PLAYED].set("Moat")
        hand_size = len(self.plr.piles[Piles.HAND])
        self.plr.call_reserve(self.card)
        self.assertEqual(len(self.plr.piles[Piles.HAND]), hand_size)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
