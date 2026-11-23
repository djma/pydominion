#!/usr/bin/env python
""" https://wiki.dominionstrategy.com/index.php/Rush"""
import unittest
from typing import Any

from dominion import Card, Game, Piles, Event, Player, OptionKeys
from dominion.TurnEffects import CardTrack


###############################################################################
class Event_Rush(Event.Event):
    """Rush"""

    def __init__(self) -> None:
        Event.Event.__init__(self)
        self.base = Card.CardExpansion.PLUNDER
        self.desc = (
            """+1 Buy; The next time you gain an Action card this turn, play it."""
        )
        self.name = "Rush"
        self.cost = 2
        self.buys = 1

    def special(self, game: Game.Game, player: Player.Player) -> None:
        effects = player.turn_effects.extra_effects
        effects["rush_pending"] = effects.get("rush_pending", 0) + 1

    def hook_any_gain_card(
        self, game: Game.Game, player: Player.Player, card: Card.Card
    ) -> dict[OptionKeys, Any]:
        effects = player.turn_effects.extra_effects
        pending = effects.get("rush_pending", 0)
        if pending and card.isAction():
            # Consume before other gain reactions, even if they move the card.
            effects["rush_pending"] = 0
            effects.setdefault("rush_gains", {})[card.uuid] = (CardTrack.capture(card), pending)
        return {}

    def hook_gain_card(
        self, game: Game.Game, player: Player.Player, card: Card.Card
    ) -> dict[OptionKeys, Any]:
        gains = player.turn_effects.extra_effects.get("rush_gains", {})
        effect = gains.pop(card.uuid, None)
        if effect is not None:
            track, count = effect
            for _ in range(count):
                if not track.is_valid():
                    break
                player.play_card(card, discard=False, cost_action=False)
                if card.location != Piles.PLAYED:
                    break
                track = CardTrack.capture(card, Piles.PLAYED)
        return {}


###############################################################################
class TestRush(unittest.TestCase):
    """Test Rush"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, events=["Rush"], initcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.event = self.g.events["Rush"]

    def test_rush(self) -> None:
        """Use Rush"""
        self.plr.coins.set(2)
        hand_size = len(self.plr.piles[Piles.HAND])
        buys = self.plr.buys.get()
        self.plr.perform_event(self.event)
        self.assertEqual(
            self.plr.buys.get(), buys
        )  # +1 for Rush, -1 for performing event
        self.assertEqual(self.plr.coins.get(), 0)
        self.plr.gain_card("Moat")
        self.assertEqual(len(self.plr.piles[Piles.HAND]), hand_size + 2)

    def test_only_first_action_is_played(self) -> None:
        self.plr.coins.set(2)
        self.plr.perform_event(self.event)
        first = self.plr.gain_card("Moat")
        second = self.plr.gain_card("Moat")
        self.assertEqual(first.location, Piles.PLAYED)
        self.assertEqual(second.location, Piles.DISCARD)

    def test_lost_card_consumes_rush(self) -> None:
        self.plr.coins.set(2)
        self.plr.perform_event(self.event)
        card = self.plr.gain_card("Moat", callhook=False)
        self.event.hook_any_gain_card(self.g, self.plr, card)
        self.plr.trash_card(card)
        self.event.hook_gain_card(self.g, self.plr, card)
        self.assertEqual(card.location, Piles.TRASH)
        self.assertEqual(self.plr.gain_card("Moat").location, Piles.DISCARD)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
