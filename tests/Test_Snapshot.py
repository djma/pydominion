"""Tests for Phase 5: Game State Snapshot, Rewind, and Branching Clone."""

import random
import unittest

from dominion import Game, Piles, Phase


class TestSnapshotAndRewind(unittest.TestCase):
    """Verify deterministic snapshot, in-place rewind, and independent cloning."""

    def test_mid_turn_rewind_and_rng_determinism(self) -> None:
        """Snapshotting mid-turn restores hand, deck, discard, resources, supply, and RNG."""
        random.seed(42)
        g = Game.TestGame(numplayers=2, initcards=["Smithy", "Village", "Bridge", "Moat"])
        g.start_game()
        plr, other = g.player_list()
        orig_ctrl = plr.controller
        orig_console = getattr(plr, "console", None)

        village = g.get_card_from_pile("Village")
        smithy = g.get_card_from_pile("Smithy")
        plr.add_card(village, Piles.HAND)
        plr.add_card(smithy, Piles.HAND)
        plr.coins.set(3)
        plr.actions.set(2)
        plr.buys.set(1)

        snap = g.snapshot()
        saved_hand = [c.name for c in plr.piles[Piles.HAND]]
        saved_deck = [c.name for c in plr.piles[Piles.DECK]]
        saved_silver_count = len(g.card_piles["Silver"])

        # Mutate state heavily: play Village, play Smithy, buy Silver, spend coins
        plr.play_card(village)
        plr.play_card(smithy)
        plr.buy_card("Silver")
        self.assertEqual(len(g.card_piles["Silver"]), saved_silver_count - 1)
        self.assertEqual(plr.buys.get(), 0)

        # Draw a deterministic random sample after first branch
        rng_after_branch1 = [random.random() for _ in range(5)]

        # Rewind in-place
        g.restore(snap)

        # Live object identities & UI handles preserved
        self.assertIs(g.player_list()[0], plr)
        self.assertIs(g.player_list()[1], other)
        self.assertIs(plr.controller, orig_ctrl)
        self.assertIs(getattr(plr, "console", None), orig_console)

        # Engine state fully restored
        self.assertEqual([c.name for c in plr.piles[Piles.HAND]], saved_hand)
        self.assertEqual([c.name for c in plr.piles[Piles.DECK]], saved_deck)
        self.assertEqual(len(g.card_piles["Silver"]), saved_silver_count)
        self.assertEqual(plr.coins.get(), 3)
        self.assertEqual(plr.actions.get(), 2)
        self.assertEqual(plr.buys.get(), 1)
        self.assertTrue(plr.piles[Piles.PLAYED].is_empty())

        # Circular refs rebound to live game/player
        for card in plr.all_cards():
            self.assertIs(card.game, g)
            self.assertIs(card.player, plr)

        # Repeat the exact same actions -> RNG produces identical sequence
        v2 = next(c for c in plr.piles[Piles.HAND] if c.name == "Village")
        s2 = next(c for c in plr.piles[Piles.HAND] if c.name == "Smithy")
        plr.play_card(v2)
        plr.play_card(s2)
        plr.buy_card("Silver")
        rng_after_branch2 = [random.random() for _ in range(5)]
        self.assertEqual(rng_after_branch1, rng_after_branch2)

        # Second restore from the SAME snapshot still works (snap was not mutated!)
        g.restore(snap)
        self.assertEqual([c.name for c in plr.piles[Piles.HAND]], saved_hand)
        self.assertEqual(len(g.card_piles["Silver"]), saved_silver_count)

    def test_cross_turn_rewind_with_duration_set_aside_and_coin_token(self) -> None:
        """Rewind restores Duration stays, set-aside mats, TurnQueue, and -$1 coin callback."""
        g = Game.TestGame(numplayers=2, initcards=["Haven", "Bridge Troll"])
        g.start_game()
        p0, p1 = g.player_list()

        haven = g.get_card_from_pile("Haven")
        p0.add_card(haven, Piles.HAND)
        p0.piles[Piles.HAND].add(g.get_card_from_pile("Gold"))
        p0.test_input = ["select gold"]
        p0.play_card(haven)
        p0.coin_token = True
        p0.end_turn()

        # Snapshot right after turn 1 (Haven in DURATION, Gold set aside, coin_token active)
        snap = g.snapshot()
        self.assertEqual(len(p0.piles[Piles.DURATION]), 1)
        self.assertEqual(len(p0.duration_stays), 1)

        # Advance turn: Haven triggers, Gold returns to hand, coin_token consumed
        p0.start_turn()
        self.assertEqual(len(p0.piles[Piles.DURATION]), 0)
        self.assertIn("Gold", p0.piles[Piles.HAND])
        p0.coins.add(3)
        self.assertEqual(p0.coins.get(), 2)  # -$1 token consumed
        self.assertFalse(p0.coin_token)

        # Rewind back to snapshot
        g.restore(snap)
        self.assertEqual(len(p0.piles[Piles.DURATION]), 1)
        self.assertEqual(len(p0.duration_stays), 1)
        self.assertTrue(p0.coin_token)
        # Verify coins.on_add callback is still properly bound to live p0!
        p0.coins.add(4)
        self.assertEqual(p0.coins.get(), 3)
        self.assertFalse(p0.coin_token)

    def test_clone_branching_isolation(self) -> None:
        """game.clone() creates a fully independent branch that does not mutate original."""
        g = Game.TestGame(numplayers=2, initcards=["Bridge", "Smithy", "Outpost"])
        g.start_game()
        p0, p1 = g.player_list()

        bridge = g.get_card_from_pile("Bridge")
        p0.add_card(bridge, Piles.HAND)
        orig_hand_size = len(p0.piles[Piles.HAND])
        orig_copper_pile = len(g.card_piles["Copper"])

        branch = g.clone()
        bp0, bp1 = branch.player_list()
        self.assertIsNot(branch, g)
        self.assertIsNot(bp0, p0)
        self.assertIsNot(bp0.controller, p0.controller)

        # Mutate branch heavily
        b_bridge = next(c for c in bp0.piles[Piles.HAND] if c.name == "Bridge")
        bp0.play_card(b_bridge)
        bp0.buy_card("Copper")
        branch.schedule_extra_turn(bp0, source="Outpost")
        self.assertEqual(bp0.card_cost(branch.card_piles["Gold"].get_top_card()), 5)
        self.assertEqual(len(branch.turn_queue.extra_queue), 1)

        # Original game is completely untouched
        self.assertEqual(len(p0.piles[Piles.HAND]), orig_hand_size)
        self.assertEqual(len(g.card_piles["Copper"]), orig_copper_pile)
        self.assertEqual(p0.coins.get(), 0)
        self.assertEqual(p0.card_cost(g.card_piles["Gold"].get_top_card()), 6)
        self.assertEqual(len(g.turn_queue.extra_queue), 0)
        g._validate_cards()
        branch._validate_cards()

    def test_snapshot_with_boons_hexes_and_landscapes(self) -> None:
        """Snapshot/restore/clone works cleanly with Boons, Hexes, Events, Projects, Landmarks, and Ways."""
        g = Game.TestGame(
            numplayers=2,
            initcards=["Druid", "Skulk", "Smithy"],
            events=["Mission"],
            projects=["Fleet"],
            landmarks=["Labyrinth"],
            ways=["Way of the Pig"],
        )
        g.start_game()
        p0, p1 = g.player_list()
        # Ensure deterministic non-interactive Hex on top of Hex deck
        for h in list(g.hexes):
            if h.name == "Delusion":
                g.hexes.remove(h)
                g.hexes.append(h)
                break
        orig_boons = [b.name for b in g.boons]
        orig_hexes = [h.name for h in g.hexes]

        snap = g.snapshot()
        # Draw a boon and a hex
        b = g.receive_boon()
        if b:
            g.discard_boon(b)
        g.distribute_hex(p0, [p1])
        self.assertEqual(len(g.hexes), len(orig_hexes) - 1)

        # Clone while mutated
        cloned = g.clone()
        self.assertEqual(len(cloned.hexes), len(orig_hexes) - 1)

        # Restore original to snapshot
        g.restore(snap)
        self.assertEqual([b.name for b in g.boons], orig_boons)
        self.assertEqual([h.name for h in g.hexes], orig_hexes)
        # Clone remains unaffected by restore of original
        self.assertEqual(len(cloned.hexes), len(orig_hexes) - 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
