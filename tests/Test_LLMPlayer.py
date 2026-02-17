#!/usr/bin/env python
"""Tests for LLMPlayer-specific behavior."""

import unittest

from dominion import Game, Piles, Phase, Prompt
from dominion.LLMPlayer import LLMPlayer


###############################################################################
class TestLLMPlayerSpendAllTreasures(unittest.TestCase):
    """Test auto-spending treasures behavior in buy phase."""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, ollama_models=["dummy-model"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def _buy_phase_options(self):
        self.plr.phase = Phase.BUY
        self.plr.piles[Piles.HAND].set("Copper", "Copper", "Copper", "Estate", "Estate")
        options = Prompt.choice_selection(self.plr)
        prompt = Prompt.generate_prompt(self.plr)
        return options, prompt

    def test_auto_spend_all_treasures_enabled(self):
        """When enabled, player auto-picks Spend all treasures in buy phase."""
        self.plr.llm_auto_spend_all_treasures = True
        options, prompt = self._buy_phase_options()

        called = False

        def fake_query_selector(*_args, **_kwargs):
            nonlocal called
            called = True
            return "0"

        self.plr._query_selector = fake_query_selector
        choice = self.plr.user_input(options, prompt)

        self.assertEqual(choice["verb"], "Spend all treasures")
        self.assertFalse(called)

    def test_auto_spend_all_treasures_disabled_uses_query_selector(self):
        """When disabled, selection still goes through _query_selector."""
        self.plr.llm_auto_spend_all_treasures = False
        options, prompt = self._buy_phase_options()
        spend_all = [opt for opt in options if opt["verb"] == "Spend all treasures"]
        self.assertTrue(spend_all)
        spend_all_selector = str(spend_all[0]["selector"])

        called = False

        def fake_query_selector(*_args, **_kwargs):
            nonlocal called
            called = True
            return spend_all_selector

        self.plr._query_selector = fake_query_selector
        choice = self.plr.user_input(options, prompt)

        self.assertTrue(called)
        self.assertEqual(choice["verb"], "Spend all treasures")

    def test_coerce_bool(self):
        """String and numeric values are coerced to sensible bool values."""
        self.assertTrue(LLMPlayer._coerce_bool(True))
        self.assertTrue(LLMPlayer._coerce_bool("true"))
        self.assertTrue(LLMPlayer._coerce_bool("1"))
        self.assertFalse(LLMPlayer._coerce_bool(False))
        self.assertFalse(LLMPlayer._coerce_bool("false"))
        self.assertFalse(LLMPlayer._coerce_bool("0"))


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
