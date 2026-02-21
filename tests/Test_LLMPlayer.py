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
class TestLLMPlayerSystemBootstrap(unittest.TestCase):
    """Test LLM bootstrap prompt construction."""

    def test_cards_in_play_in_turn_prompt(self):
        """Card list is included in the user (turn) prompt, not the system prompt."""
        game = Game.TestGame(numplayers=1, ollama_models=["dummy-model"], initcards=["Village"])
        game.start_game()
        player = game.player_list()[0]
        player._llm_strategy_generated = True
        player.llm_strategy = ""

        bootstrap = player._system_bootstrap()
        self.assertNotIn("Playing Dominion with these cards", bootstrap)

        turn_prompt = player._build_llm_turn_prompt([], ["0"])
        self.assertIn("Playing Dominion with these cards", turn_prompt)


###############################################################################
class TestExtractSelectors(unittest.TestCase):
    """Test _extract_selectors (multi-select parser)."""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, ollama_models=["dummy-model"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.legal = ["1", "2", "3", "4"]

    def test_json_selectors_array(self):
        result = self.plr._extract_selectors('{"selectors":["1","3"]}', self.legal)
        self.assertEqual(result, ["1", "3"])

    def test_json_selectors_empty(self):
        result = self.plr._extract_selectors('{"selectors":[]}', self.legal)
        self.assertEqual(result, [])

    def test_json_selectors_single(self):
        result = self.plr._extract_selectors('{"selectors":["2"]}', self.legal)
        self.assertEqual(result, ["2"])

    def test_json_selector_fallback(self):
        """Accept {"selector":"2"} as single-item list."""
        result = self.plr._extract_selectors('{"selector":"2"}', self.legal)
        self.assertEqual(result, ["2"])

    def test_json_bare_array(self):
        result = self.plr._extract_selectors('["1","4"]', self.legal)
        self.assertEqual(result, ["1", "4"])

    def test_filters_illegal(self):
        result = self.plr._extract_selectors('{"selectors":["1","99","3"]}', self.legal)
        self.assertEqual(result, ["1", "3"])

    def test_regex_fallback(self):
        result = self.plr._extract_selectors('selectors: ["2", "4"]', self.legal)
        self.assertEqual(result, ["2", "4"])

    def test_single_selector_fallback(self):
        result = self.plr._extract_selectors("2", self.legal)
        self.assertEqual(result, ["2"])

    def test_empty_string(self):
        result = self.plr._extract_selectors("", self.legal)
        self.assertIsNone(result)

    def test_nonsense_returns_none(self):
        result = self.plr._extract_selectors("hello world", self.legal)
        self.assertIsNone(result)

    def test_selectors_as_string(self):
        """Accept {"selectors":"2"} — single selector as plain string."""
        result = self.plr._extract_selectors('{"selectors":"2"}', self.legal)
        self.assertEqual(result, ["2"])


###############################################################################
class TestCardSelBatched(unittest.TestCase):
    """Test that LLMPlayer.card_sel makes a single LLM call."""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, ollama_models=["dummy-model"], initcards=["Chapel"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.plr.piles[Piles.HAND].set("Copper", "Copper", "Estate", "Silver")
        # Pre-generate strategy so it doesn't trigger an extra _call_llm.
        self.plr._llm_strategy_generated = True
        self.plr.llm_strategy = ""

    def _patch_call_llm(self, response: str):
        """Monkey-patch _call_llm to return a canned response."""
        calls = []

        def fake_call_llm(system_prompt, user_prompt):
            calls.append((system_prompt, user_prompt))
            self.plr.llm_last_output_full = response
            self.plr.llm_last_thinking_full = ""
            self.plr.llm_last_http_status = "200"
            self.plr.llm_last_latency_ms = 1
            return response

        self.plr._call_llm = fake_call_llm
        return calls

    def test_selects_multiple_cards(self):
        calls = self._patch_call_llm('{"selectors":["1","3"]}')
        selected = self.plr.card_sel(
            num=4, cardsrc=Piles.HAND, anynum=True, verbs=("Trash", "Untrash"),
        )
        self.assertEqual(len(calls), 1, "Should make exactly one LLM call")
        names = [c.name for c in selected]
        self.assertEqual(len(names), 2)

    def test_selects_none(self):
        calls = self._patch_call_llm('{"selectors":[]}')
        selected = self.plr.card_sel(
            num=4, cardsrc=Piles.HAND, anynum=True, verbs=("Trash", "Untrash"),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(selected, [])

    def test_single_candidate_force_skips_llm(self):
        self.plr.piles[Piles.HAND].set("Copper")
        calls = self._patch_call_llm('{"selectors":["1"]}')
        selected = self.plr.card_sel(num=1, cardsrc=Piles.HAND, force=True)
        self.assertEqual(len(calls), 0, "Should skip LLM when forced and only one candidate")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "Copper")

    def test_respects_max_count(self):
        calls = self._patch_call_llm('{"selectors":["1","2","3","4"]}')
        selected = self.plr.card_sel(num=2, cardsrc=Piles.HAND)
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(len(selected), 2)

    def test_empty_hand(self):
        self.plr.piles[Piles.HAND].set()
        calls = self._patch_call_llm('{"selectors":[]}')
        selected = self.plr.card_sel(num=1, cardsrc=Piles.HAND)
        self.assertEqual(len(calls), 0, "Should not call LLM on empty hand")
        self.assertEqual(selected, [])


###############################################################################
class TestPlrChooseOptionsBatched(unittest.TestCase):
    """Test that plr_choose_options uses a single LLM call."""

    def setUp(self):
        self.g = Game.TestGame(numplayers=1, ollama_models=["dummy-model"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def test_returns_chosen_answer(self):
        def fake_query_selector(legal_options, display_options=None):
            return "1"

        self.plr._query_selector = fake_query_selector
        result = self.plr.plr_choose_options(
            "Pick one",
            ("Option A", "answer_a"),
            ("Option B", "answer_b"),
        )
        self.assertEqual(result, "answer_b")

    def test_single_choice_skips_llm(self):
        called = False

        def fake_query_selector(*_args, **_kwargs):
            nonlocal called
            called = True
            return "0"

        self.plr._query_selector = fake_query_selector
        result = self.plr.plr_choose_options("Pick one", ("Only option", "only_answer"))
        self.assertEqual(result, "only_answer")
        self.assertFalse(called)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
