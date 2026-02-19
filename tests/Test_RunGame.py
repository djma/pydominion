#!/usr/bin/env python
""" Test the run_game module """

import unittest

from dominion import Game, Phase, Prompt, rungame


###############################################################################
class Test_parse_args(unittest.TestCase):
    """Test argument parsing"""

    def test_defaults(self):
        """Test that no args gives us the defaults"""
        args = rungame.parse_cli_args([])
        self.assertEqual(args.numplayers, 2)
        self.assertEqual(args.prosperity, False)
        self.assertEqual(args.initcards, [])

    def test_prosperity(self):
        """Test Prosperity flag"""
        args = rungame.parse_cli_args(["--prosperity"])
        self.assertEqual(args.prosperity, True)

    def test_numplayers(self):
        """Test changing number of players"""
        args = rungame.parse_cli_args(["--numplayers", "4"])
        self.assertEqual(args.numplayers, 4)

    def test_events(self):
        """Test using events"""
        args = rungame.parse_cli_args(["--events", "Alms"])
        self.assertEqual(args.events, ["Alms"])
        g = Game.TestGame(**vars(args))
        g.start_game()
        self.assertIn("Alms", g.events)

    def test_use_card(self):
        """Test specifying a card"""
        args = rungame.parse_cli_args(["--card", "Moat"])
        g = Game.TestGame(**vars(args))
        g.start_game()
        self.assertIn("Moat", g.card_piles)

    def test_use_landmark(self):
        """Test specifying a landmark"""
        args = rungame.parse_cli_args(["--landmark", "Aqueduct"])
        g = Game.TestGame(**vars(args))
        g.start_game()
        self.assertIn("Aqueduct", g.landmarks)

    def test_ollama_args(self):
        """Test parsing Ollama player args"""
        args = rungame.parse_cli_args(["--ollama", "llama3.2"])
        self.assertEqual(args.ollama_models, ["llama3.2"])

    def test_ollama_player_setup(self):
        """Test creating an Ollama-backed LLM player"""
        args = rungame.parse_cli_args(["--numplayers", "1", "--ollama", "llama3.2"])
        g = Game.TestGame(**vars(args))
        g.start_game()
        self.assertEqual(g.player_list()[0].__class__.__name__, "LLMPlayer")

    def test_openrouter_args(self):
        """Test parsing OpenRouter player args"""
        args = rungame.parse_cli_args(["--openrouter", "openai/gpt-4o-mini"])
        self.assertEqual(args.openrouter_models, ["openai/gpt-4o-mini"])

    def test_openrouter_player_setup(self):
        """Test creating an OpenRouter-backed LLM player"""
        args = rungame.parse_cli_args(["--numplayers", "1", "--openrouter", "openai/gpt-4o-mini"])
        g = Game.TestGame(**vars(args))
        g.start_game()
        self.assertEqual(g.player_list()[0].__class__.__name__, "LLMPlayer")

    def test_ollama_user_prompt_uses_textplayer_layout(self):
        """Test Ollama user prompt includes score/card-state sections and options."""
        args = rungame.parse_cli_args(["--numplayers", "1", "--ollama", "llama3.2"])
        g = Game.TestGame(**vars(args))
        g.start_game()
        plr = g.player_list()[0]
        plr.phase = Phase.BUY
        options = Prompt.choice_selection(plr)
        prompt = Prompt.generate_prompt(plr)
        legal = [str(opt["selector"]) for opt in options if opt["selector"] not in (None, "", "-")]
        user_prompt = plr._build_llm_turn_prompt(prompt, options, legal)
        self.assertIn("############################## Turn", user_prompt)
        self.assertIn("************ Buy Phase ************", user_prompt)
        self.assertIn("Scores:", user_prompt)
        self.assertIn("Current turn resources: Actions=", user_prompt)
        self.assertIn(" Buys=", user_prompt)
        self.assertIn(f"- {plr.name}:", user_prompt)
        self.assertIn(f"{plr.name} card state:", user_prompt)
        self.assertIn("- Deck (", user_prompt)
        self.assertIn("- Hand (", user_prompt)
        self.assertIn("- Played (", user_prompt)
        self.assertIn("- Discard (", user_prompt)
        self.assertIn("0) End Phase", user_prompt)
        self.assertIn("What to do (", user_prompt)
        self.assertIn("Legal selectors:", user_prompt)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
