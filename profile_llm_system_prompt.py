#!/usr/bin/env python3
"""
Profile what the LLM system prompt looks like in practice.

Sets up a game (1 LLMPlayer + 1 RandobotPlayer), then displays the system
prompt that would be sent to the LLM.  The system prompt is static per game
(it doesn't change turn-to-turn), so only one capture is needed.

Usage:
    uv run profile_llm_system_prompt.py
"""

import contextlib
import io
import random
import types

from dominion import Action, Game
from dominion.LLMPlayer import LLMPlayer
from dominion.Option import Option


###############################################################################
def _make_spy(game: "Game.Game") -> tuple[LLMPlayer, list[str]]:
    """Return the LLMPlayer and a list that will hold the captured system prompt."""
    players = game.player_list()
    spy = next(p for p in players if isinstance(p, LLMPlayer))
    captured: list[str] = []

    def _spy_user_input(self: LLMPlayer, options: list[Option], prompt: str) -> Option:
        if not captured:
            system_prompt = (
                f"{self._system_bootstrap()}\n\n"
                "Decision protocol for this move:\n"
                "- Pick exactly one legal selector from the list.\n"
                '- Return JSON only in this format: {"selector":"x"}\n'
                "- Do not include any explanation or extra keys."
            )
            captured.append(system_prompt)

        # Random fallback: mandatory actions first, then random non-quit
        for opt in options:
            if opt.get("action") == Action.SPENDALL:
                return opt
            if opt.get("action") == Action.PAYBACK:
                return opt
        avail = [o for o in options if o.get("selector") != "-" and o.get("action") != Action.QUIT]
        if avail:
            return random.choice(avail)
        for opt in options:
            if opt.get("action") == Action.QUIT:
                return opt
        return options[0]

    spy.user_input = types.MethodType(_spy_user_input, spy)  # type: ignore[method-assign]
    return spy, captured


###############################################################################
def main() -> None:
    print("Setting up game: 1 LLMPlayer + 1 RandobotPlayer …\n")

    with contextlib.redirect_stdout(io.StringIO()):
        g = Game.TestGame(
            numplayers=2,
            ollama_models=["dummy-model"],
            randobot=1,
            quiet=True,
        )
        g.start_game()

        spy, captured = _make_spy(g)

        # Play until we capture a prompt or the game ends
        for _ in range(20):
            if captured:
                break
            try:
                g.turn()
            except SystemExit:
                break
            if g.isGameOver():
                break

    sep = "=" * 80
    if not captured:
        print("No system prompt was captured – the LLMPlayer may not have had a turn yet.")
        return

    print(sep)
    print(f"  System prompt for {spy.name}")
    print(sep)
    print(captured[0])
    print(sep)


if __name__ == "__main__":
    main()
