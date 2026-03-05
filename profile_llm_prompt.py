#!/usr/bin/env python3
"""
Profile what LLMPlayer._build_llm_turn_prompt() looks like in practice.

Two players play each other (one LLMPlayer, one RandobotPlayer) for N turns
each, then displays the formatted prompt that would be fed to the LLM.

Usage:
    uv run profile_llm_prompt.py                        # 3 turns each, show first prompt
    uv run profile_llm_prompt.py --turns 5              # 5 turns each
    uv run profile_llm_prompt.py --index -1             # show last captured prompt
    uv run profile_llm_prompt.py --all                  # dump every captured prompt
    uv run profile_llm_prompt.py --phase action         # first action-phase prompt
    uv run profile_llm_prompt.py --phase buy            # first buy-phase prompt (before spending)
    uv run profile_llm_prompt.py --phase buy-spent      # first buy-phase prompt after treasures spent
"""

import argparse
import contextlib
import io
import random
import sys
import types
from dataclasses import dataclass

from dominion import Action, Game, Phase, Prompt
from dominion.LLMPlayer import LLMPlayer
from dominion.Option import Option

PHASE_CHOICES = ("action", "buy", "buy-spent")


###############################################################################
@dataclass
class CapturedPrompt:
    text: str
    phase: Phase
    treasures_spent: bool  # True if coins > 0 or no spend/spendall options remain


###############################################################################
def _classify(player: LLMPlayer, options: list[Option]) -> tuple[Phase, bool]:
    """Return the current phase and whether treasures have already been spent."""
    phase = player.phase
    has_spend = any(opt.get("action") in (Action.SPEND, Action.SPENDALL) for opt in options)
    coins = getattr(player, "coins", None)
    coin_value = coins.get() if hasattr(coins, "get") else int(coins or 0)
    # "spent" means we're in buy phase and there are no more treasure-spend options
    # (either spendall was already used, or coins > 0 with no spend options left)
    treasures_spent = phase == Phase.BUY and (not has_spend or coin_value > 0)
    return phase, treasures_spent


###############################################################################
def _make_spy(game: "Game.Game") -> LLMPlayer:
    """Return the LLMPlayer created by the game, patched to never hit the network."""
    players = game.player_list()
    spy = next(p for p in players if isinstance(p, LLMPlayer))
    spy.captured_prompts: list[CapturedPrompt] = []  # type: ignore[attr-defined]

    def _spy_user_input(self: LLMPlayer, options: list[Option], prompt: str) -> Option:
        legal_selectors = [str(opt["selector"]) for opt in options]
        text = self._build_llm_turn_prompt(prompt, options, legal_selectors)
        phase, spent = _classify(self, options)
        self.captured_prompts.append(CapturedPrompt(text=text, phase=phase, treasures_spent=spent))  # type: ignore[attr-defined]

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
    return spy


###############################################################################
def _filter_by_phase(captured: list[CapturedPrompt], phase_arg: str) -> list[CapturedPrompt]:
    if phase_arg == "action":
        return [c for c in captured if c.phase == Phase.ACTION]
    if phase_arg == "buy":
        return [c for c in captured if c.phase == Phase.BUY and not c.treasures_spent]
    if phase_arg == "buy-spent":
        return [c for c in captured if c.phase == Phase.BUY and c.treasures_spent]
    return captured


###############################################################################
def _print_prompt(text: str, label: str, sep: str) -> None:
    print(sep)
    print(f"  {label}")
    print(sep)
    print(text)
    print(sep)


###############################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=3, help="Turns per player to simulate (default: 3)")
    parser.add_argument("--index", type=int, default=0, help="Which prompt to display: 0=first, -1=last (default: 0)")
    parser.add_argument("--all", action="store_true", help="Dump every captured prompt (respects --phase filter)")
    parser.add_argument(
        "--phase",
        choices=PHASE_CHOICES,
        default=None,
        help="Only show prompts from a specific phase: action, buy (before spending), buy-spent (after spending)",
    )
    args = parser.parse_args()

    total_game_turns = args.turns * 2  # g.turn() advances one player at a time

    print(f"Setting up game: 1 LLMPlayer + 1 RandobotPlayer, {args.turns} turns each …\n")

    # ollama_models triggers LLMPlayer creation; randobot=1 creates one RandobotPlayer
    # Suppress stdout during game setup and play (RandobotPlayer always prints)
    with contextlib.redirect_stdout(io.StringIO()):
        g = Game.TestGame(
            numplayers=2,
            ollama_models=["dummy-model"],  # first player becomes LLMPlayer
            randobot=1,                     # second player becomes RandobotPlayer
            quiet=True,
        )
        g.start_game()

        spy = _make_spy(g)

        for _ in range(1, total_game_turns + 1):
            try:
                g.turn()
            except SystemExit:
                break
            if g.isGameOver():
                break

    all_captured: list[CapturedPrompt] = spy.captured_prompts  # type: ignore[attr-defined]
    sep = "=" * 80

    if not all_captured:
        print("No prompts were captured – the LLMPlayer may not have had a turn yet.")
        sys.exit(1)

    # Apply phase filter
    captured = _filter_by_phase(all_captured, args.phase) if args.phase else all_captured
    phase_label = f" [{args.phase}]" if args.phase else ""

    if not captured:
        counts = {k: len(_filter_by_phase(all_captured, k)) for k in PHASE_CHOICES}
        print(f"No prompts matched phase '{args.phase}'. Available: {counts}")
        sys.exit(1)

    if args.all:
        for i, entry in enumerate(captured):
            _print_prompt(entry.text, f"Prompt {i + 1} / {len(captured)}{phase_label}", sep)
    else:
        try:
            entry = captured[args.index]
        except IndexError:
            print(f"Index {args.index} out of range – only {len(captured)} prompts matched{phase_label}.")
            sys.exit(1)
        label = f"Prompt #{args.index} of {len(captured)}{phase_label} captured for {spy.name}"
        _print_prompt(entry.text, label, sep)

    total_by_phase = {k: len(_filter_by_phase(all_captured, k)) for k in PHASE_CHOICES}
    print(f"\nTotal prompts captured for {spy.name}: {len(all_captured)}  {total_by_phase}")


if __name__ == "__main__":
    main()
