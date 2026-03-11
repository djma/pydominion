"""Run bot-vs-bot matchups and produce statistics.

Usage (CLI)::

    uv run python -m dominion.bots.matchup --control BigMoney --experiment NaiveBigMoney --games 100
    uv run python -m dominion.bots.matchup --control BigMoney --experiment NaiveBigMoney \\
        --kingdom Chapel Village Smithy Market Militia Moat Festival Laboratory Workshop Witch
    uv run python -m dominion.bots.matchup --control BigMoney --experiment NaiveBigMoney --kingdom-seed 42
    uv run python -m dominion.bots.matchup --control BigMoney --experiment NaiveBigMoney --random

Usage (API)::

    from dominion.bots.matchup import Matchup
    from dominion.bots.BigMoney import BigMoney
    from dominion.bots.NaiveBigMoney import NaiveBigMoney

    m = Matchup(control_bot=BigMoney, experiment_bot=NaiveBigMoney, num_games=100)
    summary = m.run()
    summary.print()
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from dominion.Card import CardExpansion
from dominion.Player import Player
from dominion.bots import get_all_bots

EXPANSION_NAMES: dict[str, CardExpansion] = {e.name.lower(): e for e in CardExpansion if e != CardExpansion.TEST}


@dataclass
class GameResult:
    """Statistics for a single game."""

    seed: int
    outcome: int  # 1 = experiment won, -1 = control won, 0 = draw
    experiment_first: bool
    replay_command: str


@dataclass
class MatchupSummary:
    """Aggregate statistics across all games in a matchup."""

    results: list[GameResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if r.outcome == 1)

    @property
    def losses(self) -> int:
        return sum(1 for r in self.results if r.outcome == -1)

    @property
    def draws(self) -> int:
        return sum(1 for r in self.results if r.outcome == 0)

    @property
    def win_pct(self) -> float:
        """Win % where a draw counts as half a win."""
        if not self.results:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total * 100

    @property
    def draw_pct(self) -> float:
        if not self.results:
            return 0.0
        return self.draws / self.total * 100

    @property
    def first_player_win_pct(self) -> float:
        """Win % when the experiment bot played first (draws = 0.5)."""
        first_games = [r for r in self.results if r.experiment_first]
        if not first_games:
            return 0.0
        wins = sum(1 for r in first_games if r.outcome == 1)
        draws = sum(1 for r in first_games if r.outcome == 0)
        return (wins + 0.5 * draws) / len(first_games) * 100

    def print(self) -> None:
        print(f"Games: {self.total}  (W {self.wins} / L {self.losses} / D {self.draws})")
        print(f"Win %:            {self.win_pct:.1f}%")
        print(f"Draw %:           {self.draw_pct:.1f}%")
        print(f"Win % as 1st plr: {self.first_player_win_pct:.1f}%")


MAX_TURNS = 400


class Matchup:
    """Run a series of games between two bot classes."""

    def __init__(
        self,
        control_bot: type[Player],
        experiment_bot: type[Player],
        num_games: int = 100,
        kingdom: list[str] | None = None,
        kingdom_seed: int | None = None,
        expansions: list[CardExpansion] | None = None,
        game_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.control_bot = control_bot
        self.experiment_bot = experiment_bot
        self.num_games = num_games + (num_games % 2)  # round up to even
        self.kingdom = kingdom
        self.kingdom_seed = kingdom_seed
        self.expansions = expansions
        self.game_kwargs = dict(game_kwargs) if game_kwargs else {}

    def _build_replay_command(
        self,
        seed: int,
        experiment_first: bool,
    ) -> str:
        """Build a CLI command that reproduces a single game."""
        parts = [
            "uv run python -m dominion.bots.matchup",
            f"--control {self.control_bot.__name__}",
            f"--experiment {self.experiment_bot.__name__}",
            f"--seed {seed}",
            "--games 2",  # always 2 so it plays one per side
        ]
        if experiment_first:
            parts.append("--experiment-first")
        if self.kingdom:
            parts.append(f"--kingdom {' '.join(self.kingdom)}")
        elif self.kingdom_seed is not None:
            parts.append(f"--kingdom-seed {self.kingdom_seed}")
        else:
            parts.append("--random")
        if self.expansions:
            parts.append(f"--expansions {' '.join(e.name.lower() for e in self.expansions)}")
        for key, value in self.game_kwargs.items():
            if isinstance(value, bool) and value:
                parts.append(f"--{key}")
            elif isinstance(value, list):
                parts.append(f"--{key} {' '.join(str(v) for v in value)}")
            elif not isinstance(value, bool):
                parts.append(f"--{key} {value}")
        return " ".join(parts)

    def _resolve_kingdom(self) -> list[str] | None:
        """Resolve kingdom cards once for the entire matchup."""
        if self.kingdom is not None:
            return self.kingdom
        if self.kingdom_seed is not None:
            from dominion.bots.BigMoney import BigMoney

            saved = random.getstate()
            random.seed(self.kingdom_seed)
            from dominion.Game import Game

            kwargs: dict[str, Any] = {
                "initcards": [],
                "numplayers": 2,
                "quiet": True,
                "player_classes": [BigMoney, BigMoney],
            }
            if self.expansions:
                kwargs["allowed_expansions"] = self.expansions
            g = Game(**kwargs)
            g.start_game()
            cards = [
                name
                for name, _ in g.card_piles.items()
                if name in g.card_instances and not g.card_instances[name].basecard
            ]
            random.setstate(saved)
            self.kingdom = cards
            return cards
        return None  # fully random each game

    def _run_single_game(self, seed: int, experiment_first: bool) -> GameResult:
        """Play one game and return the result."""
        from dominion.Game import Game

        if experiment_first:
            classes = [self.experiment_bot, self.control_bot]
        else:
            classes = [self.control_bot, self.experiment_bot]

        kwargs: dict[str, Any] = {
            "numplayers": 2,
            "quiet": True,
            "player_classes": classes,
            **self.game_kwargs,
        }
        if self.expansions and self.kingdom is None:
            kwargs["allowed_expansions"] = self.expansions
        if self.kingdom is not None:
            kwargs["initcards"] = self.kingdom
            kwargs["num_stacks"] = len(self.kingdom)

        random.seed(seed)
        game = Game(**kwargs)
        game.start_game()

        turns = 0
        while not game.game_over and turns < MAX_TURNS:
            turns += 1
            game.turn()

        scores = game.whoWon()
        players = game.player_list()
        exp_player = players[0] if experiment_first else players[1]
        ctrl_player = players[1] if experiment_first else players[0]

        exp_score = scores[exp_player.name]
        ctrl_score = scores[ctrl_player.name]

        if exp_score > ctrl_score:
            outcome = 1
        elif exp_score < ctrl_score:
            outcome = -1
        else:
            outcome = 0

        replay = self._build_replay_command(seed, experiment_first)
        return GameResult(
            seed=seed,
            outcome=outcome,
            experiment_first=experiment_first,
            replay_command=replay,
        )

    def run(self) -> MatchupSummary:
        """Run all games and return the summary."""
        kingdom = self._resolve_kingdom()
        if kingdom is not None:
            self.kingdom = kingdom

        half = self.num_games // 2
        summary = MatchupSummary()

        for i in range(self.num_games):
            experiment_first = i < half
            seed = random.randint(0, 2**31 - 1)
            result = self._run_single_game(seed, experiment_first)
            summary.results.append(result)

        return summary


def _resolve_bot(name: str) -> type[Player]:
    """Look up a bot class by name."""
    bots = get_all_bots()
    if name in bots:
        return bots[name]
    raise SystemExit(f"Unknown bot: {name!r}. Available: {', '.join(sorted(bots))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bot-vs-bot Dominion matchup")
    parser.add_argument("--control", required=True, help="Control bot class name")
    parser.add_argument("--experiment", required=True, help="Experiment bot class name")
    parser.add_argument("--games", type=int, default=100, help="Number of games (rounded up to even)")
    parser.add_argument("--kingdom", nargs="+", default=None, help="Specific kingdom cards")
    parser.add_argument("--kingdom-seed", type=int, default=None, help="Seed for random kingdom generation")
    parser.add_argument("--random", action="store_true", default=False, help="Fully random kingdom each game")
    parser.add_argument("--seed", type=int, default=None, help="Master random seed for reproducibility")
    parser.add_argument("--prosperity", action="store_true", default=False, help="Use Colony/Platinum")
    parser.add_argument(
        "--experiment-first",
        action="store_true",
        default=False,
        help="Force experiment bot to play first (for single-game replay)",
    )
    parser.add_argument(
        "--expansions",
        nargs="+",
        default=None,
        metavar="EXPANSION",
        help=(
            f"Restrict random kingdom to cards from these expansions "
            f"(available: {', '.join(sorted(EXPANSION_NAMES))})"
        ),
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    control_cls = _resolve_bot(args.control)
    experiment_cls = _resolve_bot(args.experiment)

    expansions: list[CardExpansion] | None = None
    if args.expansions:
        expansions = []
        for name in args.expansions:
            key = name.lower()
            if key not in EXPANSION_NAMES:
                raise SystemExit(
                    f"Unknown expansion: {name!r}. Available: {', '.join(sorted(EXPANSION_NAMES))}"
                )
            expansions.append(EXPANSION_NAMES[key])

    game_kwargs: dict[str, Any] = {}
    if args.prosperity:
        game_kwargs["prosperity"] = True

    matchup = Matchup(
        control_bot=control_cls,
        experiment_bot=experiment_cls,
        num_games=args.games,
        kingdom=args.kingdom,
        kingdom_seed=args.kingdom_seed,
        expansions=expansions,
        game_kwargs=game_kwargs if game_kwargs else None,
    )

    print(f"Matchup: {args.experiment} (experiment) vs {args.control} (control)")
    print(f"Games: {matchup.num_games}")
    if matchup.kingdom:
        print(f"Kingdom: {', '.join(matchup.kingdom)}")
    elif matchup.kingdom_seed is not None:
        print(f"Kingdom seed: {matchup.kingdom_seed}")
    else:
        print("Kingdom: random each game")
    if matchup.expansions:
        print(f"Expansions: {', '.join(e.name.lower() for e in matchup.expansions)}")
    print()

    summary = matchup.run()
    summary.print()

    sys.exit(0)


if __name__ == "__main__":
    main()
