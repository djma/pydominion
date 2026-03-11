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
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from dominion.Card import CardExpansion
from dominion.Player import Player
from dominion.bots import get_all_bots

EXPANSION_NAMES: dict[str, CardExpansion] = {e.name.lower(): e for e in CardExpansion if e != CardExpansion.TEST}


@dataclass
class TurnRecord:
    """Per-turn statistics for a single player."""

    coins_available: int  # coins spent on buys + leftover unused coins
    vp: int  # VP score at end of this turn


@dataclass
class GameResult:
    """Statistics for a single game."""

    seed: int
    outcome: int  # 1 = experiment won, -1 = control won, 0 = draw
    experiment_first: bool
    replay_command: str
    turns: int = 0
    experiment_turn_records: list[TurnRecord] = field(default_factory=list)
    control_turn_records: list[TurnRecord] = field(default_factory=list)


@dataclass
class MatchupSummary:
    """Aggregate statistics across all games in a matchup."""

    results: list[GameResult] = field(default_factory=list)
    stop_reason: str | None = None

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
        return self.score_mean * 100

    @property
    def score_mean(self) -> float:
        """Average score in [0, 1], where draw counts as half."""
        if not self.results:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total

    @property
    def score_std_error(self) -> float | None:
        """Binomial-style standard error for score_mean (None until at least 2 games)."""
        n = self.total
        if n < 2:
            return None
        p = self.score_mean
        return math.sqrt(p * (1.0 - p) / n)

    @property
    def ci95_half_width(self) -> float | None:
        """95% CI half-width for score_mean."""
        se = self.score_std_error
        if se is None:
            return None
        return 1.96 * se

    @property
    def ci95(self) -> tuple[float, float] | None:
        """95% CI (lower, upper) for score_mean in [0, 1]."""
        half_width = self.ci95_half_width
        if half_width is None:
            return None
        mean = self.score_mean
        return max(0.0, mean - half_width), min(1.0, mean + half_width)

    @property
    def ci95_width(self) -> float | None:
        """95% CI full width for score_mean."""
        half_width = self.ci95_half_width
        if half_width is None:
            return None
        return 2.0 * half_width

    def adaptive_stop_reason(self, ci_width: float = 0.05) -> str | None:
        """Return stop reason when adaptive CI-width criterion is met."""
        ci_width_value = self.ci95_width
        if ci_width_value is not None and ci_width_value < ci_width:
            return f"95% CI width {ci_width_value * 100:.2f}% < {ci_width * 100:.2f}%"

        return None

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

    @property
    def avg_turns_per_game(self) -> float:
        """Average total game turns (both players combined) across all games."""
        if not self.results:
            return 0.0
        return sum(r.turns for r in self.results) / len(self.results)

    @staticmethod
    def _timeseries(records_lists: list[list[TurnRecord]], attr: str) -> list[float]:
        """Per-turn average of `attr` across games (index = turn number - 1)."""
        max_turns = max((len(r) for r in records_lists), default=0)
        result = []
        for t in range(max_turns):
            values = [getattr(r[t], attr) for r in records_lists if t < len(r)]
            result.append(sum(values) / len(values))
        return result

    @property
    def experiment_coins_timeseries(self) -> list[float]:
        return self._timeseries([r.experiment_turn_records for r in self.results], "coins_available")

    @property
    def control_coins_timeseries(self) -> list[float]:
        return self._timeseries([r.control_turn_records for r in self.results], "coins_available")

    @property
    def experiment_vp_timeseries(self) -> list[float]:
        return self._timeseries([r.experiment_turn_records for r in self.results], "vp")

    @property
    def control_vp_timeseries(self) -> list[float]:
        return self._timeseries([r.control_turn_records for r in self.results], "vp")

    @staticmethod
    def _fmt_timeseries(values: list[float]) -> str:
        return ", ".join(f"{v:.1f}" for v in values)

    def print(self) -> None:
        print(f"Games: {self.total}  (W {self.wins} / L {self.losses} / D {self.draws})")
        print(f"Win %:            {self.win_pct:.1f}%")
        print(f"Draw %:           {self.draw_pct:.1f}%")
        print(f"Win % as 1st plr: {self.first_player_win_pct:.1f}%")
        print(f"Avg turns/game:   {self.avg_turns_per_game:.1f}")
        print(f"Coins/turn (exp): {self._fmt_timeseries(self.experiment_coins_timeseries)}")
        print(f"Coins/turn (ctl): {self._fmt_timeseries(self.control_coins_timeseries)}")
        print(f"VP/turn (exp):    {self._fmt_timeseries(self.experiment_vp_timeseries)}")
        print(f"VP/turn (ctl):    {self._fmt_timeseries(self.control_vp_timeseries)}")


MAX_TURNS = 400


class TrackingMixin:
    """Mixin that records per-turn coin and VP stats on any Player subclass.

    Tracks coins_available = coins spent on buys + coins left unused at end of turn.
    Tracks VP score at end of each turn.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._turn_records: list[TurnRecord] = []
        self._coins_spent_this_turn: int = 0

    def start_turn(self) -> None:
        self._coins_spent_this_turn = 0
        super().start_turn()  # type: ignore[misc]

    def buy_card(self, card_name: str) -> None:
        before = self.coins.get()  # type: ignore[attr-defined]
        super().buy_card(card_name)  # type: ignore[misc]
        self._coins_spent_this_turn += max(0, before - self.coins.get())  # type: ignore[attr-defined]

    def end_turn(self) -> None:
        coins_leftover = self.coins.get()  # type: ignore[attr-defined]
        total_coins = self._coins_spent_this_turn + coins_leftover
        vp = self.get_score()  # type: ignore[attr-defined]
        self._turn_records.append(TurnRecord(coins_available=total_coins, vp=vp))
        super().end_turn()  # type: ignore[misc]


class Matchup:
    """Run a series of games between two bot classes."""

    def __init__(
        self,
        control_bot: type[Player],
        experiment_bot: type[Player],
        num_games: int | None = 100,
        kingdom: list[str] | None = None,
        kingdom_seed: int | None = None,
        expansions: list[CardExpansion] | None = None,
        game_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.control_bot = control_bot
        self.experiment_bot = experiment_bot
        self.num_games = None if num_games is None else num_games + (num_games % 2)  # round up to even
        self.kingdom = kingdom
        self.kingdom_seed = kingdom_seed
        self.expansions = expansions
        self.game_kwargs = dict(game_kwargs) if game_kwargs else {}
        self._show_progress = sys.stdout.isatty()
        self._last_progress_len = 0

    def _display_progress(self, game_number: int, turn_number: int) -> None:
        """Update one in-place terminal progress line."""
        if not self._show_progress:
            return
        if self.num_games is None:
            message = f"Running game {game_number} (adaptive) | turn {turn_number}"
        else:
            message = f"Running game {game_number}/{self.num_games} | turn {turn_number}"

        # Pad with spaces so remnants from longer prior messages are cleared.
        padding = max(0, self._last_progress_len - len(message))
        sys.stdout.write(f"\r{message}{' ' * padding}")
        sys.stdout.flush()
        self._last_progress_len = len(message)

    def _clear_progress(self) -> None:
        """Clear the progress line from the terminal."""
        if not self._show_progress:
            return
        if self._last_progress_len > 0:
            sys.stdout.write(f"\r{' ' * self._last_progress_len}\r")
            sys.stdout.flush()
            self._last_progress_len = 0

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

    def _run_single_game(self, seed: int, experiment_first: bool, game_number: int) -> GameResult:
        """Play one game and return the result."""
        from dominion.Game import Game

        class _TrackingExperiment(TrackingMixin, self.experiment_bot):  # type: ignore[valid-type,misc]
            pass

        class _TrackingControl(TrackingMixin, self.control_bot):  # type: ignore[valid-type,misc]
            pass

        if experiment_first:
            classes = [_TrackingExperiment, _TrackingControl]
        else:
            classes = [_TrackingControl, _TrackingExperiment]

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
            self._display_progress(game_number, turns)
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
            turns=turns,
            experiment_turn_records=list(getattr(exp_player, "_turn_records", [])),
            control_turn_records=list(getattr(ctrl_player, "_turn_records", [])),
        )

    def run(self) -> MatchupSummary:
        """Run all games and return the summary."""
        kingdom = self._resolve_kingdom()
        if kingdom is not None:
            self.kingdom = kingdom

        summary = MatchupSummary()

        if self.num_games is not None:
            half = self.num_games // 2
            try:
                for i in range(self.num_games):
                    experiment_first = i < half
                    seed = random.randint(0, 2**31 - 1)
                    result = self._run_single_game(seed, experiment_first, game_number=i + 1)
                    summary.results.append(result)
                return summary
            finally:
                self._clear_progress()

        i = 0
        try:
            while True:
                experiment_first = (i % 2) == 0
                seed = random.randint(0, 2**31 - 1)
                result = self._run_single_game(seed, experiment_first, game_number=i + 1)
                summary.results.append(result)

                stop_reason = summary.adaptive_stop_reason()
                if stop_reason is not None:
                    summary.stop_reason = stop_reason
                    return summary

                i += 1
        finally:
            self._clear_progress()


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
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help=(
            "Number of games (rounded up to even). "
            "If omitted, run adaptively until 95% CI width < 5%."
        ),
    )
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
    if args.games is not None and args.games <= 0:
        raise SystemExit("--games must be positive")

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
    if matchup.num_games is None:
        print("Games: adaptive stopping")
    else:
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
    if matchup.num_games is None:
        if summary.stop_reason:
            print(f"Stop reason:      {summary.stop_reason}")
        if summary.ci95 is not None:
            lower, upper = summary.ci95
            print(f"95% CI:           [{lower * 100:.2f}%, {upper * 100:.2f}%]")

    sys.exit(0)


if __name__ == "__main__":
    main()
