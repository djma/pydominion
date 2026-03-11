#!/usr/bin/env python

import unittest

from dominion.bots.BigMoney import BigMoney
from dominion.bots.matchup import GameResult, Matchup, MatchupSummary


def _result(outcome: int) -> GameResult:
    return GameResult(seed=0, outcome=outcome, experiment_first=True, replay_command="")


class TestMatchupSummaryAdaptiveStop(unittest.TestCase):
    def test_rejects_half_win_rate(self) -> None:
        results = [_result(1) for _ in range(80)] + [_result(-1) for _ in range(20)]
        summary = MatchupSummary(results=results)

        self.assertIsNotNone(summary.p_value_vs_fair)
        self.assertLess(summary.p_value_vs_fair, 0.05)
        reason = summary.adaptive_stop_reason()
        self.assertIsNotNone(reason)
        self.assertIn("Rejected 50% win rate", reason or "")

    def test_stops_on_ci_precision(self) -> None:
        results = [_result(1) for _ in range(5000)] + [_result(-1) for _ in range(5000)]
        summary = MatchupSummary(results=results)

        self.assertEqual(summary.score_mean, 0.5)
        self.assertEqual(summary.p_value_vs_fair, 1.0)
        self.assertIsNotNone(summary.ci95_width)
        self.assertLessEqual(summary.ci95_width, 0.02)
        reason = summary.adaptive_stop_reason()
        self.assertIsNotNone(reason)
        self.assertIn("95% CI width", reason or "")

    def test_no_stop_with_too_few_games(self) -> None:
        summary = MatchupSummary(results=[_result(1)])

        self.assertIsNone(summary.p_value_vs_fair)
        self.assertIsNone(summary.ci95)
        self.assertIsNone(summary.adaptive_stop_reason())


class TestMatchupConstructor(unittest.TestCase):
    def test_num_games_none_is_adaptive(self) -> None:
        matchup = Matchup(control_bot=BigMoney, experiment_bot=BigMoney, num_games=None)
        self.assertIsNone(matchup.num_games)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
