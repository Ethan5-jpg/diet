"""Short summaries must preserve paired outcomes and avoid mixing round winners."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import run
from experiment import INTERACTIONS, REFERENCES, model_names
from reporting import compact_summary


def sample():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text()
    )
    scores = [
        dict(model=name, repeat=repeat, n=100, R2=0.05)
        for name in model_names(config) + list(REFERENCES)
        for repeat in range(2)
    ]
    comparisons = [
        dict(model=name, repeat=repeat, delta_R2=0.001 if repeat == 0 else -0.003)
        for name in INTERACTIONS
        for repeat in range(2)
    ]
    return config, dict(scores=scores, comparisons=comparisons)


def set_score(result, name, values):
    for row in result["scores"]:
        if row["model"] == name:
            row["R2"] = values[row["repeat"]]


class ReportingChecks(unittest.TestCase):
    def test_short_summary_preserves_negative_deltas_and_uses_lower_round_ranking(self):
        config, result = sample()
        set_score(result, "catboost_oral", [0.25, 0.26])
        set_score(result, "xgboost_oral", [0.32, 0.15])
        original = copy.deepcopy(result)
        text = compact_summary(result, config)
        self.assertLess(text.index("| catboost_oral |"), text.index("| xgboost_oral |"))
        self.assertIn("均≥0.20：是", text)
        self.assertIn("均≥0.30：否", text)
        self.assertIn("-0.003000", text)
        for name in INTERACTIONS:
            self.assertIn("| " + name + " |", text)
        self.assertLessEqual(len(text.splitlines()), 28)
        self.assertEqual(result, original)

    def test_goal_requires_one_pipeline_to_pass_both_rounds_at_original_precision(self):
        config, result = sample()
        set_score(result, "catboost_oral", [0.31, 0.10])
        set_score(result, "xgboost_oral", [0.10, 0.31])
        self.assertIn("均≥0.20：否", compact_summary(result, config))
        set_score(result, "fusion", [0.1999999, 0.4])
        self.assertIn("均≥0.20：否", compact_summary(result, config))

    def test_unstarted_run_reports_status_without_creating_results(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name) / "not_started"
            self.assertIsNone(run.report(folder))
            self.assertFalse(folder.exists())
