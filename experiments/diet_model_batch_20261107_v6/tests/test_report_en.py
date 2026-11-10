"""English display preserves the distinction between prediction and interaction."""

import copy
import unittest

from show_results_en import render


def fixture():
    values = {
        "v4_hgb_oral": [0.10, 0.11],
        "kernel_main": [0.05, 0.06],
        "model_alpha": [0.20, 0.21],
        "kernel_linear": [0.051, 0.058],
    }
    scores = [
        dict(model=name, repeat=repeat, n=100, R2=value)
        for name, pair in values.items()
        for repeat, value in enumerate(pair)
    ]
    comparisons = [
        dict(
            model="kernel_linear",
            repeat=r,
            comparator="kernel_main",
            comparison="explicit_interaction",
            delta_R2=d,
            improved_folds=f,
        )
        for r, d, f in [(0, 0.001, 3), (1, -0.002, 1)]
    ]
    return dict(status="COMPLETE", scores=scores, comparisons=comparisons)


class EnglishReportChecks(unittest.TestCase):
    def test_ascii_all_models_and_matched_interaction_comparator(self):
        data = fixture()
        original = copy.deepcopy(data)
        text = render(data, "synthetic_test")
        self.assertTrue(text.isascii())
        self.assertIn("+0.001000", text)
        self.assertIn("-0.002000", text)
        self.assertIn("Best by the lower of two repeat R2 values: model_alpha", text)
        self.assertLessEqual(len(text.splitlines()), 32)
        for name in {row["model"] for row in data["scores"]}:
            self.assertIn(name, text)
        self.assertEqual(data, original)

    def test_partial_or_missing_repeat_and_duplicate_scores_are_rejected(self):
        for change in ("partial", "missing", "duplicate"):
            with self.subTest(change=change):
                data = fixture()
                if change == "partial":
                    data["status"] = "PARTIAL"
                elif change == "missing":
                    data["scores"].pop()
                else:
                    data["scores"].append(data["scores"][0])
                with self.assertRaises(ValueError):
                    render(data, "synthetic_test")

    def test_unmatched_interaction_comparator_is_rejected(self):
        data = fixture()
        data["comparisons"][0]["comparator"] = "v4_hgb_oral"
        with self.assertRaisesRegex(ValueError, "comparator"):
            render(data, "synthetic_test")
