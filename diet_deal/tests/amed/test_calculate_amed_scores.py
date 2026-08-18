import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "amed" / "05_calculate_amed_scores.py"
SPEC = importlib.util.spec_from_file_location("amed_score_calculator", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
CALCULATOR = importlib.util.module_from_spec(SPEC)
sys.modules["amed_score_calculator"] = CALCULATOR
SPEC.loader.exec_module(CALCULATOR)


class AmedEnergyAdjustmentTests(unittest.TestCase):
    def test_raw_winsorization_residual_adjustment_z_and_quintiles(self):
        data = pd.DataFrame(
            {
                "amed_total_score_0_8": pd.Series(
                    [0, 0, 0, 1, 2, 3, 4, 5, 6, 8], dtype="Int64"
                ),
                "amed_score_complete": [True] * 10,
                "energy_complete": [True] * 10,
                "mean_daily_energy_kcal": [
                    1000,
                    1800,
                    1200,
                    2400,
                    1500,
                    2100,
                    1300,
                    2700,
                    1600,
                    2300,
                ],
            }
        )

        adjusted, metrics = CALCULATOR.add_energy_adjusted_scores(data)

        winsorized = adjusted["amed_score_winsorized"]
        self.assertEqual(metrics["energy_adjustment_eligible_count"], 10)
        self.assertEqual(
            metrics["script_version"], "2026-08-14-amed-scores-v2"
        )
        self.assertEqual(metrics["winsorized_low_count"], 0)
        self.assertEqual(metrics["winsorized_high_count"], 1)
        self.assertEqual(float(winsorized.iloc[0]), 0.0)
        self.assertLess(float(winsorized.iloc[-1]), 8.0)
        self.assertNotIn(
            "amed_energy_adjusted_score_truncated", adjusted.columns
        )
        self.assertAlmostEqual(
            float(adjusted["amed_energy_adjusted_score"].mean()),
            float(winsorized.mean()),
        )
        self.assertAlmostEqual(
            float(
                adjusted["amed_energy_adjusted_score"].corr(
                    adjusted["mean_daily_energy_kcal"]
                )
            ),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(adjusted["amed_energy_adjusted_score_z"].mean()),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(adjusted["amed_energy_adjusted_score_z"].std(ddof=1)),
            1.0,
            places=12,
        )
        quintile_counts = (
            adjusted["amed_energy_adjusted_score_quintile"]
            .value_counts()
            .sort_index()
        )
        self.assertEqual(quintile_counts.to_dict(), {1: 2, 2: 2, 3: 2, 4: 2, 5: 2})
        self.assertEqual(
            adjusted["amed_total_score_0_8"].tolist(),
            data["amed_total_score_0_8"].tolist(),
        )

    def test_incomplete_energy_keeps_raw_score_but_not_adjusted_score(self):
        data = pd.DataFrame(
            {
                "amed_total_score_0_8": pd.Series(
                    [0, 1, 2, 3, 4, 5], dtype="Int64"
                ),
                "amed_score_complete": [True] * 6,
                "energy_complete": [True, True, True, True, True, False],
                "mean_daily_energy_kcal": [
                    1000,
                    1200,
                    1500,
                    1900,
                    2400,
                    pd.NA,
                ],
            }
        )

        adjusted, metrics = CALCULATOR.add_energy_adjusted_scores(data)

        self.assertEqual(metrics["energy_adjustment_eligible_count"], 5)
        self.assertEqual(metrics["energy_adjustment_missing_count"], 1)
        self.assertEqual(int(adjusted["amed_total_score_0_8"].iloc[-1]), 5)
        self.assertTrue(
            pd.isna(adjusted["amed_energy_adjusted_score"].iloc[-1])
        )
        self.assertTrue(pd.isna(adjusted["amed_score_winsorized"].iloc[-1]))
        self.assertTrue(
            pd.isna(adjusted["amed_energy_adjusted_score_z"].iloc[-1])
        )
        self.assertTrue(
            pd.isna(
                adjusted["amed_energy_adjusted_score_quintile"].iloc[-1]
            )
        )


if __name__ == "__main__":
    unittest.main()
