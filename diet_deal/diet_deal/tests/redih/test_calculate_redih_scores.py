import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "redih" / "05_calculate_redih_scores.py"
)
SPEC = importlib.util.spec_from_file_location("redih_scores", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
CALCULATOR = importlib.util.module_from_spec(SPEC)
sys.modules["redih_scores"] = CALCULATOR
SPEC.loader.exec_module(CALCULATOR)


def score_input(count):
    data = pd.DataFrame(
        {
            "participant_id": ["p{:02d}".format(index) for index in range(count)],
            "cohort": ["10k"] * count,
            "research_stage": ["00_00_visit"] * count,
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
            ][:count],
            "alcohol_complete": [True] * count,
            "energy_complete": [True] * count,
            "redih_all_current_score_component_weights_complete": [True] * count,
        }
    )
    for component in CALCULATOR.CURRENT_SCORE_COMPONENTS:
        data["mean_daily_{}_g".format(component)] = 0.0
    data["mean_daily_wine_g"] = [float(index) for index in range(count)]
    return data


class RedihScoreTests(unittest.TestCase):
    def test_wine_weight_energy_adjustment_z_and_quintiles(self):
        data = score_input(10)

        scores, parameters, distributions, qc = (
            CALCULATOR.calculate_redih_scores(data)
        )

        self.assertAlmostEqual(float(scores.loc[9, "edih_score_raw_17"]), -1.485)
        self.assertAlmostEqual(float(scores.loc[9, "redih_score_raw_17"]), 1.485)
        self.assertAlmostEqual(
            float(scores["redih_score_energy_adjusted_17"].corr(
                scores["mean_daily_energy_kcal"]
            )),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(scores["redih_score_energy_adjusted_z_17"].mean()),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(scores["redih_score_energy_adjusted_z_17"].std(ddof=1)),
            1.0,
            places=12,
        )
        quintiles = scores["redih_energy_adjusted_quintile_17"].value_counts()
        self.assertEqual(quintiles.sort_index().to_dict(), {1: 2, 2: 2, 3: 2, 4: 2, 5: 2})
        self.assertEqual(
            parameters.loc[0, "script_version"],
            "2026-08-14-redih-scores-v3",
        )
        self.assertEqual(
            parameters.loc[0, "wine_intake_definition"],
            "wine beverage weight_g/day; not pure alcohol_g/day",
        )
        self.assertIn(
            "redih_score_energy_adjusted_z_17",
            set(distributions["score_variable"]),
        )
        qc_values = dict(zip(qc["metric"], qc["value"]))
        self.assertEqual(int(qc_values["eligible_participant_count"]), 10)

    def test_incomplete_component_weight_retains_row_but_scores_are_missing(self):
        data = score_input(6)
        data.loc[5, "redih_all_current_score_component_weights_complete"] = False

        scores, _, _, qc = CALCULATOR.calculate_redih_scores(data)

        self.assertFalse(bool(scores.loc[5, "redih_score_complete"]))
        for column in CALCULATOR.SCORE_COLUMNS:
            self.assertTrue(pd.isna(scores.loc[5, column]))
        qc_values = dict(zip(qc["metric"], qc["value"]))
        self.assertEqual(int(qc_values["eligible_participant_count"]), 5)
        self.assertEqual(
            int(qc_values["component_weight_incomplete_participant_count"]), 1
        )


if __name__ == "__main__":
    unittest.main()
