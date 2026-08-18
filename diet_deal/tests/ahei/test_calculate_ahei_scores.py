import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "ahei" / "05_calculate_ahei_scores.py"
SPEC = importlib.util.spec_from_file_location("ahei_score_calculator", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
SCORER = importlib.util.module_from_spec(SPEC)
sys.modules["ahei_score_calculator"] = SCORER
SPEC.loader.exec_module(SCORER)


class AheiComponentScoreTests(unittest.TestCase):
    def test_default_population_path_matches_server_layout(self):
        self.assertEqual(
            SCORER.DEFAULT_POPULATION_CSV,
            SCORER.DATA_DIR / "Transfer" / "population" / "population.csv",
        )

    def test_linear_component_endpoints_and_alcohol_shape(self):
        values = pd.Series([0.0, 2.5, 5.0, 10.0])
        self.assertEqual(
            SCORER.score_beneficial(values, 5.0).tolist(),
            [0.0, 5.0, 10.0, 10.0],
        )
        self.assertEqual(
            SCORER.score_adverse(pd.Series([0.0, 0.5, 1.0, 2.0]), 1.0).tolist(),
            [10.0, 5.0, 0.0, 0.0],
        )
        women = SCORER.score_alcohol(
            pd.Series([0.0, 0.5, 1.5, 2.0, 2.5]),
            pd.Series(["female"] * 5),
        )
        self.assertEqual(women.tolist(), [2.5, 10.0, 10.0, 5.0, 0.0])
        men = SCORER.score_alcohol(
            pd.Series([0.0, 0.5, 2.0, 2.75, 3.5]),
            pd.Series(["male"] * 5),
        )
        self.assertEqual(men.tolist(), [2.5, 10.0, 10.0, 5.0, 0.0])

    def test_sex_specific_whole_grain_thresholds_and_raw_range(self):
        intakes = pd.DataFrame(
            {
                "participant_id": ["f", "m"],
                "cohort": ["10k", "10k"],
                "research_stage": ["00_00_visit", "00_00_visit"],
                "mean_daily_vegetables_servings": [5.0, 5.0],
                "mean_daily_fruit_servings": [4.0, 4.0],
                "mean_daily_whole_grain_proxy_g": [75.0, 90.0],
                "mean_daily_ssb_plus_fruit_juice_servings": [0.0, 0.0],
                "mean_daily_nuts_plus_legumes_servings": [1.0, 1.0],
                "mean_daily_red_plus_processed_meat_servings": [0.0, 0.0],
                "mean_daily_alcohol_servings": [0.5, 2.0],
                "vegetables_complete": [True, True],
                "fruit_complete": [True, True],
                "whole_grains_complete": [True, True],
                "ssb_plus_fruit_juice_complete": [True, True],
                "nuts_plus_legumes_complete": [True, True],
                "red_plus_processed_meat_complete": [True, True],
                "alcohol_complete": [True, True],
                "energy_complete": [True, True],
                "mean_daily_energy_kcal": [1800.0, 2200.0],
            }
        )
        population = pd.DataFrame(
            {"participant_id": ["f", "m"], "cohort": ["10k", "10k"], "sex": [0, 1]}
        )
        scores = SCORER.calculate_component_scores(intakes, population)
        self.assertEqual(scores["whole_grains_score_0_10"].tolist(), [10.0, 10.0])
        self.assertEqual(scores["mAHEI7_raw_0_70"].tolist(), [70.0, 70.0])
        self.assertTrue(scores["mAHEI7_score_complete"].all())
        self.assertNotIn("sodium_score_0_10", scores.columns)
        self.assertNotIn("trans_fat_score_0_10", scores.columns)
        self.assertNotIn("pufa_score_0_10", scores.columns)


class AheiEnergyAdjustmentTests(unittest.TestCase):
    def test_winsor_residual_adjustment_z_and_quintiles(self):
        data = pd.DataFrame(
            {
                "mAHEI7_raw_0_70": [0, 10, 20, 30, 40, 70],
                "mAHEI7_score_complete": [True] * 6,
                "energy_complete": [True] * 6,
                "mean_daily_energy_kcal": [1000, 1800, 1200, 2400, 1500, 2100],
            }
        )
        adjusted, parameters = SCORER.add_energy_adjusted_scores(data)
        self.assertAlmostEqual(
            float(adjusted["mAHEI7_score_energy_adjusted_z"].mean()), 0.0, places=12
        )
        self.assertAlmostEqual(
            float(adjusted["mAHEI7_score_energy_adjusted_z"].std(ddof=1)),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(
                adjusted["mAHEI7_score_energy_adjusted"].corr(
                    adjusted["mean_daily_energy_kcal"]
                )
            ),
            0.0,
            places=12,
        )
        self.assertEqual(parameters["eligible_participant_count"], 6)
        self.assertTrue(
            adjusted["mAHEI7_score_energy_adjusted_quintile"].between(1, 5).all()
        )

    def test_end_to_end_writer_emits_analysis_ready_outputs(self):
        rows = []
        for number in range(6):
            rows.append(
                {
                    "participant_id": "p{}".format(number),
                    "cohort": "10k",
                    "research_stage": "00_00_visit",
                    "mean_daily_vegetables_servings": float(number),
                    "mean_daily_fruit_servings": float(number) / 2,
                    "mean_daily_whole_grain_proxy_g": 15.0 * number,
                    "mean_daily_ssb_plus_fruit_juice_servings": float(number) / 10,
                    "mean_daily_nuts_plus_legumes_servings": float(number) / 5,
                    "mean_daily_red_plus_processed_meat_servings": float(number) / 4,
                    "mean_daily_alcohol_servings": float(number) / 3,
                    "vegetables_complete": True,
                    "fruit_complete": True,
                    "whole_grains_complete": True,
                    "ssb_plus_fruit_juice_complete": True,
                    "nuts_plus_legumes_complete": True,
                    "red_plus_processed_meat_complete": True,
                    "alcohol_complete": True,
                    "energy_complete": True,
                    "mean_daily_energy_kcal": [1000, 1800, 1200, 2400, 1500, 2100][number],
                }
            )
        intakes = pd.DataFrame(rows)
        population = pd.DataFrame(
            {
                "participant_id": ["p{}".format(number) for number in range(6)],
                "cohort": ["10k"] * 6,
                "sex": [0, 1, 0, 1, 0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intake_path = root / "intakes.csv"
            population_path = root / "population.csv"
            output_dir = root / "output"
            intakes.to_csv(intake_path, index=False)
            population.to_csv(population_path, index=False)
            paths = SCORER.calculate_ahei_scores(
                intake_path, population_path, output_dir
            )
            for path in paths:
                self.assertTrue(path.is_file())
            scores = pd.read_csv(paths[1])

        self.assertTrue(scores["mAHEI7_raw_0_70"].between(0, 70).all())
        self.assertAlmostEqual(
            float(scores["mAHEI7_score_energy_adjusted_z"].mean()),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(scores["mAHEI7_score_energy_adjusted_z"].std(ddof=1)),
            1.0,
            places=12,
        )
        self.assertNotIn("sodium_score_0_10", scores.columns)


if __name__ == "__main__":
    unittest.main()
