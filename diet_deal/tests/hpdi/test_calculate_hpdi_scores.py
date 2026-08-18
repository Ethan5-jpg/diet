import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "hpdi" / "05_calculate_hpdi_scores.py"
SPEC = importlib.util.spec_from_file_location("hpdi_score_calculator", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
SCORER = importlib.util.module_from_spec(SPEC)
sys.modules["hpdi_score_calculator"] = SCORER
SPEC.loader.exec_module(SCORER)


class HpdiEnergyAdjustmentTests(unittest.TestCase):
    def test_energy_adjustment_returns_sample_standardized_z_score(self):
        raw_score = pd.Series([18, 25, 30, 35, 40, 90], dtype="float64")
        energy = pd.Series(
            [1000, 1800, 1200, 2400, 1500, 2100], dtype="float64"
        )

        winsorized, adjusted, standardized, parameters = (
            SCORER.energy_adjust_score(raw_score, energy)
        )

        self.assertEqual(int(winsorized.notna().sum()), 6)
        self.assertEqual(int(adjusted.notna().sum()), 6)
        self.assertEqual(int(standardized.notna().sum()), 6)
        self.assertAlmostEqual(float(standardized.mean()), 0.0, places=12)
        self.assertAlmostEqual(
            float(standardized.std(ddof=1)), 1.0, places=12
        )
        self.assertAlmostEqual(float(adjusted.corr(energy)), 0.0, places=12)
        self.assertEqual(parameters["winsorized_low_count"], 1)
        self.assertEqual(parameters["winsorized_high_count"], 1)
        self.assertEqual(
            parameters["z_score_standard_deviation_definition"],
            "sample SD, ddof=1",
        )


class HpdiWeightEligibilityTests(unittest.TestCase):
    def test_incomplete_weight_is_not_silently_scored_as_zero_intake(self):
        rows = []
        energy_by_participant = [1200, 1600, 1300, 1700, 1500]
        for participant_number in range(1, 6):
            row = {
                "participant_id": "p{}".format(participant_number),
                "cohort": "10k",
                "research_stage": "00_00_visit",
                "mean_daily_calories_kcal_proxy": energy_by_participant[
                    participant_number - 1
                ],
                "total_weight_imputed_events": (
                    1 if participant_number == 1 else 0
                ),
                "hpdi_any_weight_imputed": participant_number == 1,
            }
            for component_number, component in enumerate(SCORER.COMPONENTS):
                row[SCORER.intake_column(component)] = (
                    10 * participant_number + component_number
                )
                row[SCORER.weight_complete_column(component)] = True
            rows.append(row)
        rows[-1][SCORER.weight_complete_column("vegetables")] = False
        intakes = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv = Path(temp_dir) / "hpdi_intakes.csv"
            output_dir = Path(temp_dir) / "output"
            intakes.to_csv(input_csv, index=False, encoding="utf-8-sig")
            _, score_path, qc_path, adjustment_path = (
                SCORER.calculate_hpdi_scores(
                    input_csv, output_dir
                )
            )
            scores = pd.read_csv(score_path, encoding="utf-8-sig")
            qc = pd.read_csv(qc_path, encoding="utf-8-sig")
            adjustment = pd.read_csv(
                adjustment_path, encoding="utf-8-sig"
            )

        complete = scores.loc[scores["participant_id"].ne("p5")]
        incomplete = scores.loc[scores["participant_id"].eq("p5")].iloc[0]
        self.assertTrue(complete["hpdi_score_raw_18_90"].notna().all())
        self.assertEqual(
            int(
                scores.loc[
                    scores["participant_id"].eq("p1"),
                    "total_weight_imputed_events",
                ].iloc[0]
            ),
            1,
        )
        self.assertTrue(
            bool(
                scores.loc[
                    scores["participant_id"].eq("p1"),
                    "hpdi_any_weight_imputed",
                ].iloc[0]
            )
        )
        self.assertFalse(bool(incomplete["hpdi_weight_complete_for_primary_score"]))
        self.assertTrue(pd.isna(incomplete["vegetables_intake_quintile"]))
        self.assertTrue(pd.isna(incomplete["hpdi_score_raw_18_90"]))
        self.assertTrue(pd.isna(incomplete["hpdi_score_energy_adjusted"]))
        self.assertTrue(pd.isna(incomplete["hpdi_score_energy_adjusted_z"]))
        self.assertTrue(
            complete["hpdi_score_energy_adjusted_z"].notna().all()
        )
        self.assertAlmostEqual(
            float(complete["hpdi_score_energy_adjusted_z"].mean()),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(complete["hpdi_score_energy_adjusted_z"].std(ddof=1)),
            1.0,
            places=12,
        )
        qc_by_metric = qc.set_index("metric")["value"]
        self.assertEqual(
            int(
                float(
                    qc_by_metric[
                        "participants_with_energy_adjusted_score_z"
                    ]
                )
            ),
            4,
        )
        self.assertAlmostEqual(
            float(qc_by_metric["hpdi_energy_adjusted_score_z_mean"]),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(adjustment["z_score_standard_deviation"].iloc[0]),
            1.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
