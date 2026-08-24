import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "02_build_covariate_master.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_covariate_master",
        str(SCRIPT_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


covariates = load_module()


class BuildCovariateMasterTests(unittest.TestCase):
    def _visit(self, participant_id, array_index=0, **values):
        row = {
            "participant_id": participant_id,
            "cohort": "10k",
            "research_stage": "00_00_visit",
            "array_index": array_index,
        }
        row.update(values)
        return row

    def _inputs(self):
        population = pd.DataFrame(
            [
                {
                    "participant_id": "1001",
                    "cohort": "10k",
                    "month_of_birth": 1,
                    "year_of_birth": 1980,
                    "sex": 0,
                },
                {
                    "participant_id": "1002",
                    "cohort": "10k",
                    "month_of_birth": 6,
                    "year_of_birth": 1970,
                    "sex": 1,
                },
                {
                    "participant_id": "1003",
                    "cohort": "10k",
                    "month_of_birth": 12,
                    "year_of_birth": 1990,
                    "sex": "Female",
                },
            ]
        )

        cgm = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    connection_id="c1",
                    collection_timestamp="2020-01-15T08:00:00+02:00",
                    cgm_device_type="libre_pro",
                ),
                self._visit(
                    "1002",
                    connection_id="c2",
                    collection_timestamp="2020-06-15T08:00:00+03:00",
                    cgm_device_type="libre_pro_iq",
                ),
            ]
        )

        sociodemographics = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    education=1,
                    collection_timestamp="2020-01-01T10:00:00+02:00",
                ),
                self._visit(
                    "1002",
                    education=2,
                    collection_timestamp="2020-06-01T10:00:00+03:00",
                ),
            ]
        )

        lifestyle = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    collection_timestamp="2020-01-02T10:00:00+02:00",
                    smoking_current_status=0,
                    smoking_past_frequency=4,
                    sleep_hours_daily=7,
                    activity_walking_10min_days_weekly=5,
                    activity_walking_minutes_daily=30,
                    activity_moderate_days_weekly=2,
                    activity_moderate_minutes_daily=45,
                    activity_vigorous_days_weekly=1,
                    activity_vigorous_minutes_daily=20,
                ),
                self._visit(
                    "1002",
                    collection_timestamp="2020-06-02T10:00:00+03:00",
                    smoking_current_status=1,
                    smoking_past_frequency=np.nan,
                    sleep_hours_daily=6,
                    activity_walking_10min_days_weekly=-1,
                    activity_walking_minutes_daily=30,
                    activity_moderate_days_weekly=2,
                    activity_moderate_minutes_daily=45,
                    activity_vigorous_days_weekly=1,
                    activity_vigorous_minutes_daily=20,
                ),
            ]
        )

        medications = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    array_index=0,
                    collection_timestamp="2020-01-03T10:00:00+02:00",
                    medication="vitamin D",
                    atc3="['A11']",
                    atc4="['A11CC']",
                    atc5="['A11CC05']",
                ),
                self._visit(
                    "1001",
                    array_index=1,
                    collection_timestamp="2020-01-03T10:00:00+02:00",
                    medication="hormone and anti-inflammatory",
                    atc3="['G03', 'M01', 'A10']",
                    atc4="[]",
                    atc5="[]",
                ),
                self._visit(
                    "1002",
                    collection_timestamp="2020-06-03T10:00:00+03:00",
                    medication="no medications taken",
                    atc3="[]",
                    atc4="[]",
                    atc5="[]",
                ),
            ]
        )

        anthropometrics = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    collection_timestamp="2020-01-15T09:00:00+02:00",
                    bmi=24.0,
                ),
                self._visit(
                    "1002",
                    collection_timestamp="2020-06-15T09:00:00+03:00",
                    bmi=30.0,
                ),
            ]
        )

        family_history = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    collection_timestamp="2020-01-04T10:00:00+02:00",
                    type_1_diabetes_family_number=0,
                    type_2_diabetes_family_number=1,
                    sudden_death_family=1,
                ),
                self._visit(
                    "1002",
                    collection_timestamp="2020-06-04T10:00:00+03:00",
                    type_1_diabetes_family_number=np.nan,
                    type_2_diabetes_family_number=np.nan,
                    sudden_death_family=0,
                ),
            ]
        )

        medical_conditions = pd.DataFrame(
            [
                self._visit(
                    "1001",
                    collection_timestamp="2020-01-05T10:00:00+02:00",
                    medical_condition="Type 2 diabetes mellitus",
                    icd11_code="5A11",
                ),
                self._visit(
                    "1002",
                    collection_timestamp="2020-06-05T10:00:00+03:00",
                    medical_condition="Hypertension",
                    icd11_code="BA00",
                ),
            ]
        )

        alcohol = pd.DataFrame(
            [
                {
                    "participant_id": "1001",
                    "cohort": "10k",
                    "research_stage": "00_00_visit",
                    "alcohol_complete": True,
                    "mean_daily_alcohol_g": 1.2,
                },
                {
                    "participant_id": "1002",
                    "cohort": "10k",
                    "research_stage": "00_00_visit",
                    "alcohol_complete": False,
                    "mean_daily_alcohol_g": 0.5,
                },
            ]
        )

        return {
            "population": population,
            "cgm": cgm,
            "sociodemographics": sociodemographics,
            "lifestyle": lifestyle,
            "medications": medications,
            "anthropometrics": anthropometrics,
            "family_history": family_history,
            "medical_conditions": medical_conditions,
            "alcohol": alcohol,
        }

    def test_builds_one_row_per_population_participant_and_keeps_missing(self):
        master, reports, summary = covariates.build_covariate_master(
            **self._inputs()
        )

        self.assertEqual(master["participant_id"].tolist(), ["1001", "1002", "1003"])
        self.assertEqual(len(master), 3)
        self.assertFalse(master.duplicated(["participant_id", "cohort"]).any())

        p1 = master.set_index("participant_id").loc["1001"]
        p2 = master.set_index("participant_id").loc["1002"]
        p3 = master.set_index("participant_id").loc["1003"]

        self.assertAlmostEqual(float(p1["age_years"]), 40.0, delta=0.1)
        self.assertEqual(p1["sex"], "female")
        self.assertEqual(p2["sex"], "male")
        self.assertEqual(p1["education_level"], "education_code_1")
        self.assertEqual(p2["education_level"], "education_code_2")
        self.assertEqual(p1["smoking_status"], "never")
        self.assertEqual(p2["smoking_status"], "current")
        self.assertEqual(float(p1["sleep_duration_hours_day"]), 7.0)

        expected_met_h_day = (3.3 * 5 * 30 + 4.0 * 2 * 45 + 8.0 * 1 * 20) / 60.0 / 7.0
        self.assertAlmostEqual(
            float(p1["physical_activity_met_h_day"]),
            expected_met_h_day,
            places=8,
        )
        self.assertTrue(math.isnan(float(p2["physical_activity_met_h_day"])))

        self.assertEqual(float(p1["vitamin_use"]), 1.0)
        self.assertEqual(float(p1["hormone_use"]), 1.0)
        self.assertEqual(float(p1["nsaid_aspirin_use"]), 1.0)
        self.assertEqual(float(p1["a10_medication_use"]), 1.0)
        self.assertEqual(float(p2["vitamin_use"]), 0.0)
        self.assertEqual(float(p2["hormone_use"]), 0.0)
        self.assertTrue(math.isnan(float(p3["vitamin_use"])))

        self.assertEqual(float(p1["family_history_diabetes"]), 1.0)
        self.assertEqual(float(p1["family_history_cvd"]), 1.0)
        self.assertTrue(math.isnan(float(p2["family_history_diabetes"])))
        self.assertEqual(float(p2["family_history_cvd"]), 0.0)
        self.assertTrue(math.isnan(float(p3["family_history_diabetes"])))

        self.assertEqual(float(p1["known_diabetes"]), 1.0)
        self.assertEqual(float(p2["known_diabetes"]), 0.0)
        self.assertTrue(math.isnan(float(p3["known_diabetes"])))
        self.assertEqual(float(p1["exclude_diabetes_or_a10"]), 1.0)
        self.assertEqual(float(p2["exclude_diabetes_or_a10"]), 0.0)
        self.assertTrue(math.isnan(float(p3["exclude_diabetes_or_a10"])))

        self.assertEqual(float(p1["alcohol_intake_g_day"]), 1.2)
        self.assertTrue(math.isnan(float(p2["alcohol_intake_g_day"])))
        self.assertTrue(bool(p1["amed_model2_covariates_complete"]))
        self.assertTrue(bool(p1["hpdi_model4_covariates_complete"]))
        self.assertFalse(bool(p2["amed_model2_covariates_complete"]))
        self.assertFalse(bool(p3["amed_model2_covariates_complete"]))

        self.assertIn("completeness", reports)
        self.assertIn("source_coverage", reports)
        self.assertIn("categorical_values", reports)
        self.assertEqual(summary["participants_in_master"], 3)

    def test_physical_activity_respects_conditional_duration_questions(self):
        lifestyle = pd.DataFrame(
            {
                "activity_walking_10min_days_weekly": [0, 0],
                "activity_walking_minutes_daily": [np.nan, np.nan],
                "activity_moderate_days_weekly": [2, 2],
                "activity_moderate_minutes_daily": [30, np.nan],
                "activity_vigorous_days_weekly": [0, 0],
                "activity_vigorous_minutes_daily": [np.nan, np.nan],
            }
        )

        result = covariates.derive_physical_activity(lifestyle)

        self.assertAlmostEqual(float(result.iloc[0]), 4.0 * 2 * 30 / 60 / 7)
        self.assertTrue(math.isnan(float(result.iloc[1])))

    def test_family_diabetes_preserves_unknown_skipped_counts(self):
        family = pd.DataFrame(
            {
                "type_1_diabetes_family_number": [np.nan, 1, 0],
                "type_2_diabetes_family_number": [np.nan, np.nan, 0],
            }
        )

        result = covariates._family_count_flag(
            family,
            ["type_1_diabetes_family_number", "type_2_diabetes_family_number"],
        )

        self.assertTrue(math.isnan(float(result.iloc[0])))
        self.assertEqual(float(result.iloc[1]), 1.0)
        self.assertEqual(float(result.iloc[2]), 0.0)

    def test_duplicate_selected_cgm_connection_is_rejected(self):
        inputs = self._inputs()
        inputs["cgm"] = pd.concat(
            [inputs["cgm"], inputs["cgm"].iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "baseline CGM.*duplicate"):
            covariates.build_covariate_master(**inputs)

    def test_writes_master_and_three_reports(self):
        master, reports, summary = covariates.build_covariate_master(
            **self._inputs()
        )
        with tempfile.TemporaryDirectory(prefix="covariate_master_") as directory:
            paths = covariates.write_outputs(
                master,
                reports,
                summary,
                Path(directory),
            )
            self.assertEqual(set(paths), {
                "master",
                "completeness",
                "source_coverage",
                "categorical_values",
                "summary",
            })
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            written = pd.read_csv(paths["master"], dtype={"participant_id": str})
            self.assertEqual(written["participant_id"].tolist(), ["1001", "1002", "1003"])


if __name__ == "__main__":
    unittest.main()
