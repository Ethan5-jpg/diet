import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "01_prepare_covariate_sources.py"
)


def load_conversion_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_covariate_sources",
        str(SCRIPT_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conversion = load_conversion_module()


class PrepareCovariateSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="covariate_conversion_"))
        self.source_root = self.temp_dir / "hpp_datasets"
        self.project_root = self.temp_dir / "co-variant"
        self._write_all_sources()

    def tearDown(self):
        shutil.rmtree(str(self.temp_dir))

    def _write_parquet(self, relative_path, rows, index_columns):
        path = self.source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows).set_index(index_columns)
        frame.to_parquet(str(path))

    def _write_all_sources(self):
        common_rows = [
            {
                "participant_id": "1001",
                "cohort": "10k",
                "research_stage": "00_00_visit",
                "array_index": 0,
            },
            {
                "participant_id": "1002",
                "cohort": "10k",
                "research_stage": "00_00_visit",
                "array_index": 0,
            },
        ]

        self._write_parquet(
            "population/population.parquet",
            [
                {
                    "participant_id": "1001",
                    "cohort": "10k",
                    "month_of_birth": 1,
                    "year_of_birth": 1980,
                    "sex": "Female",
                },
                {
                    "participant_id": "1002",
                    "cohort": "10k",
                    "month_of_birth": 6,
                    "year_of_birth": 1975,
                    "sex": "Male",
                },
            ],
            ["participant_id", "cohort"],
        )

        self._write_parquet(
            "sociodemographics/initial_medical.parquet",
            [dict(row, education=1) for row in common_rows],
            ["participant_id", "cohort", "research_stage", "array_index"],
        )

        self._write_parquet(
            "lifestyle_and_environment/lifestyle_and_environment.parquet",
            [
                dict(
                    row,
                    sleep_hours_daily=7,
                    smoking_current_status=0,
                    activity_walking_10min_days_weekly=5,
                )
                for row in common_rows
            ],
            ["participant_id", "cohort", "research_stage", "array_index"],
        )

        medication_rows = [
            dict(
                common_rows[0],
                medication="vitamin D",
                api=["Cholecalciferol"],
                atc3=["A11"],
                atc4=["A11CC"],
                atc5=["A11CC05"],
                supplements=True,
            ),
            dict(
                common_rows[1],
                medication="no medications taken",
                api=[],
                atc3=[],
                atc4=[],
                atc5=[],
                supplements=False,
            ),
        ]
        self._write_parquet(
            "medications/medications.parquet",
            medication_rows,
            ["participant_id", "cohort", "research_stage", "array_index"],
        )

        self._write_parquet(
            "anthropometrics/anthropometrics.parquet",
            [dict(row, bmi=24.5) for row in common_rows],
            ["participant_id", "cohort", "research_stage", "array_index"],
        )

        self._write_parquet(
            "family_history/initial_medical.parquet",
            [
                dict(
                    row,
                    type_1_diabetes_family_number=0,
                    type_2_diabetes_family_number=1,
                )
                for row in common_rows
            ],
            ["participant_id", "cohort", "research_stage", "array_index"],
        )

        self._write_parquet(
            "medical_conditions/medical_conditions.parquet",
            [dict(row, medical_condition="none") for row in common_rows],
            ["participant_id", "cohort", "research_stage", "array_index"],
        )

        cgm_rows = [
            dict(
                row,
                connection_id="cgm-{}".format(row["participant_id"]),
                cgm_device_type="Libre Pro IQ",
            )
            for row in common_rows
        ]
        self._write_parquet(
            "cgm/cgm.parquet",
            cgm_rows,
            [
                "participant_id",
                "cohort",
                "research_stage",
                "array_index",
                "connection_id",
            ],
        )

    def test_manifest_contains_exactly_eight_sources(self):
        self.assertEqual(len(conversion.DATASET_SPECS), 8)
        relative_paths = {
            spec.source_relative_path for spec in conversion.DATASET_SPECS
        }
        self.assertIn("cgm/cgm.parquet", relative_paths)
        self.assertNotIn("diet_logging/diet_logging_events.parquet", relative_paths)
        self.assertNotIn("sex_specific_factors/ukbb.parquet", relative_paths)

    def test_copy_convert_and_restore_all_index_columns(self):
        messages = []
        summaries = conversion.prepare_all(
            self.source_root,
            self.project_root,
            emit=messages.append,
        )

        self.assertEqual(len(summaries), 8)
        self.assertTrue(any("CGM" in message for message in messages))

        for spec in conversion.DATASET_SPECS:
            source_path = self.source_root / spec.source_relative_path
            raw_path = self.project_root / "raw" / spec.source_relative_path
            csv_path = self.project_root / "csv" / spec.csv_filename

            self.assertTrue(raw_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertEqual(
                conversion.sha256_file(source_path),
                conversion.sha256_file(raw_path),
            )

            csv_frame = pd.read_csv(str(csv_path), low_memory=False)
            parquet_frame = pd.read_parquet(str(source_path))
            self.assertEqual(len(csv_frame), len(parquet_frame))

            for required_column in spec.required_columns:
                self.assertIn(required_column, csv_frame.columns)

        medication_csv = pd.read_csv(
            str(self.project_root / "csv" / "medications.csv"),
            dtype={"participant_id": str},
        )
        self.assertEqual(medication_csv.loc[0, "participant_id"], "1001")
        self.assertIn("A11", medication_csv.loc[0, "atc3"])

    def test_missing_source_fails_before_any_output_is_written(self):
        missing_path = self.source_root / "family_history/initial_medical.parquet"
        missing_path.unlink()

        with self.assertRaises(FileNotFoundError):
            conversion.prepare_all(self.source_root, self.project_root)

        self.assertFalse((self.project_root / "raw").exists())
        self.assertFalse((self.project_root / "csv").exists())

    def test_rerun_refreshes_managed_outputs(self):
        conversion.prepare_all(self.source_root, self.project_root)

        self._write_parquet(
            "population/population.parquet",
            [
                {
                    "participant_id": "2001",
                    "cohort": "10k",
                    "month_of_birth": 2,
                    "year_of_birth": 1990,
                    "sex": "Female",
                }
            ],
            ["participant_id", "cohort"],
        )

        conversion.prepare_all(self.source_root, self.project_root)

        population_csv = pd.read_csv(
            str(self.project_root / "csv" / "population.csv"),
            dtype={"participant_id": str},
        )
        self.assertEqual(len(population_csv), 1)
        self.assertEqual(population_csv.loc[0, "participant_id"], "2001")


if __name__ == "__main__":
    unittest.main()
