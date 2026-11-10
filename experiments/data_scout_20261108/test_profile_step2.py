"""Synthetic checks for discovery filtering, cohort identity and aggregate-only output."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import profile_step2 as reader


class ScoutChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project/collaborator_handoff_20261102_v1"
        self.key = pd.DataFrame({"participant_id": range(1001, 1013), "cohort": "10k",
                                 "research_stage": "00_00_visit"})
        x = self.key.assign(input_row=range(12), final_role="discovery", n_fiber_days=4,
                            n_energy_days=5, fiber_density=12.0, mean_daily_energy_fiber=2000.0)
        self.save(x, self.root / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet")
        path = self.root / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
        path.parent.mkdir(parents=True)
        y = np.full((2, 12), np.nan)
        y[0, :8], y[1, :7], y[1, 8] = 95.0, 95.0, 95.0
        np.savez(path, y=y)
        self.raw_path = self.root / "inputs/discovery_v2/D_Glu_raw_Y_TRE_ONLY.parquet"
        self.save(self.key.assign(Y_value=95.0, Y_unit="mg/dL"), self.raw_path)
        self.cov_path = self.root.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet"
        cov = self.key.copy()
        for name in ("activity_walking_10min_days_weekly_numeric", "activity_moderate_days_weekly_numeric",
                     "activity_vigorous_days_weekly_numeric", "sleep_hours_daily_numeric"):
            cov[name] = pd.array([1, None] + [3] * 10, dtype="Int64")
        cov["alcohol_current_frequency_label"] = ["__UNKNOWN__", None] + ["Never"] * 10
        self.save(pd.concat([cov, cov.assign(cohort="other", sleep_hours_daily_numeric=9999999)],
                            ignore_index=True).sample(frac=1, random_state=1), self.cov_path)
        common = self.key.iloc[:3].loc[self.key.index[:3].repeat(3)].copy()
        common["date"] = list(pd.date_range("2026-01-01", periods=3)) * 3
        self.common_path = self.root / "inputs/daily/common_days_no_scores_TRE_ONLY.parquet"
        self.save(common, self.common_path)
        dates = self.key.assign(food_first_date="2026-01-01", food_last_date="2026-01-04",
                                Y_first_date="2026-01-02", Y_last_date="2026-01-05",
                                O_sample_date="2026-01-01", G_sample_date="2026-01-07")
        self.dates_path = self.root.parent / "analysis/D_Glu/fiber_density/base_table_TRE_ONLY.parquet"
        dates.loc[1, "G_sample_date"] = None
        self.save(dates, self.dates_path)

    @staticmethod
    def save(frame, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

    def run_reader(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            reader.main(self.root)
        return stream.getvalue()

    def test_upstream_reads_are_filtered_projected_and_output_is_aggregate(self):
        original = pq.read_table
        seen = []

        def scoped(path, **kwargs):
            self.assertEqual(len(kwargs["filters"]), 3)
            self.assertNotIn("prediction", kwargs["columns"])
            frame = original(path, **kwargs)
            keys = frame.select(reader.KEY).to_pandas()
            self.assertTrue(keys.participant_id.isin(range(1001, 1009)).all())
            self.assertEqual(keys.cohort.unique().tolist(), ["10k"])
            self.assertEqual(keys.research_stage.unique().tolist(), ["00_00_visit"])
            seen.append(str(path))
            return frame

        with patch.object(reader.pq, "read_table", scoped):
            text = self.run_reader()
        self.assertEqual(len(seen), 4)
        self.assertIn("repeat0_N=8 repeat1_N=8 shared=7", text)
        self.assertIn("raw_Glu_mg_dL: finite=8/8", text)
        self.assertIn("prepared_common_people=3/8 days=9", text)
        self.assertIn("G_sample_date: known=7/8 before/within/after=0/0/7", text)
        self.assertIn("O_sample_date: known=8/8 before/within/after=8/0/0", text)
        self.assertNotIn("9999999", text)
        self.assertNotIn("1001", text)
        self.assertLessEqual(len(text.splitlines()), 25)

    def test_missing_optional_covariates_and_dates_remain_visible(self):
        self.save(self.key, self.cov_path)
        self.save(self.key, self.dates_path)
        text = self.run_reader()
        self.assertIn("sleep_hours_daily_numeric MISSING_COLUMN", text)
        self.assertIn("TIMING_MISSING_COLUMNS=", text)
        self.assertIn("window_envelope_overlap=0/0 known", text)
        self.assertIn("G_sample_date: known=0/8", text)

    def test_duplicate_person_source_does_not_silently_change_population(self):
        key = self.key.iloc[:1]
        self.save(pd.concat([key, key], ignore_index=True), self.cov_path)
        with self.assertRaises(pd.errors.MergeError):
            self.run_reader()

    def test_role_and_row_alignment_fail_before_upstream_values_are_read(self):
        path = self.root / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
        original = pq.read_table(path).to_pandas()
        for changed in (original.assign(final_role="confirmation"), original.iloc[::-1]):
            self.save(changed, path)
            with (
                patch.object(reader.pq, "read_table", side_effect=AssertionError("Scope escaped")),
                self.assertRaises(ValueError),
            ):
                self.run_reader()

    def test_glucose_units_are_checked(self):
        self.save(self.key.assign(Y_value=5.0, Y_unit="mmol/L"), self.raw_path)
        with self.assertRaises(ValueError):
            self.run_reader()

    def test_duplicate_prepared_days_are_rejected(self):
        common = pq.read_table(self.common_path).to_pandas()
        self.save(pd.concat([common, common.iloc[:1]], ignore_index=True), self.common_path)
        with self.assertRaises(ValueError):
            self.run_reader()


if __name__ == "__main__":
    unittest.main(verbosity=2)
