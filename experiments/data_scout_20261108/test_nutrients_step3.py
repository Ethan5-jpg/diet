"""Synthetic checks for version alignment, missing nutrients and daily consistency."""

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

import nutrients_step3 as reader


class NutrientChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project/collaborator_handoff_20261102_v1"
        self.keys = pd.DataFrame({"participant_id": range(90001, 90025), "cohort": "10k",
                                  "research_stage": "00_00_visit"})
        self.save(self.keys.assign(input_row=range(24), final_role="discovery"),
                  self.root / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet")
        path = self.root / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
        path.parent.mkdir(parents=True)
        y = np.full((2, 24), np.nan)
        y[:, :20] = 95
        np.savez(path, y=y)
        base, qc = [], []
        for person, key in enumerate(self.keys.to_dict("records")):
            for day in range(12 if person == 0 else 6):
                identity = dict(key, date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=day))
                base.append(dict(identity, energy_kcal=2000.0, fiber_g=20.0 + person,
                                 energy_eligible=True, fiber_eligible=True))
                values = {n + "_complete_sum": 100.0 + person for n in reader.NUTRIENTS}
                values.update(alcohol_g_complete_sum=0.0, dietary_fiber_g_complete_sum=20.0 + person,
                              calories_kcal_complete_sum=2000.0)
                qc.append(dict(identity, **values))
        self.base = pd.DataFrame(base)
        self.qc = pd.DataFrame(qc)
        self.base_path = self.root / "inputs/daily/food_days_discovery_TRE_ONLY.parquet"
        self.qc_path = self.root.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet"
        self.save(self.base, self.base_path)
        extras = self.qc.assign(cohort="other", carbohydrate_g_complete_sum=9999999.0)
        self.save(pd.concat([self.qc, extras], ignore_index=True).sample(frac=1, random_state=11),
                  self.qc_path)

    @staticmethod
    def save(frame, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

    def run_reader(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            reader.main(self.root)
        return stream.getvalue()

    def test_filtered_daily_sources_person_weighted_quantiles_and_halves(self):
        original = pq.read_table
        paths = []

        def scoped(path, **kwargs):
            self.assertEqual(len(kwargs["filters"]), 3)
            self.assertNotIn("Y_value", kwargs["columns"])
            self.assertNotIn("food_events", str(path))
            table = original(path, **kwargs)
            keys = table.select(reader.KEY).to_pandas()
            self.assertTrue(keys.participant_id.isin(range(90001, 90021)).all())
            self.assertEqual(keys.cohort.unique().tolist(), ["10k"])
            paths.append(str(path))
            return table

        with patch.object(reader.pq, "read_table", scoped):
            text = self.run_reader()
        self.assertEqual(len(paths), 2)
        self.assertIn("current_energy_days=126 people=20/20", text)
        self.assertIn("day_key_matches=126/126 energy_mismatch=0 fiber_mismatch=0", text)
        self.assertIn("carbohydrate_g: days=126/126 people_ge3=20/20 ge7=1", text)
        self.assertIn("p10/50/90=50.95/54.75/58.55; odd_even_rho=1.000 paired_people=20", text)
        self.assertIn("p10/50/90=0.00/0.00/0.00; odd_even_rho=NA paired_people=20", text)
        self.assertNotIn("90001", text)
        self.assertNotIn("9999999", text)
        self.assertLessEqual(len(text.splitlines()), 22)

    def test_missing_values_are_not_zero_and_missing_columns_are_visible(self):
        changed = self.qc.drop(columns=["protein_g_complete_sum"])
        changed["alcohol_g_complete_sum"] = np.nan
        self.save(changed, self.qc_path)
        text = self.run_reader()
        self.assertIn("protein_g MISSING_COMPLETE_SUM", text)
        self.assertIn("alcohol_g: days=0/126 people_ge3=0/20", text)
        self.assertIn("p10/50/90=NA; odd_even_rho=NA paired_people=0", text)
        self.assertIn("joint_CARB_FAT_PROTEIN MISSING_COMPLETE_SUM", text)

    def test_energy_or_fiber_version_mismatch_stops_nutrient_interpretation(self):
        for name in ("calories_kcal_complete_sum", "dietary_fiber_g_complete_sum"):
            changed = self.qc.copy()
            changed.loc[0, name] += 1
            self.save(changed, self.qc_path)
            text = self.run_reader()
            self.assertIn("ALIGNMENT_NOT_VERIFIED", text)
            self.assertNotIn("density_", text)

    def test_missing_day_match_stops_and_duplicate_day_keys_are_rejected(self):
        self.save(self.qc.iloc[1:], self.qc_path)
        self.assertIn("ALIGNMENT_NOT_VERIFIED", self.run_reader())
        self.save(pd.concat([self.qc, self.qc.iloc[:1]], ignore_index=True), self.qc_path)
        with self.assertRaises(ValueError):
            self.run_reader()

    def test_missing_daily_qc_and_incomplete_anchor_schema_are_reported(self):
        self.qc_path.unlink()
        self.assertIn("DAILY_NUTRIENT_QC_MISSING", self.run_reader())
        self.save(self.qc.drop(columns=["calories_kcal_complete_sum"]), self.qc_path)
        self.assertIn("DAILY_NUTRIENT_QC_MISSING_ANCHORS", self.run_reader())

    def test_invalid_eligibility_and_admitted_energy_are_rejected(self):
        changed = self.base.copy()
        changed.loc[0, "energy_kcal"] = -1
        self.save(changed, self.base_path)
        with self.assertRaises(ValueError):
            self.run_reader()
        changed = self.base.copy()
        changed.loc[0, "fiber_g"] = -1
        self.save(changed, self.base_path)
        with self.assertRaises(ValueError):
            self.run_reader()
