"""Exercise the actual Parquet/NPZ reader and frozen source dependencies on fake files."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from support import fixture

import data


class AdapterChecks(unittest.TestCase):
    def test_real_reader_and_role_unit_alignment_guards(self):
        original = Path(__file__).resolve().parents[3]
        sample, _ = fixture()
        with tempfile.TemporaryDirectory() as name:
            package = Path(name).resolve() / "collaborator_handoff_20261102_v1"
            pilot = (
                original
                / "experiments/diet_response_heterogeneity_20261103_v1_attempt02/code"
            )
            if not pilot.exists():
                pilot = original / "experiments/diet_response_heterogeneity_20261103_v1"
            prep = original / "preprocessing/code/preprocessing_v1_20261030.py"
            if not prep.exists():
                prep = (
                    original
                    / "02_ALL_LOCAL_SOURCE/scripts/preprocessing_v1_20261030.py"
                )
            sources = {
                "experiments/diet_nested_stability_20261103_v4/run_nested.py": original
                / "experiments/diet_nested_stability_20261103_v4/run_nested.py",
                "experiments/diet_representation_screen_20261103_v3/run_screen.py": original
                / "experiments/diet_representation_screen_20261103_v3/run_screen.py",
                "experiments/diet_response_heterogeneity_20261103_v1_attempt02/code/run_pilot.py": pilot
                / "run_pilot.py",
                "experiments/diet_response_heterogeneity_20261103_v1_attempt02/code/core.py": pilot
                / "core.py",
                "preprocessing/code/preprocessing_v1_20261030.py": prep,
            }
            for relative, source in sources.items():
                target = package / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            n, rng = 4427, np.random.default_rng(551)
            x = pd.DataFrame(
                dict(
                    participant_id=np.arange(n) + 100000,
                    input_row=np.arange(n),
                    final_role="discovery",
                    split_unit_id=np.arange(n) // 2,
                    cohort="synthetic",
                    research_stage="test",
                    n_fiber_days=4,
                    fiber_density=rng.uniform(5, 25, n),
                    age_calendar_years=rng.uniform(30, 70, n),
                    sex_code=rng.integers(0, 2, n),
                    mean_daily_energy_fiber=rng.uniform(1500, 2500, n),
                    smoking_status="never",
                )
            )
            groups = x.split_unit_id.astype(str).to_numpy()
            x["cv_fold"] = sample.api.folds(groups, 5, 3)
            y = 100 + 5 * np.sin(np.arange(n) * 0.01)
            outcomes = x[
                ["participant_id", "input_row", "cohort", "research_stage"]
            ].copy()
            outcomes["Y_value"], outcomes["Y_unit"] = y, "mg/dL"
            inputs = package / "inputs/discovery_v2"
            inputs.mkdir(parents=True)
            cov_path = inputs / "discovery_covariates_food_TRE_ONLY.parquet"
            y_path = inputs / "D_Glu_raw_Y_TRE_ONLY.parquet"

            def write(frame, path):
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

            write(x, cov_path)
            write(outcomes, y_path)
            for site in ("oral", "gut"):
                np.savez_compressed(
                    inputs / (site + "_raw_TRE_ONLY.npz"),
                    abundance=rng.dirichlet(np.ones(12), n),
                )
                (inputs / (site + "_taxa.json")).write_text(
                    json.dumps([str(i) for i in range(12)])
                )
            reference = (
                package
                / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY"
            )
            reference.mkdir(parents=True)
            outer = np.stack(
                [x.cv_fold.to_numpy(), sample.api.folds(groups, 5, 20261104)]
            )
            np.savez_compressed(
                reference / "final_predictions.npz",
                y=np.stack([y, y]),
                outer=outer,
                hist_boost_oral=np.stack([y + 0.2, y + 0.2]),
                hist_boost_oral_gut=np.stack([y + 0.3, y + 0.3]),
            )
            loaded = data.load(package)
            self.assertEqual(len(loaded.index), n)
            np.testing.assert_array_equal(loaded.y, y)
            split = loaded.split(outer[0] != 0, outer[0] == 0, [10.0, 15.0])
            self.assertGreater(len(split.rows), 30)
            bad = x.copy()
            bad.loc[0, "final_role"] = "confirmation"
            write(bad, cov_path)
            read, seen = loaded.api.read_table, []

            def observe(path):
                seen.append(path.name)
                return read(path)

            with (
                mock.patch("data.module", return_value=loaded.api),
                mock.patch.object(loaded.api, "read_table", side_effect=observe),
            ):
                with self.assertRaisesRegex(ValueError, "discovery-only"):
                    data.load(package)
            self.assertNotIn(y_path.name, seen)
            write(x, cov_path)
            bad = outcomes.copy()
            bad["Y_unit"] = "mmol/L"
            write(bad, y_path)
            with self.assertRaisesRegex(ValueError, "Unexpected outcome unit"):
                data.load(package)
            bad = outcomes.copy()
            bad.loc[0, "participant_id"] = 9999999
            write(bad, y_path)
            with self.assertRaisesRegex(ValueError, "Outcome alignment") as failure:
                data.load(package)
            self.assertNotIn("9999999", str(failure.exception))
