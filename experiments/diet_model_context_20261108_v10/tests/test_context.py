"""Source scope, nutrient QC, aggregation definitions and missingness semantics."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from context import ALCOHOL, FOOD, KEY, LIFE, aggregate, projected_read, read_context
from support import fixture, tables


class ContextChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = fixture()

    def test_density_means_ratios_and_lifestyle_are_key_aligned(self):
        index = self.data.index.iloc[:4].copy()
        food, nutrients, life = tables(index)
        expected = aggregate(food, nutrients, life, index)
        actual = aggregate(
            food.sample(frac=1, random_state=2),
            nutrients.sample(frac=1, random_state=3),
            life.sample(frac=1, random_state=4),
            index,
        )
        for name in expected["blocks"]:
            np.testing.assert_allclose(expected["blocks"][name], actual["blocks"][name])
        nutrients.loc[:5, "carbohydrate_g_complete_sum"] = 0
        nutrients.loc[6:11, "protein_g_complete_sum"] = np.nan
        value = aggregate(food, nutrients, life, index)
        self.assertEqual(value["blocks"]["macro"][0, 0], 0)
        self.assertTrue(np.isnan(value["blocks"]["macro"][1, 2]))
        self.assertEqual(len(value["blocks"]["full"]), len(index))
        before = copy.deepcopy(index)
        nutrients.loc[12:17, "lipid_g_complete_sum"] = np.nan
        aggregate(food, nutrients, life, index)
        pd.testing.assert_frame_equal(index, before)

    def test_joint_and_fiber_day_means_have_distinct_definitions(self):
        index = self.data.index.iloc[:1].copy()
        food, nutrients, life = tables(index)
        nutrients.loc[0, "lipid_g_complete_sum"] = np.nan
        food.loc[:1, "fiber_eligible"] = False
        index["n_fiber_days"] = 4
        index["fiber_density"] = (1000 * food.fiber_g / food.energy_kcal).iloc[2:].mean()
        value = aggregate(food, nutrients, life, index)["blocks"]
        density = 1000 * nutrients.carbohydrate_g_complete_sum / food.energy_kcal
        self.assertAlmostEqual(value["macro"][0, 0], density.mean())
        self.assertAlmostEqual(value["joint"][0, 0], density.iloc[1:].mean())
        self.assertAlmostEqual(value["fiber_days"][0, 0], density.iloc[2:].mean())
        self.assertNotEqual(value["macro"][0, 0], value["joint"][0, 0])

    def test_incomplete_three_day_features_remain_missing_without_dropping_person(self):
        index = self.data.index.iloc[:2].copy()
        food, nutrients, life = tables(index)
        nutrients.loc[:3, "carbohydrate_g_complete_sum"] = np.nan
        life = life.iloc[:0]
        value = aggregate(food, nutrients, life, index)
        self.assertTrue(np.isnan(value["blocks"]["macro"][0, 0]))
        self.assertTrue(np.isnan(value["blocks"]["life"]).all())
        self.assertTrue(all(v is None for v in value["alcohol"]))
        self.assertEqual(value["audit"]["discovery_people"], 2)

    def test_day_anchor_flags_duplicates_and_scope_conflicts_are_rejected(self):
        index = self.data.index.iloc[:2].copy()
        original = tables(index)
        mutations = [
            (0, "energy_eligible", "False"),
            (0, "energy_kcal", 100),
            (1, "calories_kcal_complete_sum", 999),
            (1, "dietary_fiber_g_complete_sum", 999),
            (2, LIFE[0], 8),
            (0, "research_stage", "other_visit"),
        ]
        for table, field, value in mutations:
            with self.subTest(table=table, field=field):
                frames = [v.copy() for v in original]
                frames[table][field] = frames[table][field].astype(object)
                frames[table].loc[0, field] = value
                with self.assertRaises(ValueError):
                    aggregate(*frames, index)
        for j in range(3):
            frames = [v.copy() for v in original]
            frames[j] = pd.concat([frames[j], frames[j].iloc[:1]], ignore_index=True)
            with self.assertRaises(ValueError):
                aggregate(*frames, index)

    def test_projected_reader_filters_composite_scope_before_materializing(self):
        index = self.data.index.iloc[:2][KEY].copy()
        food, _, _ = tables(self.data.index.iloc[:2])
        other_visit = food.copy()
        other_visit["research_stage"] = "other_visit"
        other_cohort = food.copy()
        other_cohort["cohort"] = "confirmation"
        frame = pd.concat([food, other_visit, other_cohort], ignore_index=True)
        frame["Y_value"] = -99999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.parquet"
            frame.to_parquet(path)
            original = pq.read_table
            with patch("context.pq.read_table", wraps=original) as reader:
                actual = projected_read(path, FOOD, index)
                self.assertIsNotNone(reader.call_args.kwargs["filters"])
                self.assertNotIn("Y_value", reader.call_args.kwargs["columns"])
                self.assertEqual(list(actual.columns), FOOD)
                self.assertEqual(len(actual), len(food))

    def test_context_loader_preserves_full_input_row_mapping_and_eligible_scope(self):
        index = self.data.index.iloc[:5].copy()
        food, nutrients, life = tables(index)
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "project/handoff"
            (package / "inputs/daily").mkdir(parents=True)
            (package.parent / "analysis/shared").mkdir(parents=True)
            food.to_parquet(package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet")
            nutrients.to_parquet(package.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet")
            life.to_parquet(package.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet")
            mask = np.array([True, False, True, True, False])
            extra, paths = read_context(package, index, mask)
            expected = aggregate(
                food.loc[food.participant_id.isin(index.participant_id[mask])],
                nutrients.loc[nutrients.participant_id.isin(index.participant_id[mask])],
                life.loc[life.participant_id.isin(index.participant_id[mask])],
                index.loc[mask],
            )
            self.assertEqual(len(paths), 3)
            self.assertEqual(extra["audit"]["discovery_people"], 3)
            for name, values in extra["blocks"].items():
                np.testing.assert_allclose(values[mask], expected["blocks"][name])
                self.assertTrue(np.isnan(values[~mask]).all())
            self.assertTrue(all(v is None for v in extra["alcohol"][~mask]))

    def test_log_ratios_are_invariant_to_shared_macro_scale_away_from_floor(self):
        index = self.data.index.iloc[:2].copy()
        food, nutrients, life = tables(index)
        a = aggregate(food, nutrients, life, index)["blocks"]["ratios"]
        for name in ("carbohydrate_g", "lipid_g", "protein_g"):
            nutrients[name + "_complete_sum"] *= 5
        b = aggregate(food, nutrients, life, index)["blocks"]["ratios"]
        np.testing.assert_allclose(a[:, :2], b[:, :2], atol=1e-14)

    def test_missing_required_columns_fail_before_reading_source_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.parquet"
            pd.DataFrame(dict(unrelated=[1])).to_parquet(path)
            with patch("context.pq.read_table", side_effect=AssertionError("Unexpected value read")):
                with self.assertRaises(ValueError):
                    projected_read(path, KEY + [ALCOHOL], self.data.index.iloc[:2])

    def test_primary_exposure_and_day_count_drift_are_rejected(self):
        index = self.data.index.iloc[:2].copy()
        frames = tables(index)
        for column in ("fiber_density", "n_fiber_days"):
            changed = index.copy()
            changed[column] += 1
            with self.assertRaises(ValueError):
                aggregate(*frames, changed)

    def test_combined_context_retains_sodium_and_sensitivities_hold_other_inputs_fixed(self):
        blocks = self.data.extra["blocks"]
        np.testing.assert_array_equal(blocks["full"][:, :4], blocks["sodium"])
        np.testing.assert_array_equal(blocks["full"][:, 4:], blocks["life"])
        np.testing.assert_array_equal(blocks["macro_life"][:, :3], blocks["macro"])
        for name in ("joint", "fiber_days"):
            np.testing.assert_array_equal(blocks[name][:, 3:], blocks["full"][:, 3:])

    def test_nullable_numeric_and_boolean_columns_preserve_missingness_and_population(self):
        index = self.data.index.iloc[:3].copy()
        food, nutrients, life = tables(index)
        nutrients.loc[1, "carbohydrate_g_complete_sum"] = np.nan
        for name in ("energy_eligible", "fiber_eligible"):
            food[name] = food[name].astype(object)
            food.loc[0, name] = np.nan
        food.loc[0, ["energy_kcal", "fiber_g"]] = np.nan
        index.loc[index.index[0], "n_fiber_days"] = 5
        index.loc[index.index[0], "fiber_density"] += 0.5
        expected = aggregate(food, nutrients, life, index)
        for name in ("energy_kcal", "fiber_g"):
            food[name] = food[name].astype("Float64")
        for name in ("energy_eligible", "fiber_eligible"):
            food[name] = food[name].astype("boolean")
        for name in nutrients:
            if name.endswith("_sum"):
                nutrients[name] = nutrients[name].astype("Float64")
        for name in LIFE:
            life[name] = life[name].astype("Float64")
        for flag_dtype in ("boolean", "Int64"):
            for name in ("energy_eligible", "fiber_eligible"):
                food[name] = food[name].astype(flag_dtype)
            actual = aggregate(food, nutrients, life, index)
            self.assertEqual(actual["audit"], expected["audit"])
            for name in expected["blocks"]:
                np.testing.assert_allclose(actual["blocks"][name], expected["blocks"][name])

    def test_categorical_identity_keys_ignore_unobserved_categories(self):
        index = self.data.index.iloc[:3].copy()
        frames = tables(index)
        expected = aggregate(*frames, index)
        for frame in [index] + list(frames):
            frame["cohort"] = pd.Categorical(frame.cohort, categories=["10k", "unobserved_cohort"])
            frame["research_stage"] = pd.Categorical(
                frame.research_stage, categories=["00_00_visit", "unobserved_visit"]
            )
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            actual = aggregate(*frames, index)
        self.assertFalse([w for w in caught if issubclass(w.category, FutureWarning)])
        self.assertEqual(actual["audit"], expected["audit"])
