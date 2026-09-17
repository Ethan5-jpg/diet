#!/usr/bin/env python3
"""Synthetic regression and upload-layout tests; no cohort extracts are used."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import new_cgm_data as data
import new_cgm_models as models
import legacy_diet_models as legacy_diet
import legacy_microbiome_models as legacy_micro
import new_cgm_visuals as visuals


def zscore(values):
    series = pd.Series(values)
    return (series - series.mean()) / series.std(ddof=1)


def write_fixture(root, n=240, species_n=5):
    rng = np.random.RandomState(519)
    ids = ["%06d" % i for i in range(n)]
    cgm = pd.DataFrame({"participant_id": ids, "cohort": "10k", "research_stage": "00_00_visit",
                        "array_index": 0, "connection_id": ["conn%d" % i for i in range(n)],
                        "cgm_device_type": np.where(np.arange(n) % 2 == 0, "device_a", "device_b")})
    above = np.where(np.arange(n) % 4 == 0, rng.uniform(0, 8, n), 0)
    below = rng.uniform(0, 12, n)
    values = {"cgm_mage": 30 + rng.normal(size=n) * 5, "cgm_above_180": above,
              "cgm_below_70": below, "cgm_in_range_70_180": 100 - above - below}
    for metric, raw in values.items():
        clean = pd.Series(raw.copy())
        if metric == "cgm_mage":
            clean.iloc[0] = np.nan
        cgm[metric + "_raw"] = raw
        cgm[metric + "_clean"] = clean
        cgm[metric + "_z"] = zscore(clean)
    scores = pd.DataFrame({"participant_id": ids})
    for score in list(data.EXPOSURES)[:4]:
        scores[data.EXPOSURES[score][0]] = zscore(rng.normal(size=n))
    scores.loc[1, "AHEI_z"] = np.nan
    new = pd.DataFrame({"participant_id": ids, "mean_daily_energy_kcal": rng.normal(2100, 300, n)})
    for column in data.NEW_SOURCE_COLUMNS:
        new[column] = zscore(rng.normal(size=n))
    new.loc[2, "modified_eat_lancet13_z"] = np.nan
    cov = pd.DataFrame({"participant_id": ids})
    for column in data.CONTINUOUS + ["bmi", "alcohol_intake_g_day"]:
        cov[column] = rng.normal(10, 2, n)
    for column in data.BINARY:
        cov[column] = rng.randint(0, 2, n)
    cov["sex"] = np.where(np.arange(n) % 3 == 0, "male", "female")
    cov["education_level"] = np.where(np.arange(n) % 5 == 0, "low", "high")
    cov["smoking_status"] = np.array(["never", "former", "current"])[rng.randint(0, 3, n)]
    cov["cgm_device_type"] = cgm["cgm_device_type"]
    for column in ["known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]:
        cov[column] = 0.0
    cov.loc[3, ["known_diabetes", "exclude_diabetes_or_a10"]] = 1
    cov.loc[4:12, ["known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]] = np.nan
    cov.loc[13, "bmi"] = np.nan
    cov.loc[14, "alcohol_intake_g_day"] = np.nan
    for flag in data.FLAGS:
        cov[flag] = True
        if "model3" in flag:
            cov.loc[13, flag] = False
        if flag.startswith("hpdi"):
            cov.loc[14, flag] = False
    micro = pd.DataFrame({"participant_id": ids, "cohort": "10k", "research_stage": "00_00_visit", "array_index": 0})
    for i in range(species_n):
        micro["s__species_%02d" % i] = zscore(rng.normal(size=n))
    config = {"expected_species": species_n, "minimum_model_n": 30,
              "run_strict_sensitivity": True, "run_same_cohort_comparisons": True}
    for key, frame in [("cgm_csv", cgm), ("old_scores_csv", scores), ("new_scores_csv", new),
                        ("covariates_csv", cov), ("microbiome_csv", micro)]:
        path = root / (key + ".csv")
        frame.to_csv(path, index=False)
        config[key] = str(path)
    return config


class DataTests(unittest.TestCase):
    def test_five_sources_keep_outcome_specific_missing_and_unknown_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_fixture(Path(directory))
            diet, micro, species, audit, summary, _ = data.prepare_tables(config)
            self.assertEqual(len(diet), 240)
            self.assertEqual(len(species), 5)
            self.assertEqual(summary["diet_excluded_diabetes_a10_n"], 1)
            self.assertEqual(summary["diet_unknown_exclusion_n"], 9)
            self.assertEqual(summary["diet_primary_n"] - summary["diet_strict_n"], 9)
            self.assertEqual(audit.set_index("outcome").loc["cgm_mage", "z_n"], 239)
            self.assertEqual(diet["cgm_above_180_z"].notna().sum(), 240)
            self.assertEqual(diet.participant_id.iloc[0], "000000")

    def test_duplicate_ids_missing_columns_and_nonfinite_values_stop(self):
        for change in ("duplicate", "missing", "infinite", "flag_contradiction", "wrong_species"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                config = write_fixture(Path(directory))
                path = Path(config["covariates_csv"])
                cov = pd.read_csv(path, dtype={"participant_id": str})
                if change == "duplicate":
                    cov = pd.concat([cov, cov.iloc[[0]]])
                elif change == "missing":
                    cov = cov.drop(columns="bmi")
                elif change == "infinite":
                    cov.loc[0, "age_years"] = np.inf
                elif change == "flag_contradiction":
                    cov.loc[0, "a10_medication_use"] = 1
                else:
                    config["expected_species"] = 6
                cov.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    data.prepare_tables(config)

    def test_old_cgm_or_altered_z_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_fixture(Path(directory))
            path = Path(config["cgm_csv"])
            cgm = data.read_table(path)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                data.validate_cgm(cgm.drop(columns="cgm_above_180_z"))
            cgm["cgm_above_180_z"] = pd.to_numeric(cgm["cgm_above_180_z"]) + 2
            with self.assertRaisesRegex(ValueError, "standardized"):
                data.validate_cgm(cgm)

    def test_whitespace_and_decimal_ids_normalize_without_losing_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.csv"
            path.write_text("participant_id,x\n 000123.0 ,a\n000124,b\n")
            self.assertEqual(data.read_table(path).participant_id.tolist(), ["000123", "000124"])
            path.write_text("participant_id,x\n000123.0,a\n000123,b\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                data.read_table(path)

    def test_bmi_alcohol_and_energy_selection_are_exposure_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_fixture(Path(directory))
            diet = data.prepare_tables(config)[0]
            ahei2 = models.select_diet(diet, "AHEI", "TAR180", "primary", 2)
            ahei3 = models.select_diet(diet, "AHEI", "TAR180", "primary", 3)
            self.assertEqual(len(ahei2) - len(ahei3), 1)
            hpdi = models.select_diet(diet, "hPDI", "TAR180", "primary", 2)
            self.assertNotIn("000014", hpdi.participant_id.tolist())
            diet.loc[diet.participant_id.eq("000020"), "mean_daily_energy_kcal"] = np.nan
            self.assertIn("000020", models.select_diet(diet, "AHEI", "TAR180", "primary", 2).participant_id.tolist())
            self.assertNotIn("000020", models.select_diet(diet, "NOVA4", "TAR180", "primary", 2).participant_id.tolist())


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = write_fixture(Path(cls.temp.name))
        cls.diet, cls.micro, cls.species, _, _, _ = data.prepare_tables(cls.config)
        with redirect_stdout(io.StringIO()):
            cls.diet_results = models.fit_diet(cls.diet, cls.config)[0]
            cls.micro_results = models.fit_microbiome(cls.micro, cls.species, cls.config)[0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_visual_counts_exclude_failed_tests_and_keep_zero_distinct_from_na(self):
        diet = self.diet_results.copy()
        selected = diet.analysis_set.eq("primary") & diet.model.eq(2) & diet.outcome.eq("MAGE")
        indices = diet.index[selected].tolist()
        diet.loc[indices, ["FDR_family", "FDR_global"]] = 1.0
        diet.loc[indices[:3], "FDR_family"] = 0.001
        diet.loc[indices[0], "beta"] = 0.4
        diet.loc[indices[1], "beta"] = -0.4
        diet.loc[indices[2], "status"] = "failed"
        diet.loc[indices[3], "beta"] = 0.0
        d = visuals.prepare_results(diet, "diet")
        m = visuals.prepare_results(self.micro_results, "microbiome")
        counts = visuals.intuitive_counts(d, m)
        row = counts.loc[counts.branch.eq("diet") & counts.model.eq(2) & counts.outcome.eq("MAGE")].iloc[0]
        self.assertEqual((row.computed_tests, row.failed_tests, row.family_significant_positive,
                          row.family_significant_negative, row.family_significant_total), (6, 1, 1, 1, 2))
        beta, _, _ = visuals.heatmap_arrays(d.loc[d.model.eq(2)], "exposure", list(data.EXPOSURES))
        self.assertTrue(np.isnan(beta[2, 0]))
        self.assertEqual(beta[3, 0], 0.0)

    def test_visual_species_selection_marks_exploration_and_retains_full_names(self):
        m = visuals.prepare_results(self.micro_results, "microbiome")
        m["FDR_family"] = 1.0
        chosen, mode = visuals.choose_species(m, 2)
        self.assertEqual(mode, "exploratory_no_significant_species")
        self.assertEqual(set(chosen.species), set(self.species))
        species = self.species[0]
        m.loc[m.species.eq(species) & m.model.eq(2), "FDR_family"] = 0.01
        chosen, mode = visuals.choose_species(m, 2)
        self.assertEqual(chosen.species.tolist(), [species])
        self.assertEqual(mode, "significant_in_at_least_one_outcome")

    def test_visual_duplicate_and_invalid_completed_statistics_are_rejected(self):
        duplicate = pd.concat([self.diet_results, self.diet_results.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            visuals.prepare_results(duplicate, "diet")
        invalid = self.diet_results.copy()
        invalid.loc[invalid.analysis_set.eq("primary"), "FDR_family"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            visuals.prepare_results(invalid, "diet")

    def test_all_seven_diets_four_outcomes_and_scenarios(self):
        self.assertEqual(len(self.diet_results), 7 * 4 * 7)
        self.assertEqual(set(self.diet_results.outcome), set(data.OUTCOMES))
        self.assertEqual(set(self.diet_results.exposure), set(data.EXPOSURES))
        self.assertTrue(self.diet_results.status.eq("computed").all())
        self.assertEqual(len(self.micro_results), 5 * 4 * 7)
        self.assertTrue(self.micro_results.status.eq("computed").all())

    def test_family_sizes_are_16_8_4_and_global28(self):
        result = self.diet_results
        for family, size in [("original_four", 16), ("new_primary", 8), ("exploratory", 4)]:
            self.assertEqual(set(result.loc[result.family.eq(family), "family_tests_planned"]), {size})
        self.assertEqual(set(result.global_tests_planned), {28})
        self.assertEqual(set(self.micro_results.family_tests_planned), {5})
        self.assertEqual(set(self.micro_results.global_tests_planned), {20})

    def test_same_cohort_comparisons_have_exact_sample_identity(self):
        for result, branch in [(self.diet_results, "diet"), (self.micro_results, "microbiome")]:
            table = models.comparisons(result, branch)
            self.assertTrue(table.sample_sha256_same_sample.eq(table.sample_sha256_adjusted).all())
            self.assertTrue(table.N_same_sample.eq(table.N_adjusted).all())

    def test_mage_missing_does_not_reduce_tar_sample_and_no_z_recalculation(self):
        a = self.diet_results.query("analysis_set == 'primary' and model == 0 and exposure == 'NOVA4'").set_index("outcome")
        self.assertEqual(a.loc["TAR180", "N"] - a.loc["MAGE", "N"], 1)
        source = data.read_table(self.config["cgm_csv"]).set_index("participant_id")
        merged = self.diet.set_index("participant_id")
        np.testing.assert_allclose(merged.cgm_above_180_z, pd.to_numeric(source.cgm_above_180_z), rtol=0, atol=0)

    def test_new_scores_add_energy_hpdi_adds_alcohol(self):
        selected = models.select_diet(self.diet, "NOVA4", "TAR180", "primary", 3)
        _, names, _, _ = models.diet_design(selected, "NOVA4", 3)
        self.assertIn("mean_daily_energy_kcal_z", names)
        self.assertIn("bmi_z", names)
        self.assertNotIn("alcohol_intake_g_day_z", names)
        selected = models.select_diet(self.diet, "hPDI", "TAR180", "primary", 2)
        _, names, _, _ = models.diet_design(selected, "hPDI", 2)
        self.assertIn("alcohol_intake_g_day_z", names)
        self.assertNotIn("mean_daily_energy_kcal_z", names)

    def test_ols_agrees_with_independent_lstsq(self):
        selected = models.select_diet(self.diet, "AMED", "TAR180", "primary", 3)
        X, _, _, _ = models.diet_design(selected, "AMED", 3)
        y = selected.cgm_above_180_z.to_numpy()
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta
        mse = np.dot(residual, residual) / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * mse)
        result = self.diet_results.query("analysis_set == 'primary' and model == 3 and exposure == 'AMED' and outcome == 'TAR180'").iloc[0]
        self.assertAlmostEqual(result.beta, beta[1], places=10)
        self.assertAlmostEqual(result.SE, se[1], places=10)

    def test_microbiome_vectorized_matches_direct_ols(self):
        selected = models.select_micro(self.micro, "TBR70", "primary", 3)
        C = legacy_micro.build_covariate_design(selected, True)[0]
        X = np.column_stack([C, selected[self.species[0]].to_numpy(float)])
        fit = legacy_diet.fit_ols(X, selected.cgm_below_70_z.to_numpy(float))
        row = self.micro_results.query("analysis_set == 'primary' and model == 3 and outcome == 'TBR70'").iloc[0]
        self.assertAlmostEqual(row.beta, fit["coef"][-1], places=10)
        self.assertAlmostEqual(row.SE, fit["se"][-1], places=10)

    def test_failed_tests_do_not_shrink_bh_family(self):
        result = self.diet_results.query("analysis_set == 'primary' and model == 0").copy()
        i = result.index[0]
        result.loc[i, ["p_value", "status"]] = [np.nan, "failed"]
        corrected = models.correct_fdr(result, "diet")
        group = corrected.loc[corrected.family.eq("original_four")]
        expected = legacy_diet.bh_fdr(group.p_value.fillna(1).to_numpy())
        self.assertTrue(pd.isna(corrected.loc[i, "FDR_family"]))
        np.testing.assert_allclose(group.FDR_family.iloc[1:], expected[1:])
        self.assertEqual(set(group.family_tests_planned), {16})

    def test_constant_species_failure_is_local_and_reported(self):
        micro = self.micro.copy()
        micro[self.species[0]] = 0
        config = dict(self.config, run_strict_sensitivity=False, run_same_cohort_comparisons=False)
        with redirect_stdout(io.StringIO()):
            result = models.fit_microbiome(micro, self.species, config)[0]
        self.assertTrue(result.loc[result.species.eq(self.species[0]), "status"].eq("failed").all())
        self.assertTrue(result.loc[~result.species.eq(self.species[0]), "status"].eq("computed").all())

    def test_full_379_species_scale_and_1516_test_families(self):
        rng = np.random.RandomState(208)
        species = ["s__full_%03d" % i for i in range(379)]
        features = pd.DataFrame(rng.normal(size=(len(self.micro), 379)), columns=species, index=self.micro.index)
        frame = pd.concat([self.micro.drop(columns=self.species), features], axis=1)
        config = dict(self.config, expected_species=379, run_strict_sensitivity=False, run_same_cohort_comparisons=False)
        with redirect_stdout(io.StringIO()):
            result = models.fit_microbiome(frame, species, config)[0]
        self.assertEqual(len(result), 379 * 4 * 3)
        self.assertTrue(result.status.eq("computed").all())
        self.assertEqual(set(result.family_tests_planned), {379})
        self.assertEqual(set(result.global_tests_planned), {1516})


class ReuseAndCLITests(unittest.TestCase):
    def test_extracted_legacy_functions_are_unchanged(self):
        import hashlib
        provenance = json.loads((ROOT / "docs/legacy_provenance.json").read_text())
        for entry in provenance:
            target = ROOT / entry["target"]
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), entry["target_sha256"])
            source = ROOT.parent / entry["source_relative_to_parent"]
            if source.exists():
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry["source_sha256"])

    def test_full_upload_folder_cli_check_run_and_repeat(self):
        with tempfile.TemporaryDirectory(prefix="cgm4 isolated ") as directory:
            root = Path(directory)
            project = root / "HPP/Data/diet_microbiome_glucose_analysis/4New_cgm"
            project.mkdir(parents=True)
            shutil.copytree(ROOT / "scripts", project / "scripts")
            shutil.copy2(ROOT / "run_analysis.sh", project / "run_analysis.sh")
            fixture = root / "input"
            fixture.mkdir()
            config = write_fixture(fixture)
            config["run_strict_sensitivity"] = False
            config["run_same_cohort_comparisons"] = False
            (project / "config").mkdir()
            (project / "config/analysis.json").write_text(json.dumps(config))
            before = [data.fingerprint(config[k]) for k in config if k.endswith("_csv")]
            env = os.environ.copy()
            env["PYTHON_BIN"] = sys.executable
            for extra, expected in [(["--check-only"], "inputs_checked_no_models_fitted"), ([], "completed")]:
                result = subprocess.run(["bash", str(project / "run_analysis.sh")] + extra, cwd=directory,
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("STATUS=" + expected, result.stdout)
            manifests = list((project / "outputs").glob("*/reports/manifest.json"))
            self.assertEqual(len(manifests), 2)
            final_manifest = [p for p in manifests if json.loads(p.read_text())["status"] == "completed"][0]
            out = final_manifest.parent.parent
            result = pd.read_csv(out / "models/diet_cgm_all_models.csv")
            self.assertEqual(len(result), 84)
            self.assertEqual(set(result.outcome), set(data.OUTCOMES))
            self.assertTrue((out / "reports/结果摘要.md").exists())
            record = json.loads(final_manifest.read_text())["visualization"]
            self.assertEqual(record["figure_count"], 7)
            figures = Path(record["figure_dir"])
            self.assertEqual(len(list(figures.iterdir())), 21)
            page = Path(record["report"]).read_text()
            self.assertIn("width:100%", page)
            self.assertIn("直观统计.csv", page)
            import re
            for target in re.findall(r'(?:href|src)="([^"]+)"', page):
                self.assertTrue((Path(record["report"]).parent / target).exists(), target)
            snapshots = [data.fingerprint(p) for p in (out / "models").glob("*.csv")]
            replot = subprocess.run([sys.executable, str(project / "scripts/new_cgm_visuals.py"),
                                     "--run-dir", str(out)], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(replot.returncode, 0, replot.stdout + replot.stderr)
            self.assertEqual(len(list((out / "visuals").iterdir())), 2)
            self.assertEqual(snapshots, [data.fingerprint(p) for p in (out / "models").glob("*.csv")])
            self.assertEqual(before, [data.fingerprint(config[k]) for k in config if k.endswith("_csv")])
            self.assertFalse((project.parent / "outputs").exists())

    def test_missing_input_exit_is_nonzero_and_has_diagnostic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "4New_cgm"
            shutil.copytree(ROOT / "scripts", project / "scripts")
            (project / "config").mkdir()
            config = json.loads((ROOT / "config/analysis.json").read_text())
            config["cgm_csv"] = str(Path(directory) / "missing.csv")
            (project / "config/analysis.json").write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(project / "scripts/run_analysis.py"), "--check-only"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("STATUS=failed", result.stdout)
            manifest = next((project / "outputs").glob("*/reports/manifest.json"))
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")
            self.assertFalse(list((project / "outputs").glob("*/models/*.csv")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
