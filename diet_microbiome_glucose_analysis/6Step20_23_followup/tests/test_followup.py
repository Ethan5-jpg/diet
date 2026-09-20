import ast
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import pandas as pd
from common import Context, AuditError, bools, norm_id, sample_hash
from cohort import load_population, audit_lineage
from analyses import pairwise
from run_followup import PACKAGE, parser, run, load_frozen
from fixtures import fixture


class PrimitiveTests(unittest.TestCase):
    def test_booleans_false_not_truthy_string(self):
        self.assertEqual(bools(pd.Series(["False", "0", "TRUE", 1])).tolist(), [False, False, True, True])

    def test_unknown_boolean_rejected(self):
        with self.assertRaises(AuditError): bools(pd.Series([np.nan]))

    def test_empty_ids_rejected(self):
        with self.assertRaises(AuditError): norm_id(pd.Series([" "]))

    def test_identity_leading_zero_and_decimal(self):
        self.assertEqual(norm_id(pd.Series(["001", "12.0"])).tolist(), ["001", "12"])

    def test_hash_invariant_to_order(self):
        self.assertEqual(sample_hash(pd.Series(["a", "b"])), sample_hash(pd.Series(["b", "a"])))

    def test_pairwise_ranks_after_missing_filter(self):
        frame = pd.DataFrame(dict(a=[1, 2, 3, 100, 101], b=[1, np.nan, 2, 3, np.nan], c=[5] * 5))
        r, p, n, pairs = pairwise(frame, 3)
        self.assertAlmostEqual(r.loc["a", "b"], 1.)
        self.assertEqual(n.loc["a", "b"], 3)
        self.assertTrue(np.isnan(r.loc["a", "c"]))
        self.assertLess(p.loc["a", "b"], 1.)

    def test_ties_and_nonfinite_values(self):
        f = pd.DataFrame(dict(a=[1, 1, 2, np.inf], b=[1, 2, 3, 4]))
        r, _, n, _ = pairwise(f, 3)
        self.assertAlmostEqual(r.loc["a", "b"], np.sqrt(3) / 2)
        self.assertEqual(n.loc["a", "b"], 3)
        self.assertTrue(np.isnan(pairwise(f, 4)[0].loc["a", "b"]))

    def test_python38_syntax(self):
        for p in (PACKAGE / "scripts").glob("*.py"):
            ast.parse(p.read_text(), feature_version=(3, 8))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "Data"
        self.plan, self.folder, self.cgmpath, self.models = fixture(self.root)
        self.args = parser().parse_args(["--data-root", str(self.root), "--step19b-run", str(self.folder), "--output-parent", str(Path(self.tmp.name) / "runs")])

    def tearDown(self):
        self.tmp.cleanup()

    def context(self):
        out = Path(self.tmp.name) / "audit"
        (out / "tables").mkdir(parents=True, exist_ok=True)
        ctx = Context(self.root, out, PACKAGE)
        ctx.synthetic = True
        return ctx

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code, out = run(self.args, self.plan)
        return code, out, json.loads((out / "manifest.json").read_text())

    def test_end_to_end_figures_and_no_participant_export(self):
        code, out, manifest = self.execute()
        self.assertEqual(code, 0, (out / "SCREENSHOT_SUMMARY.txt").read_text())
        self.assertEqual(manifest["data_kind"], "SYNTHETIC_TEST_ONLY")
        self.assertEqual(len(list((out / "figures").glob("*.png"))), 5)
        audit = pd.read_csv(out / "tables/21_path_sample_reconstruction.csv")
        self.assertEqual(len(audit), 66)
        self.assertTrue(audit.sample_match.all())
        contexts = pd.read_csv(out / "tables/21_sequential_common_cohort_flow.csv")
        self.assertEqual(contexts.iloc[0].N, 159)
        self.assertEqual(contexts[contexts.cohort.eq("strict")].iloc[0].N, 158)
        with zipfile.ZipFile(out / "aggregate_results.zip") as z:
            self.assertIn("manifest.json", z.namelist())
            for name in z.namelist():
                if name.endswith(".csv"):
                    self.assertNotIn(b"SYNTHETIC_0001", z.read(name))
        if os.environ.get("HPP_FOLLOWUP_KEEP_QA"):
            dest = Path(os.environ["HPP_FOLLOWUP_KEEP_QA"])
            dest.mkdir(parents=True, exist_ok=True)
            for p in (out / "figures").iterdir(): shutil.copy2(p, dest / p.name)

    def test_changed_source_rejected_before_outputs(self):
        source = self.root / "diet_microbiome_glucose_analysis/4New_cgm/outputs/synthetic_primary/models/model.csv"
        source.write_text(source.read_text() + "\n")
        code, _, manifest = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(manifest["stages"][0]["status"], "FAIL")

    def test_production_rejects_synthetic_manifest(self):
        ctx = self.context(); ctx.synthetic = False
        with self.assertRaisesRegex(AuditError, "Synthetic"):
            load_frozen(ctx, self.folder, self.plan)

    def test_missing_medication_is_not_imputed(self):
        ctx = self.context()
        cg, masks, detail = load_population(ctx, pd.read_csv(self.cgmpath), self.models)
        audit = pd.read_csv(ctx.out / "tables/21_path_sample_reconstruction.csv")
        n = audit.groupby("model").complete_case_N.max()
        self.assertEqual(n.primary - n.protocol, 1)
        self.assertEqual(n.primary - n.strict, 1)

    def test_sample_hash_mismatch_rejected(self):
        self.models["primary"].loc[0, "sample_sha256"] = "bad"
        with self.assertRaisesRegex(AuditError, "sample hash mismatch"):
            load_population(self.context(), pd.read_csv(self.cgmpath), self.models)

    def test_duplicate_participant_rejected(self):
        cg = pd.read_csv(self.cgmpath)
        cg.loc[1, "participant_id"] = cg.loc[0, "participant_id"]
        with self.assertRaisesRegex(AuditError, "duplicate"):
            load_population(self.context(), cg, self.models)

    def test_wrong_sdw_alias_detected(self):
        cg = pd.read_csv(self.cgmpath)
        cg["cgm_sdw_raw"] = cg.cgm_sdb_raw
        report = audit_lineage(self.context(), cg, ["cgm_sdw"], self.root / "Transfer/cgm/iglu.csv")
        self.assertEqual(report.iloc[0].raw_lineage_status, "FAIL")

    def test_z_sign_reversal_detected(self):
        cg = pd.read_csv(self.cgmpath); cg["cgm_cogi_z"] *= -1
        report = audit_lineage(self.context(), cg, ["cgm_cogi"], self.root / "Transfer/cgm/iglu.csv")
        self.assertEqual(report.iloc[0].clean_to_z_status, "FAIL")

    def test_missing_source_is_pending_not_assumed(self):
        cg = pd.read_csv(self.cgmpath).drop(columns=["cgm_sdw_raw", "cgm_sdw_clean"])
        report = audit_lineage(self.context(), cg, ["cgm_sdw"], self.root / "Transfer/cgm/iglu.csv")
        self.assertEqual(report.iloc[0].raw_lineage_status, "PENDING_COLUMN")

    def test_covariate_drift_rejected(self):
        self.models["protocol"].loc[0, "covariates"] = "age_years"
        with self.assertRaisesRegex(AuditError, "covariate list differs"):
            load_population(self.context(), pd.read_csv(self.cgmpath), self.models)

    def test_lineage_failure_skips_correlations(self):
        cg = pd.read_csv(self.cgmpath)
        cg["cgm_sdw_raw"] = cg.cgm_sdb_raw
        cg.to_csv(self.cgmpath, index=False)
        code, out, manifest = self.execute()
        self.assertEqual(code, 2)
        stages = {s["stage"]: s["status"] for s in manifest["stages"]}
        self.assertEqual(stages["20_field_lineage"], "REVIEW_REQUIRED")
        self.assertEqual(stages["20_correlations"], "SKIPPED_DEPENDENCY")
        self.assertFalse((out / "tables/20_primary_spearman.csv").exists())

    def test_repeated_read_cannot_reset_fingerprint(self):
        ctx = self.context()
        ctx.record(self.cgmpath, "first")
        self.cgmpath.write_text(self.cgmpath.read_text() + "\n")
        with self.assertRaisesRegex(AuditError, "Previously read input changed"):
            ctx.record(self.cgmpath, "second")

    def test_existing_18g_disagreement_is_reported(self):
        file = self.root / "diet_microbiome_glucose_analysis/4New_cgm/outputs/paper20_mediation_phenotype_domains/reports/18g_mean_gmi_redundancy_audit.csv"
        file.parent.mkdir(parents=True)
        pd.DataFrame([dict(mean_highest_robust_paths=999, gmi_highest_robust_paths=0, shared_exposure_species_pairs=0,
                           union_exposure_species_pairs=999, jaccard_mean_vs_gmi=0)]).to_csv(file, index=False)
        code, out, manifest = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(next(s for s in manifest["stages"] if s["stage"] == "20_domains")["status"], "FAIL")

    def test_missing_cgm_row_not_eligible_unknown(self):
        # An absent covariate record cannot be treated as a permissible unknown exclusion flag.
        file = self.root / "co-variant/outputs/data/02_covariate_master.csv"
        cov = pd.read_csv(file)
        cov = cov[~cov.participant_id.eq("SYNTHETIC_0001")]
        cov.to_csv(file, index=False)
        with self.assertRaisesRegex(AuditError, "sample hash mismatch"):
            load_population(self.context(), pd.read_csv(self.cgmpath), self.models)


class FullScaleTests(unittest.TestCase):
    def test_1490_871_109_and_4470_sample_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Data"
            plan, folder, _, _ = fixture(root, full=True)
            args = parser().parse_args(["--data-root", str(root), "--step19b-run", str(folder), "--output-parent", str(Path(tmp) / "runs")])
            with contextlib.redirect_stdout(io.StringIO()): code, out = run(args, plan)
            self.assertEqual(code, 0, (out / "SCREENSHOT_SUMMARY.txt").read_text())
            audit = pd.read_csv(out / "tables/21_path_sample_reconstruction.csv")
            self.assertEqual(len(audit), 4470)
            self.assertTrue(audit.sample_match.all())
            domain = pd.read_csv(out / "tables/20_domain_denominators_and_retention.csv")
            full = domain[domain.panel.eq("full20")]
            self.assertEqual(full.candidate_paths.sum(), 1490)
            self.assertEqual(full.retained_both.sum(), 871)
            self.assertEqual(full.primary_robust.sum(), 1003)


if __name__ == "__main__": unittest.main()
