"""Synthetic-only fixtures. Never read server or participant data."""
import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("direction", PACKAGE / "scripts/19b_core_taxa_direction_synthesis.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(root):
    source = root / "4New_cgm"
    rows = [
        ("AHEI", "cgm_mean", "k__Bacteria|g__A|s__Same", "glycemic_level"),
        ("AMED", "cgm_cv", "k__Bacteria|g__A|s__Same", "glycemic_variability"),
        ("Carbohydrate_pct", "cgm_hbgi", "k__Bacteria|g__A|s__Same", "glycemic_risk_range"),
        ("NOVA4", "cgm_mean", "k__Bacteria|g__B|s__Same", "glycemic_level"),
        ("rEDIH", "cgm_cv", "k__Bacteria|g__B|s__Same", "glycemic_variability"),
        ("hPDI", "cgm_cv", "k__Bacteria|g__C|s__C", "glycemic_variability"),
        ("EAT13", "cgm_hbgi", "k__Bacteria|g__D|s__D", "glycemic_risk_range"),
        ("AHEI", "cgm_cv", "k__Bacteria|g__A|s__Same", "glycemic_variability"),
        ("AMED", "cgm_hbgi", "k__Bacteria|g__E|s__E", "glycemic_risk_range"),
    ]
    data = pd.DataFrame(rows, columns=m.KEYS + ["phenotype_domain"])
    n = len(data)
    high = np.arange(n) < n - 1
    frozen = data[m.KEYS].copy()
    frozen["highest_robustness"] = high
    models = {}
    for tag, N in [("primary", 100), ("strict", 90), ("protocol", 95)]:
        d = data[m.KEYS].copy()
        a = np.array([.2, -.2, .2, -.2, .2, -.2, .2, .2, .2])
        b = np.array([.3, .3, -.3, -.3, .3, -.3, -.3, .3, .3])
        d["a_diet_to_microbiome"] = a
        d["b_microbiome_to_cgm"] = b
        d["indirect_effect"] = a * b
        d["direct_effect"] = a * b
        d["total_effect"] = 2 * a * b
        for effect in ["indirect", "direct", "total"]:
            d[effect + "_CI95_lower"] = d[effect + "_effect"] - .01
            d[effect + "_CI95_upper"] = d[effect + "_effect"] + .01
        d["N"] = N
        d["bootstrap_n"] = 1000
        d["sample_sha256"] = "a" * 64
        d["covariates"] = "age_years;sex" + (";antibiotic_use;ppi_use" if tag == "protocol" else "")
        d["role"] = "primary"
        d["FDR_BH_within_score_outcome"] = np.where(high, .01, .9)
        d["bootstrap_p_indirect"] = np.where(high, .001, .8)
        d["mediation_FDR05_primary"] = high
        d["bridge_direction_consistent_mediator"] = high
        for base, value in [("significant", high), ("consistent", high), ("robust", high), ("indirect_effect", a*b),
                            ("N", N), ("FDR", d["FDR_BH_within_score_outcome"]), ("bootstrap_p", d["bootstrap_p_indirect"])]:
            frozen[base + "_" + tag] = value
        name = "paper20_mediation_20260920_" + tag
        if tag == "protocol":
            name = "paper20_mediation_protocol_abxppi_20260920_primary"
        filename = "18c_strict_mediation_paths.csv" if tag == "strict" else "18b_primary_mediation_paths.csv"
        path = source / "outputs" / name / "models" / filename
        path.parent.mkdir(parents=True)
        d.to_csv(path, index=False)
        models[tag] = path
    frozen["primary_retained_strict"] = high
    frozen["primary_retained_protocol"] = high
    annotated = frozen.copy()
    annotated["phenotype_domain"] = data["phenotype_domain"]
    h = annotated[high].copy()
    summary = h.groupby("species").agg(phenotype_domains=("phenotype_domain", "nunique"), highest_robust_paths=("species", "size"), diet_exposures=("exposure", "nunique"), cgm_outcomes=("outcome_field", "nunique")).reset_index()
    tables = {"18f_all": frozen, "18f_high": frozen[high], "18g": annotated, "19a": h, "19a_species": summary,
              "19a_pan": summary[summary.phenotype_domains.eq(3)], "19a_cross": summary[summary.phenotype_domains.eq(2)]}
    paths = {}
    for name, frame in tables.items():
        path = source / "outputs" / m.SOURCE_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        paths[name] = path
    expected = dict(candidates=9, high=8, species=4, pan=1, cross=1, single=2, primary=8, retained_strict=8, retained_protocol=8,
                    glycemic_level=2, glycemic_variability=4, glycemic_risk_range=2)
    return source, models, paths, expected


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source, self.models, self.paths, self.expected = fixture(self.root)
        self.args = m.parser().parse_args(["--source-root", str(self.source), "--output-parent", str(self.root / "outputs"), "--no-plots"])
        self.args.synthetic = True

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.run(self.args, self.expected)

    def edit(self, path, fun):
        frame = pd.read_csv(path)
        frame = fun(frame)
        frame.to_csv(path, index=False)

    def assert_failure(self, text):
        status, out = self.execute()
        self.assertEqual(status, 2)
        self.assertIn(text, (out / "19b_SCREENSHOT_SUMMARY.txt").read_text())
        self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "FAIL")

    def test_four_sign_combinations(self):
        self.assertEqual(m.signs(pd.Series([.2*.3, .2*-.3, -.2*.3, -.2*-.3])).tolist(), ["positive", "negative", "negative", "positive"])

    def test_invalid_and_near_zero_signs(self):
        self.assertEqual(m.signs(pd.Series([np.nan, np.inf, -np.inf, 0, 1e-14, -1e-14])).tolist(), ["unknown"]*3 + ["near_zero"]*3)

    def test_unknown_boolean_rejected(self):
        for value in [None, np.nan, "maybe", ""]:
            with self.assertRaises(m.AuditError):
                m.booleans(pd.Series([value]))

    def test_mixed_and_uncertain_not_collapsed(self):
        self.assertEqual(m.sign_class(["positive", "negative"]), "mixed")
        self.assertEqual(m.sign_class(["positive", "unknown"]), "uncertain_or_near_zero")
        self.assertEqual(m.stability(pd.DataFrame([["positive", "negative", "positive"], ["positive", "near_zero", "positive"]])).tolist(), ["sign_flip", "uncertain"])

    def test_end_to_end_and_read_only(self):
        hashes = {p: m.sha256(p) for p in self.source.rglob("*.csv")}
        status, out = self.execute()
        self.assertEqual(status, 0)
        d = pd.read_csv(out / "19b_path_direction_details.csv")
        self.assertEqual(len(d), 8)
        self.assertEqual(d.species.nunique(), 4)
        self.assertEqual(d[d.display_species.eq("Same")].taxon_id.nunique(), 2)
        self.assertEqual(set(map(tuple, d[m.KEYS].values)), set(map(tuple, pd.read_csv(self.paths["19a"])[m.KEYS].values)))
        self.assertEqual(hashes, {p: m.sha256(p) for p in hashes})
        self.assertTrue(d.biological_direction_status.eq("descriptive_only_no_health_classification").all())
        summary = pd.read_csv(out / "19b_species_direction_summary.csv")
        self.assertEqual(summary.loc[summary.domain_breadth.eq(3), "indirect_primary_class"].iloc[0], "mixed")

    def test_product_mismatch_fails(self):
        self.edit(self.models["primary"], lambda d: d.assign(a_diet_to_microbiome=d.a_diet_to_microbiome*2))
        self.assert_failure("a*b differs")

    def test_missing_and_nonfinite_fail(self):
        self.edit(self.models["primary"], lambda d: d.assign(a_diet_to_microbiome=np.inf))
        self.assert_failure("missing/nonfinite")

    def test_duplicate_key_fails(self):
        self.edit(self.models["primary"], lambda d: pd.concat([d, d.iloc[:1]]))
        self.assert_failure("0 matching models")

    def test_missing_path_fails(self):
        self.edit(self.models["strict"], lambda d: d.iloc[1:])
        self.assert_failure("0 matching models")

    def test_wrong_effect_run_fails(self):
        self.edit(self.models["protocol"], lambda d: d.assign(indirect_effect=d.indirect_effect*2))
        self.assert_failure("0 matching models")

    def test_ambiguous_candidates_fail_without_mtime_choice(self):
        other = self.source / "outputs/paper20_mediation_NEW_primary/models/18b_primary_mediation_paths.csv"
        other.parent.mkdir(parents=True)
        shutil.copyfile(self.models["primary"], other)
        self.assert_failure("2 matching models")

    def test_explicit_candidate_resolves_ambiguity(self):
        other = self.source / "outputs/paper20_mediation_NEW_primary/models/18b_primary_mediation_paths.csv"
        other.parent.mkdir(parents=True)
        shutil.copyfile(self.models["primary"], other)
        config = self.root / "inputs.json"
        config.write_text(json.dumps({"primary": str(self.models["primary"])}))
        self.args.inputs = str(config)
        self.assertEqual(self.execute()[0], 0)

    def test_newer_wrong_candidate_not_used(self):
        other = self.source / "outputs/paper20_mediation_NEW_primary/models/18b_primary_mediation_paths.csv"
        other.parent.mkdir(parents=True)
        d = pd.read_csv(self.models["primary"])
        d["N"] = 77
        d.to_csv(other, index=False)
        status, out = self.execute()
        self.assertEqual(status, 0)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["selected_models"]["primary"], str(self.models["primary"].resolve()))

    def test_missing_numeric_fails(self):
        self.edit(self.models["primary"], lambda d: d.assign(b_microbiome_to_cgm=np.nan))
        self.assert_failure("missing/nonfinite")

    def test_shuffled_model_order_matches(self):
        self.edit(self.models["primary"], lambda d: d.iloc[::-1])
        self.assertEqual(self.execute()[0], 0)

    def test_changed_fdr_not_accepted(self):
        self.edit(self.models["strict"], lambda d: d.assign(FDR_BH_within_score_outcome=.02))
        self.assert_failure("0 matching models")

    def test_duplicate_dictionary_key_fails(self):
        dictionary = self.root / "dictionary.csv"
        d = pd.read_csv(self.args.dictionary)
        pd.concat([d, d.iloc[:1]]).to_csv(dictionary, index=False)
        self.args.dictionary = str(dictionary)
        self.assert_failure("duplicate identities")

    def test_pending_interpretation_still_outputs_math_only(self):
        dictionary = self.root / "dictionary.csv"
        d = pd.read_csv(self.args.dictionary)
        d.loc[d.field.eq("cgm_cv"), "verification"] = "source_definition_pending"
        d.to_csv(dictionary, index=False)
        self.args.dictionary = str(dictionary)
        status, out = self.execute()
        self.assertEqual(status, 0)
        self.assertIn("PENDING_OUTCOME_MEANING=cgm_cv", (out / "19b_SCREENSHOT_SUMMARY.txt").read_text())

    def test_zero_sign_is_not_negative(self):
        self.assertEqual(m.signs(pd.Series([0.0, -0.0])).tolist(), ["near_zero", "near_zero"])

    def test_inherited_label_change_fails(self):
        self.edit(self.paths["19a"], lambda d: d.assign(consistent_primary=False))
        self.assert_failure("consistent_primary differs")

    def test_wrong_domain_count_fails(self):
        self.expected["glycemic_level"] = 3
        self.assert_failure("glycemic_level: observed=2, expected=3")

    def test_missing_medication_covariates_fail(self):
        self.edit(self.models["protocol"], lambda d: d.assign(covariates="age_years;sex"))
        self.assert_failure("both antibiotic and PPI")

    def test_unknown_dictionary_field_fails(self):
        dictionary = self.root / "dictionary.csv"
        d = pd.read_csv(self.args.dictionary)
        d[~d.field.eq("cgm_cv")].to_csv(dictionary, index=False)
        self.args.dictionary = str(dictionary)
        self.assert_failure("Unmapped outcome")

    def test_check_only_is_not_result_pass(self):
        self.args.check_only = True
        status, out = self.execute()
        self.assertEqual(status, 0)
        self.assertFalse((out / "19b_path_direction_details.csv").exists())
        self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "CHECK_ONLY_PASS")

    def test_expected_real_counts_not_bypassed(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status, out = m.run(self.args)
        self.assertEqual(status, 2)
        self.assertIn("expected=1490", (out / "19b_SCREENSHOT_SUMMARY.txt").read_text())

    def test_ci_order_fails(self):
        self.edit(self.models["primary"], lambda d: d.assign(indirect_CI95_lower=1))
        self.assert_failure("reversed CI")

    def test_bad_tolerance_has_screenshot_failure(self):
        self.args.atol = -1
        self.assert_failure("atol must be finite and nonnegative")

    def test_no_plots_still_has_representative_paths_and_core_page(self):
        status, out = self.execute()
        self.assertEqual(status, 0)
        self.assertEqual(len(pd.read_csv(out / "19b_representative_paths.csv")), 3)
        pages = list(out.glob("19b_CORE_*_SCREENSHOT.txt"))
        self.assertEqual(len(pages), 1)
        text = pages[0].read_text()
        self.assertIn("SYNTHETIC_TEST_ONLY", text)
        self.assertIn("95% CI", text)
        self.assertIn("Carbohydrate_pct", text)

    def test_missing_inherited_header_explains_failure(self):
        self.edit(self.paths["18f_all"], lambda d: d.drop(columns=["highest_robustness"]))
        self.assert_failure("missing columns ['highest_robustness']")


if __name__ == "__main__":
    unittest.main()
