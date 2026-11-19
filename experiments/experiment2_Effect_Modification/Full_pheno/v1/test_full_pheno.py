"""Generated data tests. No native paths, predictors, or outcomes are used."""

from hashlib import sha256
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import protocol as p
from metrics import cluster_contrasts, prediction
from models import fit_fold
from reference import load_v10, receipts
import run
from run import evaluate, header_numbers, read_task, task_path
from source import AuditError, Snapshot, Study, clean_outcome, inventory, raw_ecology, required_files, species_indices


def text_file(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")


def array_file(path, **columns):
    path.parent.mkdir(exist_ok=True, parents=True)
    buf = BytesIO()
    np.savez_compressed(buf, **columns)
    path.write_bytes(buf.getvalue())


def arrow_file(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    pq.write_table(pa.table(data), path)


def file_entry(root, rel):
    val = (root / rel).read_bytes()
    return dict(relative_path=rel, bytes=len(val), sha256=sha256(val).hexdigest())


def manifest(root):
    items = sorted(x.relative_to(root).as_posix() for x in root.rglob("*")
                   if x.is_file() and x != root / p.MANIFEST)
    text_file(root / p.MANIFEST, [file_entry(root, f) for f in items])


def fixture(root):
    """Mirror the handoff's on-disk schema, including shuffled raw outcome rows."""
    rng = np.random.default_rng(90)
    n = 600
    ids = np.arange(9100001, 9100001 + n)
    h = np.arange(n) // 2
    f = h % len(p.FOLDS)
    marker_species = [
        "g__Ruminococcus|s__Ruminococcus_bromii",
        "s__Bifidobacterium_adolescentis",
        "g__Agathobacter|s__Agathobacter_rectalis",
        "s__One", "s__Two", "s__Three", "s__Four", "s__Five", "s__Six",
    ]
    raw = {site: rng.dirichlet(np.ones(len(marker_species))*0.7, size=n) for site in ("oral", "gut")}
    for site, a in raw.items():
        taxa = marker_species if site == "gut" else [f"s__oral_{i}" for i in range(a.shape[1])]
        text_file(root / p.INPUTS / f"{site}_taxa.json", taxa)
        array_file(root / p.INPUTS / f"{site}_raw_TRE_ONLY.npz", abundance=a)
    fiber, upf = rng.uniform(6, 22, n), rng.uniform(.08, .5, n)
    age, sex, energy = rng.uniform(28, 74, n), rng.integers(0, 2, n), rng.uniform(1500, 2500, n)
    day = np.full(n, 4.)
    day[:7] = 2
    fiber[:7] = np.nan
    keys = dict(participant_id=ids.astype(float), cohort=["10k"]*n, research_stage=["00_00_visit"]*n)
    cov = dict(**keys, input_row=np.arange(n), split_unit_id=h, cv_fold=f, final_role=["discovery"]*n,
               age_calendar_years=age, sex_code=sex, smoking_status=rng.choice(["no", "yes", ""], n),
               fiber_density=fiber, n_fiber_days=day, n_energy_days=np.full(n, 4),
               mean_daily_energy_fiber=energy, mean_daily_energy_UPF=energy)
    arrow_file(root / p.INDEX, cov)
    arrow_file(root / p.ROLES, dict(participant_id=np.r_[ids, ids[-1]+1, ids[-1]+2],
                                    cohort=["10k"]*(n+2), research_stage=["00_00_visit"]*(n+2),
                                    split_unit_id=np.r_[h, h.max()+1, h.max()+1],
                                    final_role=["discovery"]*n+["confirmation_reserved"]*2))
    u = (fiber - 10) / 5
    v = np.maximum(fiber - 12.5, 0) / 5
    upf_u = (upf - .25) / .1
    supported = np.ones(n, bool)
    supported[8] = False
    for fold in (*p.FOLDS, None):
        tr = f != fold if fold is not None else np.ones(n, bool)
        pcs = rng.normal(size=(n, 5))
        pcs = (pcs - pcs[tr].mean(0)) / pcs[tr].std(0)
        array_file(root / p.fold_file(fold), training=tr, supported=supported,
                   oral_PC=pcs, gut_PC=pcs[:, ::-1], fiber_u=u, fiber_h=v,
                   UPF_u=upf_u, UPF_h=upf_u**2)
        text_file(root / p.upf_metadata(fold), {"UPF": {"center": .25, "scale": .1, "knots": [-1., 0., 2.]}})

    m = np.log1p(raw["gut"][:, :3].sum(1) / p.PANEL_UNIT)
    s = raw_ecology(raw["gut"])[:, 0]
    diet = np.nan_to_num(u)
    flow = []
    for j, endpoint in enumerate(p.ENDPOINTS):
        y = 90 + 0.25 * age + 2 * sex + rng.normal(0, 4, n) + rng.normal(0, 1, h.max()+1)[h]
        y += 9 * diet * (m - m.mean()) + 15 * diet * (s - s.mean())
        y[14:14+j] = np.nan
        kept = rng.permutation(np.flatnonzero(np.isfinite(y)))
        y_table = dict({c: np.asarray(keys[c])[kept] for c in p.KEY},
                       input_row=kept, Y_value=y[kept], Y_unit=["mg/dL" if j == 0 else "test-unit"]*len(kept))
        arrow_file(root / p.outcome_file(endpoint), y_table)
        for fold in (*p.FOLDS, None):
            tr = f != fold if fold is not None else np.ones(n, bool)
            for exposure in p.EXPOSURES:
                ok = np.isfinite(clean_outcome(y, tr)) & supported
                if exposure == "fiber_density":
                    ok &= np.isfinite(u) & (day >= 3)
                flow.append(dict(preprocessing="full" if fold is None else f"fold{fold}",
                                 analysis_id=endpoint, exposure_id=exposure,
                                 eligible=int(ok.sum()), training=int((ok & tr).sum()), validation=int((ok & ~tr).sum())))
    text_file(root / p.FLOW, flow)
    manifest(root)


class FullPhenoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory(prefix="hpp_full_pheno_synthetic_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name) / p.HANDOFF
        fixture(cls.root)

    def study(self, scope):
        tasks = p.MODES[scope]
        snapshot = Snapshot(self.root, required_files(tasks))
        return Study(snapshot, with_upf=(scope in ("upf", "all")), expected=None)

    def test_inventory_only_opens_footers(self):
        tasks = p.MODES["fiber"]
        snap = Snapshot(self.root, required_files(tasks))
        with patch.object(snap, "bytes", side_effect=AssertionError("outcome values loaded")):
            inv = inventory(snap, tasks)
        self.assertEqual((inv["discovery_rows"], inv["roster_rows"]), (600, 602))
        self.assertEqual(set(inv["raw_outcome_rows"]), set(p.ENDPOINTS))

    def test_held_out_y_does_not_change_predictions(self):
        study = self.study("bridge")
        endpoint, exposure = "D_Glu", "fiber_density"
        outcome = study.outcome(endpoint)
        before = fit_fold(study, exposure, endpoint, outcome, 0)
        changed = outcome.y.copy()
        changed[before.rows] += 0.001
        after = fit_fold(study, exposure, endpoint, type(outcome)(changed, outcome.unit), 0)
        self.assertFalse(np.array_equal(before.y, after.y))
        np.testing.assert_array_equal(before.rows, after.rows)
        np.testing.assert_array_equal(before.predictions, after.predictions)

    def test_panel_species_reject_ambiguous_aliases(self):
        good = ["s__Ruminococcus_bromii", "s__Bifidobacterium_adolescentis",
                "g__Agathobacter|s__Agathobacter_rectalis"]
        self.assertEqual(list(species_indices(good).values()), [0, 1, 2])
        with self.assertRaisesRegex(AuditError, "THREE_SPECIES_MATCH_COUNT"):
            species_indices([*good, "s__Eubacterium_rectale"])
        with self.assertRaisesRegex(AuditError, "THREE_SPECIES_MATCH_COUNT"):
            species_indices([*good[:2], good[-1] + "|t__SGB0001"])

    def test_mismatched_outcome_key_even_with_valid_checksum_stops(self):
        study = self.study("bridge")
        rel = p.outcome_file("D_Glu")
        table = pq.ParquetFile(BytesIO((self.root / rel).read_bytes())).read()
        name = "participant_id"
        i = table.column_names.index(name)
        bad = table.set_column(i, name, pa.array(table[name].to_pylist()[::-1]))
        buf = pa.BufferOutputStream()
        pq.write_table(bad, buf)
        with patch.object(study.snapshot, "bytes", side_effect=lambda _: buf.getvalue().to_pybytes()):
            with self.assertRaisesRegex(AuditError, "OUTCOME_COMPOSITE_KEY_MISMATCH"):
                study.outcome("D_Glu")

    def test_outcome_hash_mismatch_stops(self):
        study = self.study("bridge")
        rel = p.outcome_file("D_Glu")
        path, _ = study.snapshot.paths[rel]
        study.snapshot.paths[rel] = (path, "0"*64)
        with self.assertRaisesRegex(AuditError, "INPUT_HASH_MISMATCH"):
            study.outcome("D_Glu")

    def test_ratio_and_delta_use_the_correct_denominator(self):
        rng = np.random.default_rng(18)
        y = rng.normal(size=300)
        main, panel, full = .4*y, .7*y, .8*y
        families = np.arange(len(y))//2
        result = prediction(y, np.column_stack((main, panel, full)), families, 100, 9)
        sst = ((y-y.mean())**2).sum()
        direct = [1-((y - a)**2).sum()/sst for a in (main, panel, full)]
        pairs = result["pairs"]
        self.assertAlmostEqual(pairs["all_interactions"]["delta_r2"], direct[2]-direct[0], places=13)
        self.assertAlmostEqual(pairs["all_interactions"]["share_pct"], 100*(direct[2]-direct[0])/direct[2], places=12)
        self.assertAlmostEqual(pairs["shannon_increment"]["delta_r2"], direct[2]-direct[1], places=13)
        self.assertAlmostEqual(sum(pairs[k]["delta_r2"] for k in ("panel_minus_off", "shannon_increment")),
                               pairs["all_interactions"]["delta_r2"], places=13)
        case = prediction(y, np.column_stack((main, panel, -y)), families, 100, 9)["pairs"]["all_interactions"]
        self.assertLess(case["delta_r2"], 0)
        self.assertIsNone(case["share_pct"])
        self.assertIn("unstable", case["share_status"])
        with self.assertRaisesRegex(AuditError, "SCORING_COLUMNS_DO_NOT_MATCH_MODELS"):
            prediction(y, np.column_stack((main, panel)), families, 100, 9)

    def test_negative_delta_is_visible_with_stable_positive_r2(self):
        y = np.linspace(-10., 20., 500) - 5.
        g = np.arange(len(y)) // 2
        stats = prediction(y, np.column_stack((.75*y, .80*y, .70*y)), g, 200, 1)
        pair = stats["pairs"]["all_interactions"]
        self.assertLess(pair["delta_r2"], 0.)
        self.assertLess(pair["share_pct"], 0.)
        self.assertLess(pair["delta_ci95"][1], 0.)

    def test_rank_deficient_modification_does_not_return_arbitrary_coefficient(self):
        rng = np.random.default_rng(128)
        z = rng.normal(size=200)
        x = np.column_stack((np.ones(len(z)), z, z))
        y = 2 + 3*z + rng.normal(0, .1, len(z))
        rank, effects = cluster_contrasts(x, y, np.arange(len(y))//2, np.array([[0, 1, 0], [0, 1, 1]]))
        self.assertEqual(rank, 2)
        self.assertEqual(effects[0]["status"], "unavailable")
        self.assertAlmostEqual(effects[1]["estimate"], 3, places=1)
        self.assertGreater(effects[1]["simultaneous_ci95"][1]-effects[1]["simultaneous_ci95"][0],
                           effects[1]["ci95"][1]-effects[1]["ci95"][0])

    def test_numerical_noise_not_rescaled_into_a_valid_modifier(self):
        rng = np.random.default_rng(19)
        n = 400
        z = rng.normal(size=(n, 2))
        y = 100 + z[:, 0] + .3*z[:, 1] + rng.normal(size=n)
        design = np.column_stack((np.ones(n), z[:, 0], 1e-15*z[:, 1]))
        rank, effects = cluster_contrasts(design, y, np.arange(n)//2, np.array([[0., 1., 0.], [0., 0., 1.]]))
        self.assertEqual(rank, 2)
        self.assertEqual(effects[0]["status"], "ok")
        self.assertEqual(effects[1]["status"], "unavailable")

    def test_fixed_recipe_detects_generated_interactions(self):
        study = self.study("bridge")
        result = evaluate(study, "fiber_density", "D_Glu", study.outcome("D_Glu"), draws=200)
        self.assertEqual(result["unit"], "mg/dL")
        self.assertEqual(result["people"], sum(f["held_people"] for f in result["folds"]))
        self.assertGreater(result["score"]["pairs"]["all_interactions"]["delta_r2"], .10)
        self.assertGreater(result["score"]["pairs"]["shannon_increment"]["delta_r2"], .02)
        for marker in p.MARKERS:
            v = result["association"]["markers"][marker]
            self.assertEqual(v["status"], "ok")
            self.assertGreater(v["estimate"], 1.)
            self.assertTrue(v["support"]["supported"])

    def test_end_to_end_expand_bridge_to_eighteen_and_detect_tamper(self):
        # Run the exact server wrapper in a generated copy of the upload layout.
        with TemporaryDirectory(prefix="hpp_fullpheno_test_run_") as dest:
            version = Path(dest) / "exp2/Full_pheno/v1"
            version.mkdir(parents=True)
            for name in run.RELEASE_FILES:
                shutil.copyfile(p.HERE / name, version / name)
            spec = version / "protocol.py"
            spec.write_text(spec.read_text().replace("BOOTSTRAP_DRAWS = 9999", "BOOTSTRAP_DRAWS = 90"))
            driver = version / "run.py"
            driver.write_text(driver.read_text().replace(
                'study = Study(snapshot, with_upf=any(x == "UPF" for x, _ in tasks))',
                'study = Study(snapshot, with_upf=any(x == "UPF" for x, _ in tasks), expected=None)',
            ))
            # Exercise launcher resolution without modifying the user's interpreter.
            pointer = version.parents[2] / "experiments/diet_effect_modification_20261109_v12/.python_path"
            pointer.parent.mkdir(parents=True)
            pointer.write_text(sys.executable + "\n")
            release = dict(version=p.VERSION, files={name: file_entry(version, name)["sha256"]
                                                    for name in run.RELEASE_FILES})
            text_file(version / "upload_manifest.json", release)
            env = {k: v for k, v in os.environ.items() if k not in {"HPP_FULL_PHENO_PYTHON", "HPP_EXP2_PYTHON"}}

            def launch(scope, *mode):
                command = ["bash", str(version / "server_run.sh"), "--handoff", str(self.root),
                           "--threads", "2", "--scope", scope, *mode]
                result = subprocess.run(command, cwd=dest, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            inv = launch("fiber", "--inventory")
            self.assertIn("FULL_PHENO_V1_INVENTORY", inv)
            out = version / "runs/TRE_ONLY/discovery-1"
            self.assertFalse(out.exists())
            self.assertIn("FULL_PHENO_V1_COMPLETE", launch("bridge"))
            bridge = task_path(out, "fiber_density", "D_Glu")
            saved = bridge.read_bytes()
            mismatch = subprocess.run(["bash", str(version / "server_run.sh"), "--handoff", str(self.root),
                                       "--threads", "1", "--scope", "upf"],
                                      cwd=dest, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("RUN_REGISTRATION_DIFFERS", mismatch.stdout)
            self.assertFalse(list(out.glob("UPF__*.json")))
            log = launch("all")
            self.assertIn("FULL_PHENO_V1_REUSE", log)
            self.assertIn('"combinations": 18', log)
            self.assertEqual(bridge.read_bytes(), saved)
            results = [read_task(task_path(out, x, e), x, e, release)["result"] for x, e in p.MODES["all"]]
            self.assertNotEqual(results[0]["people"], results[8]["people"])
            self.assertTrue(all(r["score"]["status"] == "ok" for r in results))
            self.assertTrue(all(r["association"]["moderation_dose"] == "linear_calibrated_fraction" for r in results[9:]))
            self.assertTrue((out / "all_summary.md").is_file())
            with (out / "all_summary.csv").open(newline="", encoding="utf-8") as f:
                table = list(csv.DictReader(f))
            self.assertEqual(len(table), 18)
            for record, got in zip(results, table, strict=True):
                for key in ("panel_minus_off", "shannon_increment", "all_interactions"):
                    pair = record["score"]["pairs"][key]
                    self.assertAlmostEqual(float(got[f"{key}_delta_r2_pp"]), 100*pair["delta_r2"])
                values = header_numbers(record)
                self.assertAlmostEqual(values["panel_minus_off_delta_r2_pp"],
                                       values["all_interactions_delta_r2_pp"]-values["shannon_increment_delta_r2_pp"])
            report = launch("all", "--report")
            self.assertIn("V10_COMPLETION_RECEIPT_NOT_FOUND", report)
            self.assertIn("Panel delta [95%]", report)
            self.assertNotIn("910000", (out / "all_summary.csv").read_text() + log + report)
            self.assertFalse(list(out.glob("*.npz")))
            path = task_path(out, "fiber_density", "D_Glu")
            item = json.loads(path.read_text())
            item["payload"]["result"]["people"] += 1
            text_file(path, item)
            with self.assertRaisesRegex(ValueError, "RESULT_HASH_MISMATCH"):
                read_task(path, "fiber_density", "D_Glu", release)

    def test_v10_archive_matches_exactly_and_rejects_shifted_targets(self):
        study = self.study("bridge")
        outcome = study.outcome("D_Glu")
        folds = [fit_fold(study, "fiber_density", "D_Glu", outcome, f) for f in p.FOLDS]
        with TemporaryDirectory(prefix="hpp_fullpheno_v10_") as name:
            root = Path(name)
            arms = ["other", p.V10_SETTING.split("@")[0]]
            settings = ["other@1", p.V10_SETTING]
            column_hash = sha256(json.dumps([arms, settings, [0, 1], []], separators=(",", ":")).encode()).hexdigest()
            text_file(root / "design.json", dict(family_order=arms, settings_order=settings, adaptive=[],
                                                paired_column_sha256=column_hash))
            design_sha = file_entry(root, "design.json")["sha256"]
            out = root / "runs/TRE_ONLY/v10"
            sha_reg = {str(path): digest for path, digest in study.snapshot.paths.values()}
            reg = dict(source_hashes=sha_reg, release=dict(files={"design.json": design_sha}),
                       protocol=dict(paired_column_sha256=column_hash, outcome_scope="discovery only"))
            text_file(out / "registration.json", reg)
            text_file(out / "COMPLETE.json", dict(version="v10", artifacts={
                "registration.json": file_entry(out, "registration.json")["sha256"],
            }))
            for shift in (0, .01):
                for f in folds:
                    trial = out / f"r0_f{f.fold}"
                    n = len(f.rows)
                    order = np.arange(n)[::-1]
                    array_file(trial / "arrays.npz", rows=f.rows[order], groups=study.family[f.rows[order]],
                               y=f.y[order]+shift, setting_off=f.predictions[order, :2],
                               setting_on=f.predictions[order, 1:])
                    text_file(trial / "metadata.json", dict(fold=f.fold, repeat=0, paired_column_sha256=column_hash,
                                                            people=n, additive_models_checked=2))
                    text_file(trial / "receipt.json", dict(arrays_sha256=file_entry(trial, "arrays.npz")["sha256"],
                                                           metadata_sha256=file_entry(trial, "metadata.json")["sha256"]))
                with patch.object(p, "V10_DESIGN_SHA", design_sha):
                    ref = load_v10(root, study.snapshot, folds, study.family, 90)
                self.assertEqual(ref["status"], "not_matched" if shift else "matched")
                if shift:
                    self.assertIn("V10_SCORED_OUTCOME_DIFFERS", ref["reason"])
                else:
                    self.assertGreater(ref["scoring"]["pairs"]["all_interactions"]["delta_r2"], .02)
            original = receipts(root)
            trial = out / "r0_f0"
            (trial / "arrays.npz").write_bytes(b"PK\x03\x04" + b"broken zip")
            self.assertNotEqual(receipts(root), original)
            # Even a checksum-consistent, undecodable old archive must not stop A/B/C.
            text_file(trial / "receipt.json", dict(arrays_sha256=file_entry(trial, "arrays.npz")["sha256"],
                                                   metadata_sha256=file_entry(trial, "metadata.json")["sha256"]))
            with patch.object(p, "V10_DESIGN_SHA", design_sha):
                failed = evaluate(study, "fiber_density", "D_Glu", outcome, draws=90, v10_root=root)
            self.assertEqual(failed["v10_reference"], dict(status="not_verified", reason="V10_ARCHIVE_BadZipFile"))
            self.assertEqual(failed["score"]["status"], "ok")
