"""Synthetic scientific and upload regression tests; no real HPP observations."""
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from hppmeal.common import Run, read_csv, sha256, write_json
from seven_diet_meals.inputs import record_expected, unchanged
from seven_diet_meals.tests.test_analysis import make_source
from gut_meal_bridge_v2.inputs import (PACKAGE, MODULE, bundle, contract, inspect, load, resolve_species,
                                     timing_table, date_only)
from gut_meal_bridge_v2.analysis import prepare, analyze, estimate, model_columns
from gut_meal_bridge_v2.reporting import show
from gut_meal_bridge_v2.run_analysis import main, protect_output


def fixture(tmp_path, n=180):
    root, meal_run = make_source(tmp_path, n=n)
    old = json.loads((meal_run / "manifest.json").read_text())
    rng = np.random.RandomState(829)
    ids = pd.Series(["{:06d}".format(i) for i in range(n)], name="participant_id")
    microbes = pd.DataFrame(dict(participant_id=ids, cohort="10k", research_stage="00_00_visit",
        sample_name=ids + "_stool"))
    taxon_columns = []
    for taxon in contract()["taxa"]:
        col = "k__Bacteria|g__Example|s__" + taxon["species"]
        microbes[col] = rng.normal(size=n)
        taxon_columns.append(col)
    micro_path = root / old["config"]["inputs"]["microbiome"]
    micro_path.parent.mkdir(parents=True, exist_ok=True)
    microbes.to_csv(micro_path, index=False)
    old["inputs"].append(dict(label="microbiome", path=str(micro_path), sha256=sha256(micro_path)))
    meals_path = meal_run / "private/02_meals.csv"
    meals = pd.read_csv(meals_path, dtype={"participant_id": str})
    meals["connection_id"] = meals.participant_id + "_connection"
    micro_signal = meals.participant_id.map(microbes.set_index("participant_id")[taxon_columns[2]])
    meals["baseline_glucose"] += 2 * micro_signal
    meals["iauc_net_0_120"] += 120 * 3 * micro_signal
    meals["peak_rise_0_120"] += 4 * micro_signal
    meals.to_csv(meals_path, index=False)
    old["status"]["build"]["meals_sha256"] = sha256(meals_path)
    write_json(meal_run / "manifest.json", old)
    settings = json.loads((MODULE / "sources.json").read_text())
    cg = pd.DataFrame(dict(participant_id=ids, cohort="10k", research_stage="00_00_visit",
                          connection_id=ids + "_connection"))
    for f in contract()["reference_outcomes"]:
        cg[f] = .3 * microbes[taxon_columns[2]] + rng.normal(size=n)
    cg_path = root / settings["whole_cgm"]
    cg_path.parent.mkdir(parents=True, exist_ok=True)
    cg.to_csv(cg_path, index=False)
    dates = microbes[["participant_id", "cohort", "research_stage", "sample_name"]].copy()
    dates["collection_date"] = "2024-01-02"
    dates_path = root / "Transfer/gut_microbiome/gut_microbiome.csv"
    dates_path.parent.mkdir(parents=True, exist_ok=True)
    dates.to_csv(dates_path, index=False)
    nutrition = root / "meal_cgm_exploration/outputs/nutrition_fixture"
    nutrition.mkdir(parents=True)
    scores = json.loads((PACKAGE / "seven_diet_meals/sources.json").read_text())
    frozen_paths = {label: root / path for label, path in scores.items()}
    frozen_paths.update(covariates=root / old["config"]["inputs"]["covariates"],
                        meals=meals_path, whole_cgm=cg_path)
    write_json(nutrition / "manifest.json", dict(data_kind="SYNTHETIC", root=str(root), config=old["config"],
        status=dict(nutrition_bridge=dict(status="COMPLETE")),
        inputs=[dict(path=str(path), label=label, sha256=sha256(path)) for label, path in frozen_paths.items()]))
    return root, meal_run, nutrition, micro_path, dates_path, cg_path


def loaded(tmp_path, n=180):
    root, meal, nutrition, mp, dp, cp = fixture(tmp_path, n)
    root, c, paths, pinned, audit, mapping = inspect(meal, nutrition, root, MODULE / "sources.json",
        PACKAGE / "seven_diet_meals/sources.json")
    run = Run(root, tmp_path / "scratch", c)
    record_expected(run, pinned)
    meals, people = load(run, paths, mapping)
    return run, meals, people, mapping, (root, meal, nutrition, mp, dp, cp)


def run_args(files, out, extra=()):
    root, meal, nutrition, _, _, _ = files
    return ["--meal-run", str(meal), "--nutrition-run", str(nutrition), "--data-root", str(root),
            "--output-root", str(out)] + list(extra)


def replace_synthetic_meals(files, frame):
    """Rebuild only synthetic fixtures, including their historical fingerprints."""
    meal_run, nutrition = files[1:3]
    path = meal_run / "private/02_meals.csv"
    old = json.loads((meal_run / "manifest.json").read_text())
    prior = json.loads((nutrition / "manifest.json").read_text())
    assert old["data_kind"] == prior["data_kind"] == "SYNTHETIC"
    frame.to_csv(path, index=False)
    old["status"]["build"]["meals_sha256"] = sha256(path)
    for item in prior["inputs"]:
        if item["label"] == "meals":
            item["sha256"] = sha256(path)
    write_json(meal_run / "manifest.json", old)
    write_json(nutrition / "manifest.json", prior)


def test_ineligible_meals_without_connections_do_not_block_or_change_models(tmp_path):
    files = fixture(tmp_path)
    before = tmp_path / "before"
    after = tmp_path / "after"
    assert main(run_args(files, before)) == 0
    raw = read_csv(files[1] / "private/02_meals.csv")
    extra = raw.head(3).copy()
    extra["meal_id"] += "_ineligible"
    extra["analysis_ready"] = "False"
    extra["connection_id"] = [None, "", "another_excluded_connection"]
    replace_synthetic_meals(files, pd.concat([raw, extra], ignore_index=True))
    assert main(run_args(files, after)) == 0
    first, second = next(before.iterdir()), next(after.iterdir())
    pd.testing.assert_frame_equal(pd.read_csv(first / "tables/models.csv"),
                                  pd.read_csv(second / "tables/models.csv"))
    audit = pd.read_csv(second / "tables/01_meal_connection_audit.csv").set_index("scope")
    assert audit.loc["not_analysis_ready", "meals"] == 3
    assert audit.loc["not_analysis_ready", "missing_connection"] == 2
    assert audit.loc["analysis_ready", "missing_connection"] == 0


@pytest.mark.parametrize("problem", ["missing", "multiple"])
def test_analysis_ready_meal_connection_problems_still_fail(tmp_path, problem):
    files = fixture(tmp_path)
    raw = read_csv(files[1] / "private/02_meals.csv")
    raw.loc[0, "connection_id"] = None if problem == "missing" else "another_connection"
    replace_synthetic_meals(files, raw)
    out = tmp_path / "results"
    assert main(run_args(files, out)) == 2
    state = json.loads((next(out.iterdir()) / "manifest.json").read_text())["status"]["gut_meal_bridge_v2"]
    assert state["status"] == "FAILED"
    assert ("Missing meal connection ID" if problem == "missing" else "Multiple meal CGM connections") in state["reason"]


def test_end_to_end_48_models_three_pages_and_unchanged_sources(tmp_path, capsys):
    files = fixture(tmp_path)
    hashes = {p: sha256(p) for p in files[0].rglob("*") if p.is_file()}
    out = tmp_path / "results"
    assert main(run_args(files, out)) == 0
    run = next(out.iterdir())
    models = pd.read_csv(run / "tables/models.csv")
    assert len(models) == 48 and models.status.eq("ESTIMATED").all()
    for setting, group in models.groupby("setting"):
        assert group.person_sha256.nunique() == group.meal_sha256.nunique() == 1
        assert group.groupby("family").size().to_dict() == {"baseline_peak_secondary": 8, "net_primary": 4, "whole_cgm_reference": 12}
    assert len(list((run / "pages").glob("*.txt"))) == 3
    assert all(len(p.read_text().splitlines()) <= 34 for p in (run / "pages").glob("*.txt"))
    assert all("SYNTHETIC" in p.read_text() for p in (run / "pages").glob("*.txt"))
    assert not any("000001" in p.read_text() for p in (run / "pages").glob("*.txt"))
    assert all(sha256(p) == digest for p, digest in hashes.items())
    assert main(["--show", "latest", "--output-root", str(out), "--brief"]) == 0
    assert "KEY 3/3" in capsys.readouterr().out


def test_hc3_estimate_matches_independent_matrix_reference(tmp_path):
    run, meals, people, _, _ = loaded(tmp_path)
    p, m, _, _ = prepare(meals, people)
    row, _ = estimate(p, m, "primary_common", "T068", "mean_increment")
    from seven_diet_meals.models import encode
    X = np.column_stack([np.ones(len(p)), encode(p, model_columns("T068", "mean_increment"))])
    y = p.mean_increment.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bread = np.linalg.inv(X.T @ X)
    residual = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    scores = X * (residual / (1 - leverage))[:, None]
    cov = bread @ (scores.T @ scores) @ bread
    se = np.sqrt(cov[1, 1])
    assert row["estimate"] == pytest.approx(beta[1], abs=1e-8)
    assert row["se"] == pytest.approx(se, abs=1e-8)
    assert row["ci_high"] == pytest.approx(beta[1] + stats.t.ppf(.975, len(p) - X.shape[1]) * se, abs=1e-8)
    # In a finite sample the inherited diet/noise signal can correlate with taxa.
    # Adding exactly 3*taxon to the outcome must shift the slope by exactly 3.
    unplanted = p.copy()
    unplanted["mean_increment"] -= 3 * unplanted.T068
    before, _ = estimate(unplanted, m, "primary_common", "T068", "mean_increment")
    assert row["estimate"] - before["estimate"] == pytest.approx(3, abs=1e-10)
    assert row["se"] == pytest.approx(before["se"], abs=1e-10)


def test_original_units_and_baseline_not_self_adjusted(tmp_path):
    run, meals, people, mapping, files = loaded(tmp_path)
    raw = pd.read_csv(files[3], dtype={"participant_id": str}).set_index("participant_id")
    for item in mapping:
        assert np.allclose(people.loc[raw.index, item["taxon"]], raw[item["column"]])
    assert np.allclose(meals.mean_increment, meals.iauc_net_0_120 / 120)
    assert people.hPDI.max() < 999  # rejects stale deliberately wrong fixture hPDI
    columns = model_columns("T068", "baseline_glucose")
    assert "baseline_glucose" not in columns and "pre_slope_per_min" not in columns
    assert "baseline_glucose" in model_columns("T068", "mean_increment")
    p, m, _, _ = prepare(meals, people)
    result = analyze(p, m, lambda x: None)
    base = result["models"].query("setting == 'primary_common' and taxon == 'T068' and outcome == 'baseline_glucose'").iloc[0]
    assert base.status == "ESTIMATED" and base.estimate > 1


def test_common_membership_and_connection_mismatch(tmp_path):
    run, meals, people, _, _ = loaded(tmp_path)
    people.loc["000001", "T023"] = np.nan
    people.loc["000002", "reference_connection"] = "other_connection"
    people.loc["000003", "cgm_cv_z"] = np.nan
    p, m, flow, mismatch = prepare(meals, people)
    assert not set(["000001", "000002", "000003"]) & set(p.participant_id)
    assert mismatch["people"] == 1 and mismatch["meals"] == 12
    result = analyze(p, m, lambda x: None)
    assert result["models"].groupby("setting").person_sha256.nunique().eq(1).all()
    assert p.loc[p.participant_id.eq("000013"), "strict_eligible"].eq(False).all()


def test_person_equal_weight_and_mean_of_carb_square(tmp_path):
    run, meals, people, _, _ = loaded(tmp_path)
    p, m, _, _ = prepare(meals, people)
    extra = meals.loc[meals.participant_id.eq("000001")].copy()
    extra["meal_id"] += "_extra"
    p2, m2, _, _ = prepare(pd.concat([meals, extra], ignore_index=True), people)
    assert np.allclose(p.mean_increment, p2.mean_increment)
    a, _ = estimate(p, m, "primary_common", "T068", "mean_increment")
    b, _ = estimate(p2, m2, "primary_common", "T068", "mean_increment")
    assert a["estimate"] == pytest.approx(b["estimate"], abs=1e-10)
    assert (p.carb_10_squared > p.carb_10 ** 2).all()


def test_repeated_meal_support_and_missing_covariate_row(tmp_path):
    run, meals, people, _, _ = loaded(tmp_path)
    # Four meals are insufficient even across three dates; twelve meals across
    # two dates are also insufficient. A missing entire covariate row is excluded.
    keep = meals.loc[meals.participant_id.eq("000001"), "meal_id"].iloc[[0, 3, 6, 9]]
    meals = meals.loc[~meals.participant_id.eq("000001") | meals.meal_id.isin(keep)].copy()
    meals.loc[meals.participant_id.eq("000002"), "local_date"] = "2024-01-01"
    people = people.drop(index="000003")
    p, _, _, _ = prepare(meals, people)
    assert not {"000001", "000002", "000003"} & set(p.participant_id)
    assert "000004" in set(p.participant_id)


@pytest.mark.parametrize("which", ["microbiome", "whole_cgm"])
def test_duplicate_person_identity_rejected_by_loader(tmp_path, which):
    run, _, _, mapping, files = loaded(tmp_path)
    _, _, paths, _, _, _ = inspect(files[1], files[2], files[0], MODULE / "sources.json",
        PACKAGE / "seven_diet_meals/sources.json")
    path = paths[which]
    frame = read_csv(path)
    pd.concat([frame, frame.head(1)], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate identity"):
        load(run, paths, mapping)


def test_nonfinite_taxon_rejected_by_loader(tmp_path):
    run, _, _, mapping, files = loaded(tmp_path)
    _, _, paths, _, _, _ = inspect(files[1], files[2], files[0], MODULE / "sources.json",
        PACKAGE / "seven_diet_meals/sources.json")
    frame = read_csv(paths["microbiome"])
    frame.loc[0, mapping[0]["column"]] = "inf"
    frame.to_csv(paths["microbiome"], index=False)
    with pytest.raises(ValueError, match="Invalid numeric"):
        load(run, paths, mapping)


@pytest.mark.parametrize("n", [0, 30])
def test_empty_or_small_sample_keeps_48_planned_rows(tmp_path, n):
    run, meals, people, _, _ = loaded(tmp_path)
    p, m, _, _ = prepare(meals, people)
    p = p.head(n)
    m = m.loc[m.participant_id.isin(p.participant_id)]
    r = analyze(p, m, lambda x: None)["models"]
    assert len(r) == 48 and r.status.eq("NOT_ESTIMATED").all()
    assert r.q_family.isna().all() and r.q_meal12.isna().all()


def test_empty_prepare_and_missing_reference_do_not_crash(tmp_path):
    run, meals, people, _, _ = loaded(tmp_path)
    people["T023"] = np.nan
    p, m, _, _ = prepare(meals, people)
    assert len(p) == len(m) == 0
    assert len(analyze(p, m, lambda x: None)["models"]) == 48


def test_constant_taxon_and_fdr_preserve_family_denominator(tmp_path):
    run, meals, people, _, _ = loaded(tmp_path)
    people["T023"] = 0.0
    p, m, _, _ = prepare(meals, people)
    r = analyze(p, m, lambda x: None)["models"]
    assert r.loc[r.taxon.eq("T023"), "status"].eq("NOT_ESTIMATED").all()
    primary = r.query("setting == 'primary_common' and family == 'net_primary'")
    from hppmeal.models import bh_planned
    assert np.allclose(primary.q_family, bh_planned(primary.p), equal_nan=True)
    assert primary.planned_family_size.eq(4).all()


def test_duplicate_species_or_substring_rejected():
    taxa = [contract()["taxa"][2]]
    with pytest.raises(ValueError, match="found 0"):
        resolve_species(["s__Eubacterium_rectale_other"], taxa)
    with pytest.raises(ValueError, match="found 2"):
        resolve_species(["s__Eubacterium_rectale", "k__Bacteria|s__Eubacterium_rectale"], taxa)


def test_date_audit_unknown_ambiguous_exact_sample_and_conflict():
    c = json.loads((PACKAGE / "config/default.json").read_text())
    micro = pd.DataFrame(dict(participant_id=["a", "b", "c", "d", "e"], sample_name=["a2", None, "c", "d", "not_found"]))
    dates = pd.DataFrame(dict(participant_id=["a", "a", "b", "b", "c", "d", "e"],
        sample_name=["a1", "a2", "b1", "b2", "c", "d", "e"], cohort="10k", research_stage="00_00_visit",
        collection_date=["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "invalid", "2024-01-01", "2024-01-01"],
        collection_timestamp=[None, None, None, None, None, "2024-01-02T12:00:00+02:00", None]))
    result = timing_table(micro, dates, c).set_index("participant_id")
    assert result.loc["a", "sample_date"] == pd.Timestamp("2024-01-02")
    assert result.loc["a", "timing_status"] == "MATCHED_SAMPLE"
    assert result.loc["b", "timing_status"] == "AMBIGUOUS_MULTIPLE_SAMPLES"
    assert result.loc["c", "timing_status"] == "UNKNOWN_INVALID_DATE"
    assert result.loc["d", "timing_status"] == "UNKNOWN_DATE_CONFLICT"
    assert result.loc["e", "timing_status"] == "UNKNOWN_SAMPLE_MISMATCH"
    assert timing_table(micro, None, c).sample_date.isna().all()
    assert date_only("2024-01-01T23:30:00Z", "Asia/Jerusalem") == pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="duplicate identity"):
        timing_table(micro, pd.concat([dates, dates.head(1)]), c)


def test_missing_date_file_still_runs_with_unknown_timing(tmp_path):
    files = fixture(tmp_path)
    files[4].unlink()
    out = tmp_path / "results"
    assert main(run_args(files, out)) == 0
    t = pd.read_csv(next(out.iterdir()) / "tables/timing.csv")
    assert t.known_date.eq(0).all() and t.unknown_date.eq(t.people).all()


@pytest.mark.parametrize("which", ["microbiome", "whole_cgm", "meals", "current4", "new3", "corrected_hpdi"])
def test_changed_frozen_input_rejected_and_failure_saved(tmp_path, which):
    files = fixture(tmp_path)
    targets = {"microbiome": files[3], "whole_cgm": files[5], "meals": files[1] / "private/02_meals.csv"}
    scores = json.loads((PACKAGE / "seven_diet_meals/sources.json").read_text())
    targets.update({label: files[0] / path for label, path in scores.items()})
    target = targets[which]
    with target.open("a") as f:
        f.write("\n")
    out = tmp_path / "results"
    assert main(run_args(files, out)) == 2
    manifest = json.loads((next(out.iterdir()) / "manifest.json").read_text())
    assert manifest["status"]["gut_meal_bridge_v2"]["status"] == "FAILED"


def test_change_after_validation_rejected(tmp_path):
    run, _, _, _, files = loaded(tmp_path)
    with files[3].open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="changed"):
        unchanged(run)


def test_check_only_and_latest_failure_never_show_old_success(tmp_path):
    files = fixture(tmp_path)
    out = tmp_path / "results"
    assert main(run_args(files, out)) == 0
    assert main(run_args(files, out, ["--stage", "check"])) == 0
    assert main(["--show", "latest", "--output-root", str(out)]) == 2
    files[3].unlink()
    assert main(run_args(files, out)) == 2
    assert main(["--show", "latest", "--output-root", str(out)]) == 2


def test_modified_page_rejected(tmp_path):
    files = fixture(tmp_path)
    out = tmp_path / "results"
    assert main(run_args(files, out)) == 0
    run = next(out.iterdir())
    with (run / "pages/01.txt").open("a") as f:
        f.write("tamper")
    with pytest.raises(ValueError, match="changed result"):
        show(str(run), out, 1)


def test_python38_syntax_and_no_new_builtin_generic_annotations():
    for path in MODULE.glob("*.py"):
        tree = ast.parse(path.read_text(), feature_version=(3, 8))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Subscript):
                assert not (isinstance(node.annotation.value, ast.Name) and node.annotation.value.id in {"list", "dict", "set", "tuple"})


def test_protected_output_directories(tmp_path):
    for path in [MODULE, MODULE / "bad", PACKAGE / "research_records/bad"]:
        with pytest.raises(ValueError, match="outside"):
            protect_output(path, [])
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="existing run"):
        protect_output(tmp_path / "bad", [])


def test_missing_dependency_manifest_rejected(tmp_path, monkeypatch):
    import gut_meal_bridge_v2.inputs as module
    copied = tmp_path / "gut_meal_bridge_v2"
    shutil.copytree(MODULE, copied, ignore=shutil.ignore_patterns("__pycache__"))
    path = copied / "MANIFEST.json"
    body = json.loads(path.read_text())
    body["dependencies"] = {}
    write_json(path, body)
    monkeypatch.setattr(module, "MODULE", copied)
    with pytest.raises(ValueError, match="Incomplete dependency"):
        bundle()


def test_clean_upload_launch_from_other_directory(tmp_path):
    files = fixture(tmp_path)
    uploaded = tmp_path / "uploaded project"
    for folder in ["gut_meal_bridge_v2", "hppmeal", "scripts", "seven_diet_meals"]:
        shutil.copytree(PACKAGE / folder, uploaded / folder, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    env = os.environ.copy()
    env["PYTHON_BIN"] = os.sys.executable
    out = tmp_path / "uploaded output"
    process = subprocess.run(["bash", str(uploaded / "gut_meal_bridge_v2/run_gut_meal_bridge_v2.sh")] + run_args(files, out),
        cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "COMPLETE_WITH_LIMITATIONS" in process.stdout
    # The upload launcher must also fail clearly if one required file was omitted.
    (uploaded / "gut_meal_bridge_v2/CONTRACT.json").unlink()
    process = subprocess.run(["bash", str(uploaded / "gut_meal_bridge_v2/run_gut_meal_bridge_v2.sh")] + run_args(files, out),
        cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert process.returncode == 2 and "Missing/changed upload" in process.stderr
