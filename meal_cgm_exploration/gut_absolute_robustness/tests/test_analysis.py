"""Synthetic provenance, numerical and upload regression checks; no real HPP fits."""
import ast
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hppmeal.common import sha256, read_csv
from gut_meal_bridge_v2.tests.test_bridge import fixture as original_fixture, replace_synthetic_meals, run_args
from gut_meal_bridge_v2.run_analysis import main as original_main
from gut_absolute_robustness.inputs import (MODULE, PACKAGE, DEP_FILES, load_source, selected,
    validate_frames, contract)
from gut_absolute_robustness.analysis import (analyze, reproduce, common_columns, estimate_common,
    identity_checks, absolute_fdr, sensitivity_fdr, support_tables, selection_balance)
from gut_absolute_robustness.reporting import show, q_number, pages
from gut_absolute_robustness.run_analysis import main, protect_output
from seven_diet_meals.models import encode, linear_fit


def make_source(tmp_path, varied=True):
    files = original_fixture(tmp_path, n=240)
    if varied:
        raw = read_csv(files[1] / "private/02_meals.csv")
        pieces = []
        for pid, g in raw.groupby("participant_id", sort=True):
            group = int(pid) % 3
            count, dates = [(6, 3), (12, 5), (24, 8)][group]
            d = pd.concat([g, g], ignore_index=True).iloc[:count].copy()
            day = np.floor(np.arange(count) * dates / count).astype(int)
            d["meal_id"] = [pid + "_support_" + str(i) for i in range(count)]
            d["local_date"] = ["2024-01-{:02d}".format(1 + x) for x in day]
            d["start"] = 1704067200 + day * 86400 + pd.to_numeric(d.local_hour).to_numpy() * 3600
            pieces.append(d)
        replace_synthetic_meals(files, pd.concat(pieces, ignore_index=True))
    target = tmp_path / "gut_source"
    with redirect_stdout(io.StringIO()):
        assert original_main(run_args(files, target)) == 0
    return next(target.iterdir())


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_source(tmp_path_factory.mktemp("gut_support_source"))


def source_frames(source):
    return load_source(source)[:4]


def test_original_source_reproduces_all_48_and_preserves_ids(source):
    p, m, old, _ = source_frames(source)
    assert p.participant_id.str.startswith("000").all()
    assert set(p.education_level) == {"1", "2", "3"}
    assert len(reproduce(p, m, old, lambda _: None)) == 48
    assert (p.absolute_post_mean == p.baseline_glucose + p.mean_increment).all()


def test_thresholds_include_boundaries_and_filter_people_only(source):
    p, m, _, _ = source_frames(source)
    last = set(p.participant_id)
    for meals, dates in [(5, 3), (10, 5), (20, 7)]:
        keep, member = selected(p, m, "primary_common", meals, dates)
        assert set(keep.participant_id).issubset(last)
        last = set(keep.participant_id)
        assert keep.meals.ge(meals).all() and keep.dates.ge(dates).all()
        baseline = p.set_index("participant_id").loc[keep.participant_id].reset_index()
        pd.testing.assert_frame_equal(keep, baseline, check_exact=True)
        assert len(member) == keep.meals.sum()
    # Explicit inclusivity at both boundaries.
    a = p.head(2).copy()
    a.loc[a.index[0], ["meals", "dates"]] = [10, 5]
    a.loc[a.index[1], ["meals", "dates"]] = [10, 4]
    keep, _ = selected(a, m, "primary_common", 10, 5)
    assert list(keep.participant_id) == [a.iloc[0].participant_id]


def test_strict_nested_and_means_not_restandardized(source):
    p, m, _, _ = source_frames(source)
    for t in contract()["thresholds"]:
        primary, _ = selected(p, m, "primary_common", t["meals"], t["dates"])
        strict, _ = selected(p, m, "strict_common", t["meals"], t["dates"])
        assert len(strict) < len(primary) and set(strict.participant_id).issubset(primary.participant_id)
        for field in ["T023", "T046", "T068", "T096", "carb_10_squared", "mean_increment"]:
            pd.testing.assert_series_equal(strict.set_index("participant_id")[field], p.set_index("participant_id").loc[strict.participant_id, field], check_exact=True)


@pytest.mark.parametrize("problem", ["duplicate", "meal_count", "strict", "missing", "membership", "bad_flag"])
def test_invalid_common_inputs_rejected(source, problem):
    p, m, old, sample = source_frames(source)
    if problem == "duplicate": p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    if problem == "meal_count": p.loc[0, "meals"] += 1
    if problem == "strict": m.loc[0, "strict_eligible"] = not m.loc[0, "strict_eligible"]
    if problem == "missing": p.loc[0, "T068"] = np.nan
    if problem == "membership": old.loc[0, "person_sha256"] = "different"
    if problem == "bad_flag": p["strict_eligible"] = "unknown"
    with pytest.raises(ValueError): validate_frames(p, m, old, sample)


def test_changed_private_input_is_rejected(source, tmp_path):
    copied = tmp_path / "original"
    shutil.copytree(source, copied)
    with (copied / "private/person_inputs.csv").open("a") as f: f.write("\n")
    with pytest.raises(ValueError, match="changed original result"): load_source(copied)


def test_common_absolute_hc3_matches_independent_matrix_reference(source):
    p, m, _, _ = source_frames(source)
    row = estimate_common(p, m, "primary_common", "T046", "absolute_post_mean")
    X = np.column_stack([np.ones(len(p)), encode(p, common_columns("T046"))])
    # Original OLS discards only constants; match that identifiable design.
    active = np.r_[True, X[:, 1:].std(axis=0) > 1e-10]
    X = X[:, active]
    y = p.absolute_post_mean.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bread = np.linalg.inv(X.T @ X)
    residual = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    scores = X * (residual / (1 - leverage))[:, None]
    covariance = bread @ scores.T @ scores @ bread
    se = np.sqrt(covariance[1, 1])
    critical = stats.t.ppf(.975, len(p) - X.shape[1])
    np.testing.assert_allclose([row["estimate"], row["se"], row["ci_low"], row["ci_high"]],
                               [beta[1], se, beta[1] - critical * se, beta[1] + critical * se], rtol=1e-7, atol=1e-8)


def test_baseline_adjusted_absolute_is_not_an_independent_finding(source):
    p, _, _, _ = source_frames(source)
    cols = common_columns("T046") + ["baseline_glucose", "pre_slope_per_min"]
    a = linear_fit(p.absolute_post_mean, encode(p, cols), "T046")[0][1]
    b = linear_fit(p.mean_increment, encode(p, cols), "T046")[0][1]
    for field in ["estimate", "se", "ci_low", "ci_high", "p"]:
        assert np.isclose(a[field], b[field], rtol=1e-7, atol=1e-8)


def test_complete_design_identity_and_fdr_ne_denominators(source):
    p, m, old, _ = source_frames(source)
    result = analyze(p, m, old, lambda _: None)
    a, b = result["absolute"], result["sensitivity"]
    assert len(a) == 24 and len(b) == 72 and a.status.eq("ESTIMATED").all() and b.status.eq("ESTIMATED").all()
    assert result["identity"].status.eq("PASS_ALGEBRA_ONLY").all()
    assert np.max(np.abs(result["identity"].coefficient_identity_residual)) < 1e-8
    # Independent known BH ranks with one NE included as p=1 in 12 planned tests.
    for _, ix in a.groupby("setting").groups.items():
        a.loc[ix, "p"] = [.01] + [.5] * 10 + [np.nan]
    q = absolute_fdr(a)
    assert np.allclose(q.groupby("setting").head(1).q_panel12, .12)
    assert q.q_panel12.isna().sum() == 2
    for _, ix in b.groupby(["setting", "threshold", "family"]).groups.items():
        b.loc[ix, "p"] = [.01] + [.5] * (len(ix) - 2) + [np.nan]
    q = sensitivity_fdr(b)
    assert np.allclose(q.loc[q.family.eq("net_primary")].groupby(["setting", "threshold"]).head(1).q_family, .04)
    assert q.q_family.isna().sum() == 12


def test_identity_failure_rejected_and_ne_not_fabricated(source):
    p, m, old, _ = source_frames(source)
    a = analyze(p, m, old, lambda _: None)["absolute"]
    ix = a.index[a.outcome.eq("absolute_post_mean")][0]
    a.loc[ix, "estimate"] += 1
    with pytest.raises(ValueError, match="identity failed"): identity_checks(a)
    a.loc[ix, "status"] = "NOT_ESTIMATED"
    a.loc[ix, ["estimate", "se", "ci_low", "ci_high", "p"]] = np.nan
    assert identity_checks(a).status.eq("NOT_EVALUABLE").sum() == 1


def test_screenshot_q_preserves_side_of_point05():
    for q in [.0004, .04999, .05001, .05, np.nextafter(.05, 0), np.nextafter(.05, 1), 1.]:
        shown = float(q_number(q))
        assert (shown < .05) == (q < .05)
        assert (shown == .05) == (q == .05)
    assert q_number(0) == "0*" and q_number(np.nan) == "NA"


def test_zero_variance_separation_is_flagged_in_summary(source):
    p, m, old, _ = source_frames(source)
    example = p.head(4).copy()
    example["sex"] = ["A", "A", "B", "B"]
    example["bmi"] = 22.5
    balance = pd.DataFrame(selection_balance(example, example.iloc[:2], "primary_common", "m10d5"))
    indexed = balance.set_index("field")
    assert indexed.loc["sex=A", "smd_status"] == "ZERO_VARIANCE_SEPARATION"
    assert pd.isna(indexed.loc["sex=A", "standardized_difference"])
    assert indexed.loc["bmi", "smd_status"] == "IDENTICAL_CONSTANTS"
    assert pd.isna(indexed.loc["bmi", "standardized_difference"])
    missing = pd.DataFrame(selection_balance(example, example, "primary_common", "m10d5"))
    assert missing.smd_status.eq("INSUFFICIENT_GROUP_SIZE").all()
    flow, original_balance, _ = support_tables(p, m)
    result = analyze(p, m, old, lambda _: None)
    other = original_balance.loc[~(original_balance.setting.eq("primary_common") & original_balance.threshold.eq("m10d5"))]
    tables = dict(support_flow=flow, selection_balance=pd.concat([other, balance], ignore_index=True),
                  absolute_models=result["absolute"], support_models=result["sensitivity"], algebra_check=result["identity"])
    rendered = pages("synthetic_review", dict(data_kind="SYNTHETIC", status="COMPLETE_WITH_LIMITATIONS"), tables)
    assert "sex=A=零方差完全分离" in rendered[0]


def test_absolute_mean_matches_curve_integral_with_negative_net_area():
    from hppmeal.cgm import response
    cfg = json.loads((PACKAGE / "config/default.json").read_text())["cgm"]
    minutes = np.arange(-30, 121, 15, dtype=float)
    glucose = np.where(minutes < 0, 100., 100. - minutes / 6.)
    result, curve = response(minutes * 60, glucose, 0., cfg)
    assert result["cgm_valid"] and result["iauc_net_0_120"] < 0
    x, y, _ = curve
    direct_absolute = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.) / 120.
    reconstructed = result["baseline_glucose"] + result["iauc_net_0_120"] / 120.
    assert np.isclose(reconstructed, direct_absolute) and np.isclose(direct_absolute, 90.)


@pytest.mark.parametrize("problem", ["small", "constant_taxon", "constant_outcome", "singular"])
def test_nonempty_unestimable_groups_keep_missing_effects(source, problem):
    p, m, _, _ = source_frames(source)
    if problem == "small": p = p.head(49).copy()
    if problem == "constant_taxon": p["T046"] = 0.
    if problem == "constant_outcome": p["absolute_post_mean"] = 100.
    if problem == "singular": p["age_years"] = p.bmi
    m = m.loc[m.participant_id.isin(p.participant_id)]
    row = estimate_common(p, m, "primary_common", "T046", "absolute_post_mean")
    assert row["status"] == "NOT_ESTIMATED" and row["reason"]
    assert all(pd.isna(row[field]) for field in ["estimate", "se", "ci_low", "ci_high", "p"])


def test_end_to_end_three_pages_source_unchanged_and_private_ids(source, tmp_path, capsys):
    before = {str(f): sha256(f) for f in source.rglob("*") if f.is_file()}
    outroot = tmp_path / "new"
    assert main(["--source", str(source), "--output-root", str(outroot)]) == 0
    out = next(outroot.iterdir())
    state = json.loads((out / "manifest.json").read_text())
    assert state["status"] == "COMPLETE_WITH_LIMITATIONS" and state["data_kind"] == "SYNTHETIC"
    assert state["source_reproduction"] == "PASS_48" and state["absolute_estimated"] == 24 and state["support_new_estimated"] == 48
    pages = list((out / "pages").glob("*.txt"))
    assert len(pages) == 3
    for page in pages:
        text = page.read_text()
        assert "SYNTHETIC" in text and "000001" not in text and "participant_id" not in text
        assert len(text.splitlines()) <= 38 and max(map(len, text.splitlines())) <= 160
    selection_lines = (out / "pages/01_summary.txt").read_text().splitlines()
    assert any("不含餐数/日期数" in line for line in selection_lines)
    assert not any(" meals=" in line or " dates=" in line for line in selection_lines)
    assert (out / "private/participant_membership.csv").is_file()
    for f in (out / "tables").glob("*.csv"):
        assert "participant_id" not in pd.read_csv(f).columns
    assert all(sha256(Path(f)) == h for f, h in before.items())
    assert main(["--show", "latest", "--output-root", str(outroot), "--page", "2"]) == 0
    capsys.readouterr()
    with (out / "pages/02_summary.txt").open("a") as f: f.write("changed")
    with pytest.raises(ValueError, match="changed summary"): show(out, outroot, 2)


def test_partial_groups_not_relaxed_or_dropped(tmp_path):
    source = make_source(tmp_path, varied=False)  # 12 meals, only 4 dates: no 10/5 or 20/7 people.
    root = tmp_path / "partial"
    assert main(["--source", str(source), "--output-root", str(root)]) == 2
    out = next(root.iterdir())
    m = json.loads((out / "manifest.json").read_text())
    d = pd.read_csv(out / "tables/support_models.csv")
    assert m["status"] == "PARTIAL" and m["support_new_estimated"] == 0
    new = d.loc[d.threshold.ne("m5d3")]
    assert len(new) == 48 and new.status.eq("NOT_ESTIMATED").all() and new.q_family.isna().all()
    assert len(pd.read_csv(out / "tables/not_estimated.csv")) == 48


def test_output_protection_and_latest_failure(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    (source / "manifest.json").write_text('{}')
    with pytest.raises(ValueError): protect_output(source, source / "nested")
    with pytest.raises(ValueError): protect_output(tmp_path / "another", source / "nested")
    failed = tmp_path / "gut_absolute_robustness_99999999"
    failed.mkdir()
    (failed / "manifest.json").write_text(json.dumps(dict(data_kind="SYNTHETIC", status="FAILED", error="Synthetic failure")))
    with pytest.raises(ValueError, match="not complete"): show("latest", tmp_path)


def test_python38_and_isolated_upload_cli(source, tmp_path):
    for f in MODULE.glob("*.py"):
        ast.parse(f.read_text(), feature_version=(3, 8))
    deployed = tmp_path / "uploaded" / "meal_cgm_exploration"
    deployed.mkdir(parents=True)
    shutil.copytree(MODULE, deployed / MODULE.name, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in DEP_FILES:
        f = deployed / name
        f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE / name, f)
    original = deployed / "outputs" / contract()["source_run"]
    shutil.copytree(source, original)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    run = subprocess.run([sys.executable, str(deployed / MODULE.name / "run_analysis.py")], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("KEY ") == 3 and "SOURCE_REPRODUCTION=PASS_48" in run.stdout
    assert "000001" not in run.stdout
    assert not (deployed / "window_qc_followup").exists() and not (deployed / "window_qc_diagnosis").exists()
