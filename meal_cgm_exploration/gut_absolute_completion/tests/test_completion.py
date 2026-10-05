"""Synthetic end-to-end and independent numerical checks; no real HPP fits."""
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
from hppmeal.common import sha256
from seven_diet_meals.models import encode
from gut_absolute_robustness.tests.test_analysis import make_source
from gut_absolute_robustness.run_analysis import main as previous_main
from gut_absolute_robustness.inputs import selected
from gut_absolute_robustness.analysis import common_columns
from gut_absolute_completion.inputs import (MODULE, PACKAGE, DEP_FILES, contract, load_inputs, NUMERIC)
from gut_absolute_completion import analysis
from gut_absolute_completion.analysis import complete, panel_fdr
from gut_absolute_completion import run_analysis as runner
from gut_absolute_completion.reporting import show
from gut_absolute_completion.figures import limits


def make_runs(root, varied=True):
    source = make_source(root / "fixture", varied=varied)
    target = root / "previous"
    with redirect_stdout(io.StringIO()):
        code = previous_main(["--source", str(source), "--output-root", str(target)])
    assert code == (0 if varied else 2)
    return source, next(target.iterdir())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return make_runs(tmp_path_factory.mktemp("absolute_complete_source"))


@pytest.fixture
def data(runs):
    return load_inputs(*runs)


def test_exactly_32_new_fits_and_40_verified_references(data, monkeypatch):
    calls = []
    original = analysis.estimate_common
    def counted(p, m, setting, taxon, outcome):
        calls.append((setting, taxon, outcome))
        return original(p, m, setting, taxon, outcome)
    monkeypatch.setattr(analysis, "estimate_common", counted)
    common, identities, reuse = complete(data, lambda _: None)
    assert len(calls) == 32 and all(x[2] != "baseline_glucose" for x in calls)
    assert len(common) == 72 and len(reuse) == 40 and common.status.eq("ESTIMATED").all()
    assert len(identities) == 24 and identities.status.eq("PASS_ALGEBRA_ONLY").all()
    assert common.estimate_origin.value_counts().to_dict() == {"NEW_FIT": 32, "previous_a": 24, "previous_b": 16}
    for row in common.loc[common.estimate_origin.ne("NEW_FIT")].itertuples():
        old = data[row.estimate_origin]
        r = old.loc[old.setting.eq(row.setting) & old.threshold.eq(row.threshold) & old.taxon.eq(row.taxon) & old.outcome.eq(row.outcome)].iloc[0]
        for field in NUMERIC:
            assert getattr(row, field) == r[field]
    merged = common.loc[common.threshold.ne("m5d3") & common.outcome.eq("baseline_glucose")].merge(data["previous_b"], on=["setting", "threshold", "taxon", "outcome"], suffixes=("_new", "_old"))
    assert (np.abs(merged.q_panel12 - merged.q_family) > 1e-8).any()


def test_new_absolute_hc3_matches_independent_matrix(data):
    common, _, _ = complete(data, lambda _: None)
    p, _ = selected(data["persons"], data["members"], "strict_common", 20, 7)
    row = common.loc[common.setting.eq("strict_common") & common.threshold.eq("m20d7") & common.taxon.eq("T046") & common.outcome.eq("absolute_post_mean")].iloc[0]
    X = np.column_stack([np.ones(len(p)), encode(p, common_columns("T046"))])
    X = X[:, np.r_[True, X[:, 1:].std(axis=0) > 1e-10]]
    y = p.absolute_post_mean.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bread = np.linalg.inv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    scores = X * ((y - X @ beta) / (1 - leverage))[:, None]
    covariance = bread @ scores.T @ scores @ bread
    se = np.sqrt(covariance[1, 1])
    df = len(p) - X.shape[1]
    critical = stats.t.ppf(.975, df)
    np.testing.assert_allclose(row[NUMERIC].to_numpy(float), [beta[1], se, beta[1] - critical * se, beta[1] + critical * se, 2 * stats.t.sf(abs(beta[1] / se), df)], rtol=1e-7, atol=1e-8)


def test_planned_fdr_keeps_ne_and_groups_separate(data):
    d, _, _ = complete(data, lambda _: None)
    for _, ix in d.groupby(["setting", "threshold"]).groups.items():
        d.loc[ix, "p"] = [.01] + [.5] * 10 + [np.nan]
    corrected = panel_fdr(d)
    for _, g in corrected.groupby(["setting", "threshold"]):
        assert np.isclose(g.q_panel12.iloc[0], .12) and pd.isna(g.q_panel12.iloc[-1])
    with pytest.raises(ValueError, match="Incomplete"):
        panel_fdr(d.iloc[1:])


def repin(previous, name, transform):
    path = previous / name
    frame = pd.read_csv(path, dtype={"participant_id": str}, float_precision="round_trip")
    transform(frame)
    frame.to_csv(path, index=False, float_format="%.17g")
    m = json.loads((previous / "manifest.json").read_text())
    m["result_sha256"][name] = sha256(path)
    (previous / "manifest.json").write_text(json.dumps(m))


@pytest.mark.parametrize("problem", ["file", "source", "kind", "q", "members", "count", "status"])
def test_changed_or_mismatched_previous_run_rejected(runs, tmp_path, problem):
    source, old = runs
    previous = tmp_path / "previous"
    shutil.copytree(old, previous)
    manifest = previous / "manifest.json"
    state = json.loads(manifest.read_text())
    if problem == "file":
        with (previous / "tables/absolute_models.csv").open("a") as f: f.write("\n")
    elif problem in ["source", "kind", "status"]:
        if problem == "source": state["input_sha256"][str(source / "private/person_inputs.csv")] = "different"
        if problem == "kind": state["data_kind"] = "REAL"
        if problem == "status": state["status"] = "FAILED"
        manifest.write_text(json.dumps(state))
    elif problem == "q":
        repin(previous, "tables/absolute_models.csv", lambda d: d.loc.__setitem__((0, "q_panel12"), .999))
    elif problem == "members":
        repin(previous, "private/participant_membership.csv", lambda d: d.loc.__setitem__((0, "participant_id"), "not_in_source"))
    elif problem == "count":
        repin(previous, "tables/support_models.csv", lambda d: d.loc.__setitem__((0, "people"), 1))
    with pytest.raises(ValueError): load_inputs(source, previous)


def test_moving_both_runs_preserves_content_link(runs, tmp_path):
    source, previous = runs
    shutil.copytree(source, tmp_path / "source")
    shutil.copytree(previous, tmp_path / "previous")
    data = load_inputs(tmp_path / "source", tmp_path / "previous")
    assert data["data_kind"] == "SYNTHETIC" and len(data["previous_b"]) == 72


def test_end_to_end_tables_figures_three_pages_and_read_only_sources(runs, tmp_path, capsys):
    source, previous = runs
    before = {str(f): sha256(f) for root in runs for f in root.rglob("*") if f.is_file()}
    target = tmp_path / "new"
    assert runner.main(["--source", str(source), "--previous", str(previous), "--output-root", str(target)]) == 0
    out = next(target.iterdir())
    state = json.loads((out / "manifest.json").read_text())
    assert state["common_estimated"] == 72 and state["new_estimated"] == 32 and state["algebra_pass"] == 24
    assert state["figures_status"] == "COMPLETE_2_PNG_PDF_SVG" and state["sources_unchanged"] == "PASS"
    for extension in ["png", "pdf", "svg"]:
        files = list((out / "figures").glob("*." + extension))
        assert len(files) == 2 and all(f.stat().st_size > 1000 for f in files)
    for svg in (out / "figures").glob("*.svg"):
        assert "SYNTHETIC DATA" in svg.read_text() and "T023" in svg.read_text()
    assert len(pd.read_csv(out / "tables/all_meal_models.csv")) == 144
    assert (out / "private/participant_membership.csv").is_file()
    for file in (out / "tables").glob("*.csv"):
        assert not {"participant_id", "meal_id", "person_sha256", "meal_sha256"} & set(pd.read_csv(file).columns)
    for file in (out / "pages").glob("*.txt"):
        text = file.read_text()
        assert "SYNTHETIC" in text and "000001" not in text
        assert len(text.splitlines()) <= 38 and max(map(len, text.splitlines())) <= 160
    assert all(sha256(Path(f)) == h for f, h in before.items())
    assert runner.main(["--show", "latest", "--output-root", str(target), "--page", "2"]) == 0
    capsys.readouterr()
    with (out / "pages/02_summary.txt").open("a") as f: f.write("changed")
    with pytest.raises(ValueError, match="changed page"): show(out, target, 2)


def test_empty_high_support_retains_all_planned_ne(tmp_path, capsys):
    source, previous = make_runs(tmp_path / "low", varied=False)
    d = load_inputs(source, previous)
    common, identity, audit = complete(d, lambda _: None)
    high = common.loc[common.threshold.ne("m5d3")]
    assert len(high) == 48 and high.status.eq("NOT_ESTIMATED").all() and high.q_panel12.isna().all()
    assert len(audit) == 40 and identity.status.eq("NOT_EVALUABLE").sum() == 16
    assert limits(high) == (-1., 1.)
    target = tmp_path / "partial"
    assert runner.main(["--source", str(source), "--previous", str(previous), "--output-root", str(target)]) == 2
    out = next(target.iterdir())
    state = json.loads((out / "manifest.json").read_text())
    assert state["status"] == "PARTIAL" and state["common_estimated"] == 24 and state["new_estimated"] == 0
    assert state["figures_status"] == "COMPLETE_2_PNG_PDF_SVG"
    assert len(pd.read_csv(out / "tables/not_estimated.csv")) == 96
    assert "NE" in (out / "figures/Figure1_common_adjustment.svg").read_text()


def test_plot_dependency_failure_preserves_analysis(runs, tmp_path, monkeypatch, capsys):
    def unavailable(*args): raise ImportError("Synthetic missing plotting dependency")
    monkeypatch.setattr(runner, "make_figures", unavailable)
    source, previous = runs
    target = tmp_path / "noplot"
    assert runner.main(["--source", str(source), "--previous", str(previous), "--output-root", str(target)]) == 2
    out = next(target.iterdir())
    state = json.loads((out / "manifest.json").read_text())
    assert state["status"] == "COMPLETE_ANALYSIS_FIGURES_UNAVAILABLE" and state["common_estimated"] == 72
    assert (out / "FIGURES_ERROR.txt").is_file() and len(list((out / "pages").glob("*.txt"))) == 3
    assert len(pd.read_csv(out / "tables/common_models.csv")) == 72


def test_protection_and_latest_failure(runs, tmp_path):
    source, previous = runs
    for root in [source, source / "nested", previous / "nested"]:
        with pytest.raises(ValueError): runner.protect_output(source, previous, root)
    failed = tmp_path / "gut_absolute_completion_99999999"
    failed.mkdir()
    (failed / "manifest.json").write_text(json.dumps(dict(status="FAILED", data_kind="UNVERIFIED", error="Synthetic failure")))
    with pytest.raises(ValueError, match="Synthetic failure"): show("latest", tmp_path)


def test_python38_and_isolated_pheno_upload(runs, tmp_path):
    for f in MODULE.glob("*.py"):
        ast.parse(f.read_text(), feature_version=(3, 8))
    deployed = tmp_path / "upload" / "meal_cgm_exploration"
    deployed.mkdir(parents=True)
    shutil.copytree(MODULE, deployed / MODULE.name, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in DEP_FILES:
        target = deployed / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE / name, target)
    shutil.copytree(runs[0], deployed / "outputs" / contract()["original_run"])
    shutil.copytree(runs[1], deployed / "outputs" / contract()["previous_run"])
    pheno = Path("/opt/anaconda3/envs/pheno/bin/python")
    executable = str(pheno) if pheno.is_file() else sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([executable, str(deployed / MODULE.name / "run_analysis.py")], cwd=tmp_path,
                            capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("KEY ") == 3 and "000001" not in result.stdout
    assert not (deployed / "window_qc_followup").exists()
    assert not (deployed / "window_qc_diagnosis").exists()
