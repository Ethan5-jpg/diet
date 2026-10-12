import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from daily.build import ingest_cgm, ingest_diet, diet_day
from daily.common import sha256
from daily.models import fit_all

PACKAGE = Path(__file__).resolve().parents[1]


def run(package, *args):
    return subprocess.run([sys.executable, str(package / "run_analysis.py")] + list(args),
                          capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("relocated")
    package = root / "different_server_name" / "Data" / "diet_gut_cgm_interaction" / "daily_within_person"
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "outputs", "tests"))
    out = root / "fictional_build"
    result = run(package, "--demo", "--stage", "build", "--output-dir", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    return package, out


def test_cross_chunk_duplicate_is_flagged(config, tmp_path):
    config["chunksize"] = 1
    rows = []
    for i, hour in enumerate([7, 12, 18, 7]):
        rows.append(dict(participant_id="001", cohort="10k", research_stage="00_00_visit",
             collection_timestamp="2024-01-01 {:02d}:00:00+02:00".format(hour), collection_date="2024-01-01",
             food_id=str(hour), calories_kcal=600, carbohydrate_g=60, lipid_g=20, protein_g=20,
             dietary_fiber_g=5, alcohol_g=0, timezone="asia/jerusalem"))
    p = tmp_path / "food.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bad_people (participant_id TEXT, source TEXT)")
    ingest_diet(con, p, {"7": 4, "12": 1, "18": 1}, {"001": "c"}, config)
    d = pd.read_sql_query("SELECT * FROM food", con)
    assert d.participant_id.unique().tolist() == ["001"]
    assert "SUSPECT_DUPLICATE_FOOD" in diet_day(d, config)["diet_reasons"]
    con.close()


@pytest.mark.parametrize("chunk_size", [1, 100])
def test_other_cgm_connections_can_empty_whole_files_and_chunks(config, tmp_path, chunk_size):
    config["chunksize"] = chunk_size
    def row(connection, timestamp):
        return dict(participant_id="001", cohort=config["cohort"], research_stage=config["research_stage"],
                    connection_id=connection, collection_timestamp=timestamp, glucose="100")
    # A selected person can have an entire CSV for a connection that was not
    # chosen by the frozen final table. Its bad timestamp must not taint them.
    blocks = [[row("other", "2024-01-01T12:00:00+02:00"), row("other", "invalid")],
              [row("chosen", "2024-01-01T12:00:00+02:00"),
               row("other", "invalid"), row("chosen", "2024-01-01T12:15:00+02:00")]]
    files = []
    for i, rows in enumerate(blocks):
        path = tmp_path / ("sensor_{}.csv".format(i))
        pd.DataFrame(rows).to_csv(path, index=False)
        files.append(path)
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE bad_people (participant_id TEXT, source TEXT)")
        counts = ingest_cgm(con, files, {"001": "chosen"}, config)
        assert counts["input_rows"] == 5
        assert counts["other_connection_rows"] == 3
        assert counts["cached_rows"] == 2
        assert counts["unresolved_time_or_timezone"] == 0
        saved = pd.read_sql_query("SELECT * FROM cgm ORDER BY t", con)
        assert saved.connection_id.eq("chosen").all()
        assert saved.t.tolist() == [1704103200., 1704104100.]
        assert con.execute("SELECT COUNT(*) FROM bad_people").fetchone()[0] == 0
    finally:
        con.close()


def test_relocated_standalone_build_and_resume(built):
    package, out = built
    before = json.loads((out / "manifest.json").read_text())
    assert before["status"] == "BUILD_COMPLETE"
    assert before["data_kind"] == "SYNTHETIC"
    assert before["build_audit"]["matched_days"] == 960
    result = run(package, "--stage", "analyze", "--run-dir", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"]
    assert m["model_summary"] == dict(estimated=84, planned=84, primary_sd_estimated=4)
    d = pd.read_csv(out / "tables/10_interactions.csv")
    assert len(d) == 84
    assert d.groupby("spec").size().eq(12).all()
    assert d.groupby("spec").q_primary4.count().eq(4).all()
    focal = d.loc[d.spec.eq("primary") & d.exposure.eq("NOVA4") & d.taxon.eq("T068") & d.outcome.eq("SD")].iloc[0]
    assert focal.beta == pytest.approx(-.3, abs=.05)
    for i in [1,2,3]: assert "SYNTHETIC" in (out / ("SCREEN_{}.txt".format(i))).read_text()
    for name in ["SCREEN_1.txt", "SCREEN_2.txt", "SCREEN_3.txt", "SUMMARY.md"]:
        assert "SYNTH_000" not in (out / name).read_text()
    again = run(package, "--stage", "analyze", "--run-dir", str(out))
    assert again.returncode == 2
    assert json.loads((out / "manifest.json").read_text())["status"] == m["status"]
    new = run(package, "--demo", "--output-dir", str(out))
    assert new.returncode == 2


def test_missing_fiber_preserves_planned_family(built):
    package, out = built
    from run_analysis import load_saved_panel
    d = load_saved_panel(out / "private/daily_panel.csv")
    c = json.loads((out / "analysis_config.json").read_text())
    d["fiber_density"] = np.nan
    results, coefficients, flows, slopes, anchors = fit_all(d, c)
    assert len(results) == 84
    assert results.loc[results.exposure.eq("Fiber"), "status"].eq("NOT_ESTIMATED").all()
    assert results.loc[results.exposure.eq("Fiber"), "p"].isna().all()
    primary = results.loc[results.spec.eq("primary")]
    from daily.models import bh
    np.testing.assert_allclose(primary.q_all12.to_numpy(), bh(primary.p.to_numpy()), equal_nan=True)


def test_changed_input_refuses_resume_and_preserves_build(built, tmp_path):
    package, out = built
    # Copy a real completed build artifact, then restore its verified build status
    # solely inside this isolated synthetic test; no actual analysis run is edited.
    copied = tmp_path / "changed_build"
    shutil.copytree(out, copied)
    m = json.loads((copied / "manifest.json").read_text())
    m["status"] = "BUILD_COMPLETE"
    source = next(r for r in m["inputs"] if r["label"] == "nova_mapping")
    original = Path(source["path"])
    saved = original.read_bytes()
    try:
        original.write_bytes(saved + b"\n")
        (copied / "manifest.json").write_text(json.dumps(m))
        r = run(package, "--stage", "analyze", "--run-dir", str(copied))
        assert r.returncode == 2
        assert "Input content changed" in r.stderr
        assert json.loads((copied / "manifest.json").read_text())["status"] == "BUILD_COMPLETE"
    finally:
        original.write_bytes(saved)


def test_display_refuses_modified_output(built, tmp_path):
    package, out = built
    copied = tmp_path / "changed_display"
    shutil.copytree(out, copied)
    with (copied / "SCREEN_2.txt").open("a") as f: f.write("altered\n")
    r = subprocess.run([sys.executable, str(package / "show_results.py"), "--run-dir", str(copied)], capture_output=True, text=True)
    assert r.returncode == 2
    assert "hash mismatch" in r.stderr


def test_default_data_root_in_relocated_upload(built, tmp_path):
    package, out = built
    m = json.loads((out / "manifest.json").read_text())
    synthetic_root = Path(m["data_root"])
    expected_root = package.parent.parent
    for row in m["inputs"]:
        source = Path(row["path"])
        target = expected_root / source.relative_to(synthetic_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.symlink_to(source)
    checked = tmp_path / "preflight"
    result = run(package, "--stage", "check", "--output-dir", str(checked))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((checked / "manifest.json").read_text())
    assert report["data_root"] == str(expected_root)
    assert report["status"] == "PREFLIGHT_PASSED"


def test_saved_panel_preserves_literal_identity_categories_and_precision(tmp_path):
    from run_analysis import load_saved_panel
    p = tmp_path / "panel.csv"
    d = pd.DataFrame(dict(participant_id=["NA", "001"], connection_id=["NA", "0001"],
                          date=["2024-01-01"] * 2, sex=["NA", "01"], education_level=["01", "02"],
                          sample_minus_day=[np.nan, 3.], dietary_value=[.12345678901234568, np.nan]))
    for col in ["diet_pass", "cgm_pass", "cgm_strict", "primary_eligible", "strict_eligible", "exclusion_unknown", "has_covariate_record"]:
        d[col] = [True, False]
    d.to_csv(p, index=False)
    restored = load_saved_panel(p)
    for c in ["participant_id", "connection_id", "sex", "education_level"]:
        assert restored[c].tolist() == d[c].tolist()
    assert restored.dietary_value.iloc[0] == d.dietary_value.iloc[0]
    assert np.isnan(restored.dietary_value.iloc[1])
    assert np.isnan(restored.sample_minus_day.iloc[0])
    assert restored.cgm_pass.tolist() == [True, False]


def test_no_overlapping_days_reports_feasibility_and_all_planned_rows(monkeypatch, tmp_path):
    from daily import synthetic
    from run_analysis import main
    original = synthetic.create_fixture
    def disjoint_fixture(root, config):
        info = original(root, config, n_people=4, n_days=6)
        path = root / config["inputs"]["diet"]
        d = pd.read_csv(path)
        t = pd.to_datetime(d.collection_timestamp, utc=True) + pd.Timedelta(days=365)
        local = t.dt.tz_convert(config["time"]["local_timezone"])
        d["collection_timestamp"] = local.map(lambda t: t.isoformat())
        d["collection_date"] = local.dt.strftime("%Y-%m-%d")
        d.to_csv(path, index=False)
        return info
    monkeypatch.setattr(synthetic, "create_fixture", disjoint_fixture)
    out = tmp_path / "no_overlap"
    assert main(["--demo", "--stage", "all", "--output-dir", str(out)]) == 2
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "NO_ESTIMABLE_PRIMARY_MODELS"
    assert m["build_audit"]["matched_days"] == 0
    assert m["build_audit"]["diet_only_days"] == 24
    d = pd.read_csv(out / "tables/10_interactions.csv")
    assert len(d) == 84 and d.status.eq("NOT_ESTIMATED").all()
    assert d.p.isna().all() and d.q_all12.isna().all()
    assert all((out / ("SCREEN_{}.txt".format(n))).is_file() for n in [1, 2, 3])


def test_contrast_failure_cannot_leave_partial_significance(built, monkeypatch):
    from daily import models
    from run_analysis import load_saved_panel
    package, out = built
    d = load_saved_panel(out / "private/daily_panel.csv")
    c = json.loads((out / "analysis_config.json").read_text())
    def fail(fit, weights):
        raise ValueError("Injected contrast failure")
    monkeypatch.setattr(models, "linear_contrast", fail)
    results, coefficients, flows, slopes, anchors = fit_all(d, c)
    primary = results.loc[results.spec.eq("primary")]
    assert primary.status.eq("NOT_ESTIMATED").all()
    assert primary[["beta", "se", "p", "q_primary4", "q_all12"]].isna().all().all()
    assert coefficients.spec.ne("primary").all()
    assert slopes.empty


def test_all_at_once_matches_split_build_analyze(built, tmp_path):
    package, out = built
    new = tmp_path / "all_at_once"
    result = run(package, "--demo", "--stage", "all", "--output-dir", str(new))
    assert result.returncode == 0, result.stdout + result.stderr
    a = pd.read_csv(out / "tables/10_interactions.csv")
    b = pd.read_csv(new / "tables/10_interactions.csv")
    pd.testing.assert_frame_equal(a, b, rtol=1e-12, atol=1e-12)
