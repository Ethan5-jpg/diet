"""Regression checks discovered during the second upload-readiness review."""
import copy
import json
import shutil

import pytest

from fixtures import make_data
from run_pipeline import main
from hppmeal.common import sha256, validate_config


def built_run(tmp_path, config):
    c, cfg = make_data(tmp_path / "data", config, people=2, days=2)
    c["plots"]["example_meals"] = 0
    cfg.write_text(json.dumps(c))
    args = ["--data-root", str(tmp_path / "data"), "--config", str(cfg)]
    assert main(args + ["--output-root", str(tmp_path / "out"), "--stage", "build"]) == 0
    run = next((tmp_path / "out").iterdir())
    return c, cfg, args, run


def test_added_cgm_file_rejected_on_resume(tmp_path, config):
    _, _, args, run = built_run(tmp_path, config)
    original = sha256(run / "manifest.json")
    shutil.copyfile(tmp_path / "data/cgm/00000.csv", tmp_path / "data/cgm/new.csv")
    assert main(args + ["--run-dir", str(run), "--stage", "audit"]) == 2
    assert sha256(run / "manifest.json") == original


def test_rejected_request_does_not_change_config_snapshot(tmp_path, config):
    c, cfg, args, run = built_run(tmp_path, config)
    snapshots = {p: sha256(run / p) for p in ["manifest.json", "effective_config.json", "code_fingerprint.json", "SUMMARY.md"]}
    c["time"]["consumption_time_verified"] = False
    cfg.write_text(json.dumps(c))
    assert main(args + ["--run-dir", str(run), "--stage", "build"]) == 2
    assert all(sha256(run / p) == digest for p, digest in snapshots.items())


@pytest.mark.parametrize("section,key,value", [
    ("cgm", "baseline_min_points", 1),
    ("ingest", "chunksize", 1.5),
    ("plots", "example_meals", -1),
    ("meal", "nova_min_coverage", float("nan")),
    ("analysis", "minimum_people", True),
])
def test_invalid_working_config_is_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_check_is_lightweight_and_reports_downstream_missing_files(tmp_path, config):
    c, cfg = make_data(tmp_path / "data", config, people=1, days=2)
    (tmp_path / "data/micro.csv").unlink()
    out = tmp_path / "checks"
    assert main(["--data-root", str(tmp_path / "data"), "--config", str(cfg), "--output-root", str(out), "--stage", "check"]) == 0
    run = next(out.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"]["check"]["build_ready"]
    assert not manifest["status"]["check"]["model_input_headers_ready"]
    assert not (run / "private/cache.sqlite").exists()


def test_model_review_does_not_change_completed_snapshots(tmp_path, config):
    c, cfg, args, run = built_run(tmp_path, config)
    paths = ["effective_config.json", "manifest.json", "SUMMARY.md", "code_fingerprint.json"]
    expected = {p: sha256(run / p) for p in paths}
    assert main(args + ["--run-dir", str(run), "--stage", "build"]) == 2
    assert all(sha256(run / p) == expected[p] for p in paths)


def test_malformed_covariate_csv_rejected_before_wrong_identity_merge(tmp_path):
    from hppmeal.common import read_csv
    p = tmp_path / "broken.csv"
    p.write_text('participant_id,flag\nEXTRA,123,0\n')
    with pytest.raises(ValueError, match="Malformed CSV"):
        read_csv(p)


def test_ambiguous_taxon_not_chosen_arbitrarily(config):
    from hppmeal.common import resolve_taxa
    columns = ["participant_id", "Eubacterium_rectale_SGB1", "Eubacterium_rectale_SGB2"]
    result = resolve_taxa(columns, config["analysis"]["taxa"])
    assert result[0]["status"] == "MISSING_OR_AMBIGUOUS_COLUMN"
    assert result[0]["matches"] == 2 and result[0]["column"] == ""


def test_atomic_json_retains_previous_file_when_serialization_fails(tmp_path):
    from hppmeal.common import write_json
    p = tmp_path / "manifest.json"
    write_json(p, {"valid": True})
    before = p.read_bytes()
    with pytest.raises(TypeError):
        write_json(p, {"unsupported": object()})
    assert p.read_bytes() == before


def test_nearly_collinear_design_does_not_report_unstable_effect():
    import numpy as np
    import pandas as pd
    from hppmeal.models import within_fit
    rng = np.random.default_rng(99)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame(dict(participant_id=np.repeat(np.arange(20), 10).astype(str),
                           cohort="c", research_stage="v", meal_id=np.arange(n).astype(str),
                           nova_per_10pp=x, near_duplicate=x + rng.normal(0, 1e-10, n), y=rng.normal(size=n)))
    with pytest.raises(ValueError, match="unstable"):
        within_fit(df, "y", ["nova_per_10pp", "near_duplicate"])


def test_output_directory_with_spaces_and_uri_reserved_characters(tmp_path, config):
    c, cfg = make_data(tmp_path / "data", config, people=1, days=2)
    c["plots"]["example_meals"] = 0
    cfg.write_text(json.dumps(c))
    out = tmp_path / "输出 directory #one?"
    assert main(["--data-root", str(tmp_path / "data"), "--config", str(cfg), "--output-root", str(out)]) == 0
    assert next(out.iterdir()).joinpath("private/02_meals.csv").is_file()


def test_lowercase_timezone_as_in_actual_export(config):
    from test_core import food_data
    from hppmeal.meals import build_meals
    food = food_data([0, 5, 200])
    food["tz"] = "asia/jerusalem"
    meals = build_meals(food, config)
    assert not meals.timezone_conflict.any()
    assert meals.local_hour.notna().all()
