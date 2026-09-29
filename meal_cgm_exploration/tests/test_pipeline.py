import json
from pathlib import Path

import pandas as pd
import pytest

from fixtures import make_data
from run_pipeline import main
from hppmeal.common import sha256


def only_run(root):
    return next(Path(root).iterdir())


def test_complete_synthetic_pipeline(tmp_path, config):
    data = tmp_path / "data"
    c, cfg = make_data(data, config)
    out = tmp_path / "out"
    code = main(["--data-root", str(data), "--config", str(cfg), "--output-root", str(out), "--stage", "all"])
    run = only_run(out)
    assert code == 0, (run / "SUMMARY.md").read_text()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["data_kind"] == "SYNTHETIC"
    assert all(x["status"] == "PASS" for x in manifest["status"].values())
    m = pd.read_csv(run / "private" / "02_meals.csv", dtype={"participant_id": str})
    assert len(m) == 12 * 6 * 3
    assert m.participant_id.str.len().eq(5).all()
    assert m.analysis_ready.sum() > 100
    results = pd.read_csv(run / "tables" / "03_within_person_models.csv")
    assert results.status.eq("ESTIMATED").all()
    assert (run / "figures" / "02_meal_examples.png").stat().st_size > 1000


def test_resume_after_time_confirmation_and_input_change_rejection(tmp_path, config):
    data = tmp_path / "data"
    c, cfg = make_data(data, config)
    c["time"]["consumption_time_verified"] = False
    cfg.write_text(json.dumps(c))
    out = tmp_path / "out"
    assert main(["--data-root", str(data), "--config", str(cfg), "--output-root", str(out), "--stage", "all"]) == 2
    run = only_run(out)
    assert not (run / "tables" / "03_within_person_models.csv").exists()
    c["time"]["consumption_time_verified"] = True
    cfg.write_text(json.dumps(c))
    assert main(["--data-root", str(data), "--config", str(cfg), "--run-dir", str(run), "--stage", "analyze"]) == 0
    original = sha256(run / "manifest.json")
    with (data / "diet.csv").open("a") as f:
        f.write("\n")
    assert main(["--data-root", str(data), "--config", str(cfg), "--run-dir", str(run), "--stage", "audit"]) == 2
    assert sha256(run / "manifest.json") == original


def test_cgm_without_timestamp_fails_no_fake_success(tmp_path, config):
    data = tmp_path / "data"
    c, cfg = make_data(data, config, people=1, days=2)
    pd.DataFrame({"glucose": [90, 100]}).to_csv(data / "cgm" / "00000.csv", index=False)
    out = tmp_path / "out"
    assert main(["--data-root", str(data), "--config", str(cfg), "--output-root", str(out)]) == 2
    run = only_run(out)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"]["check"]["status"] == "FAILED"
    assert "audit" not in manifest["status"]
    assert not (run / "private" / "02_meals.csv").exists()


def test_synthetic_not_allowed_as_real(tmp_path, config):
    data = tmp_path / "data"
    c, cfg = make_data(data, config, people=1, days=2)
    c["data_kind"] = "REAL"
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="Synthetic"):
        main(["--data-root", str(data), "--config", str(cfg), "--output-root", str(tmp_path / "out")])


def test_bad_diet_time_and_duplicate_food_not_silently_ignored(tmp_path, config):
    data = tmp_path / "data"
    c, cfg = make_data(data, config, people=2, days=2)
    diet = pd.read_csv(data / "diet.csv", dtype={"participant_id": str})
    i = diet.index[diet.participant_id.eq("00000")][0]
    diet.loc[i, "collection_timestamp"] = "cannot_parse"
    j = diet.index[diet.participant_id.eq("00001")][0]
    diet = pd.concat([diet, diet.loc[[j]]], ignore_index=True)
    diet.to_csv(data / "diet.csv", index=False)
    out = tmp_path / "out"
    assert main(["--data-root", str(data), "--config", str(cfg), "--output-root", str(out)]) == 0
    run = only_run(out)
    m = pd.read_csv(run / "private" / "02_meals.csv", dtype={"participant_id": str})
    assert m.loc[m.participant_id.eq("00000"), "diet_time_issue"].all()
    assert not m.loc[m.participant_id.eq("00000"), "analysis_ready"].any()
    duplicates = m.loc[m.duplicate_item]
    assert len(duplicates) == 1 and not duplicates.meal_eligible.any()


def test_missing_nova_mapping_builds_but_never_fits_empty_exposure(tmp_path, config):
    data = tmp_path / "data"
    c, cfg = make_data(data, config)
    (data / "nova.csv").unlink()
    c["plots"]["example_meals"] = 0
    cfg.write_text(json.dumps(c))
    out = tmp_path / "out"
    assert main(["--data-root", str(data), "--config", str(cfg), "--output-root", str(out), "--stage", "all"]) == 2
    run = only_run(out)
    m = pd.read_csv(run / "private" / "02_meals.csv")
    assert not m.nova_eligible.any()
    fit = pd.read_csv(run / "tables" / "03_within_person_models.csv")
    assert fit.status.eq("NOT_ESTIMATED").all()
