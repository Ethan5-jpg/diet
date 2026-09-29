"""Prevent cross-visit joins and incomplete runs being presented as successful."""
import json

import pandas as pd
import pytest

from fixtures import make_data
from test_readiness import built_run
from run_pipeline import main
from hppmeal.common import Run, acquire_run_lock, sha256
from hppmeal.models import covariates, load_microbiome, audit_microbiome_dates


@pytest.mark.parametrize("source,filename,visit_column", [
    ("covariates", "cov.csv", "cgm_research_stage"),
    ("microbiome", "micro.csv", "research_stage"),
])
def test_participant_tables_filter_before_joining(tmp_path, config, source, filename, visit_column):
    c, _ = make_data(tmp_path / "data", config, people=3, days=1)
    p = tmp_path / "data" / filename
    d = pd.read_csv(p, dtype=str)
    # The same person in a different visit/cohort must not create a duplicate
    # or provide an eligible covariate/microbiome measurement.
    extra = d.iloc[[0]].copy()
    extra[visit_column] = "01_00_visit"
    d.loc[1, visit_column] = "01_00_visit"
    d.loc[2, "cohort"] = "another_cohort"
    pd.concat([d, extra], ignore_index=True).to_csv(p, index=False)
    run = Run(tmp_path / "data", tmp_path / "out", c)
    selected = covariates(run) if source == "covariates" else load_microbiome(run)[0]
    assert selected.participant_id.tolist() == ["00000"]
    scope = pd.read_csv(run.out / "tables" / ("03_{}_scope.csv".format(source)))
    assert scope.selected_rows.iloc[0] == 1
    assert scope.other_cohort_or_visit.iloc[0] == 3


@pytest.mark.parametrize("filename,loader,visit_column", [
    ("cov.csv", covariates, "cgm_research_stage"),
    ("micro.csv", load_microbiome, "research_stage"),
])
def test_visit_metadata_is_required_not_guessed(tmp_path, config, filename, loader, visit_column):
    c, _ = make_data(tmp_path / "data", config, people=1, days=1)
    p = tmp_path / "data" / filename
    pd.read_csv(p, dtype=str).drop(columns=[visit_column]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="metadata required"):
        loader(Run(tmp_path / "data", tmp_path / "out", c))


def test_duplicate_microbiome_in_selected_visit_is_rejected(tmp_path, config):
    c, _ = make_data(tmp_path / "data", config, people=1, days=1)
    p = tmp_path / "data/micro.csv"
    d = pd.read_csv(p, dtype=str)
    pd.concat([d, d], ignore_index=True).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate identity"):
        load_microbiome(Run(tmp_path / "data", tmp_path / "out", c))


def test_covariate_visit_alias_conflict_is_rejected(tmp_path, config):
    c, _ = make_data(tmp_path / "data", config, people=1, days=1)
    p = tmp_path / "data/cov.csv"
    d = pd.read_csv(p, dtype=str)
    d["research_stage"] = "01_00_visit"
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="conflicting visit"):
        covariates(Run(tmp_path / "data", tmp_path / "out", c))


def test_preflight_reports_missing_scope_without_blocking_build(tmp_path, config):
    c, cfg = make_data(tmp_path / "data", config, people=1, days=1)
    p = tmp_path / "data/micro.csv"
    pd.read_csv(p).drop(columns=["cohort"]).to_csv(p, index=False)
    assert main(["--data-root", str(tmp_path / "data"), "--config", str(cfg),
                 "--output-root", str(tmp_path / "out"), "--stage", "check"]) == 0
    run = next((tmp_path / "out").iterdir())
    state = json.loads((run / "manifest.json").read_text())["status"]["check"]
    assert state["build_ready"] and not state["model_input_headers_ready"]


def test_configured_sample_date_file_must_exist(tmp_path, config):
    c, _ = make_data(tmp_path / "data", config, people=1, days=1)
    c["inputs"]["microbiome_dates"] = "missing_dates.csv"
    with pytest.raises(ValueError, match="Configured microbiome_dates file missing"):
        audit_microbiome_dates(Run(tmp_path / "data", tmp_path / "out", c), pd.DataFrame())


def test_run_directory_lock_prevents_second_process_writing(tmp_path, config):
    _, _, args, run = built_run(tmp_path, config)
    before = {p: sha256(run / p) for p in ["manifest.json", "effective_config.json", "SUMMARY.md"]}
    with acquire_run_lock(run):
        assert main(args + ["--stage", "analyze", "--run-dir", str(run)]) == 2
    assert all(sha256(run / p) == value for p, value in before.items())
    # OS lock is released without deleting the file, including after failures.
    with acquire_run_lock(run):
        pass


def test_failure_after_analysis_returns_never_leaves_pass(tmp_path, config, monkeypatch):
    _, _, args, run = built_run(tmp_path, config)

    def mutate_source_then_return_success(current, meals):
        with current.path("diet").open("a") as f:
            f.write("\n")
        return dict(status="PASS")

    monkeypatch.setattr("hppmeal.models.analyze", mutate_source_then_return_success)
    assert main(args + ["--stage", "analyze", "--run-dir", str(run)]) == 2
    status = json.loads((run / "manifest.json").read_text())["status"]
    assert status["analyze"]["status"] == "FAILED"
    assert "Input changed during run" in status["analyze"]["reason"]
    assert "FAILED" in (run / "SUMMARY.md").read_text()


def test_interrupt_persists_incomplete_stage_and_releases_lock(tmp_path, config, monkeypatch):
    _, _, args, run = built_run(tmp_path, config)

    def interrupt(current, meals):
        assert "RUNNING" in (current.out / "SUMMARY.md").read_text()
        raise KeyboardInterrupt()

    monkeypatch.setattr("hppmeal.models.analyze", interrupt)
    assert main(args + ["--stage", "analyze", "--run-dir", str(run)]) == 130
    status = json.loads((run / "manifest.json").read_text())["status"]
    assert status["build"]["status"] == "PASS"
    assert status["analyze"]["status"] == "INTERRUPTED"
    assert "INTERRUPTED" in (run / "SUMMARY.md").read_text()
    with acquire_run_lock(run):
        pass


def test_show_results_does_not_display_failed_stage_tables(tmp_path, config, monkeypatch, capsys):
    import show_results
    _, _, _, run = built_run(tmp_path, config)
    capsys.readouterr()
    p = run / "manifest.json"
    state = json.loads(p.read_text())
    state["status"]["analyze"] = dict(status="FAILED")
    p.write_text(json.dumps(state))
    (run / "tables/03_within_person_models.csv").write_text("model,status\nDO_NOT_DISPLAY,ESTIMATED\n")
    monkeypatch.setattr("sys.argv", ["show_results.py", "--run-dir", str(run)])
    show_results.main()
    output = capsys.readouterr().out
    assert "DO_NOT_DISPLAY" not in output
    assert "不展示可能残留的结果表" in output


def test_changed_meal_table_rejected_before_time_confirmation_snapshot(tmp_path, config):
    c, cfg = make_data(tmp_path / "data", config, people=1, days=2)
    c["time"]["consumption_time_verified"] = False
    c["plots"]["example_meals"] = 0
    cfg.write_text(json.dumps(c))
    args = ["--data-root", str(tmp_path / "data"), "--config", str(cfg)]
    assert main(args + ["--output-root", str(tmp_path / "out"), "--stage", "build"]) == 0
    run = next((tmp_path / "out").iterdir())
    expected = {p: sha256(run / p) for p in ["manifest.json", "effective_config.json", "SUMMARY.md"]}
    c["time"]["consumption_time_verified"] = True
    cfg.write_text(json.dumps(c))
    with (run / "private/02_meals.csv").open("a") as f:
        f.write("\n")
    assert main(args + ["--stage", "analyze", "--run-dir", str(run)]) == 2
    assert all(sha256(run / p) == value for p, value in expected.items())
