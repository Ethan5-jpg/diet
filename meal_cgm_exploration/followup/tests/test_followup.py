import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE))

from hppmeal.common import Run, sha256, write_json
from hppmeal.report import repeat_pairs
from followup.io_utils import core_fingerprint, inspect_source, cohort_meals
from followup.repeats import (checked_pairs, difference_summary, integer_weighted_quantile,
                             template_structure, structure_summary, OUTCOMES)
from followup.prediction import (features, temporal_split, prepare_prediction, fit_predict,
                                 predict_cohort, FEATURES)
from followup.run_repeated import main


def make_meals(people=60, days=9):
    rng = np.random.RandomState(47)
    rows = []
    for p in range(people):
        for day in range(days):
            for template in range(2):
                base = 95 + 4 * p / people + rng.normal(0, 10)
                slope = rng.normal(0, 0.15)
                hour = 8 + 5 * template + rng.uniform(-0.2, 0.2)
                t = 1704067200 + day * 86400 + hour * 3600
                mean = 35 + 0.78 * base + template * 6 + 9 * slope + rng.normal(0, 0.7)
                rows.append(dict(participant_id="{:05d}".format(p), cohort="10k", research_stage="00_00_visit",
                    meal_id="m{:05d}_{}_{}".format(p, day, template), signature="food_{}".format(template),
                    start=t, end=t + 600, local_date="2024-01-{:02d}".format(day + 1), local_hour=hour,
                    analysis_ready=True, primary_eligible=True, strict_eligible=True,
                    connection_id="c{:05d}".format(p), duration_minutes=10,
                    calories_kcal=200 + template * 150, carbohydrate_g=20 + template * 20,
                    previous_gap_minutes=180 + 60 * template, baseline_glucose=base,
                    pre_slope_per_min=slope, iauc_net_0_120=(mean - base) * 120,
                    mean_glucose=mean, mean_increment=mean - base,
                    peak_rise_0_120=mean - base + 12))
    return pd.DataFrame(rows)


def make_source(tmp_path, meals=None):
    root, source = tmp_path / "data", tmp_path / "original run"
    root.mkdir(parents=True)
    (root / "SYNTHETIC_TEST_ONLY.json").write_text('{"synthetic": true}')
    c = json.loads((PACKAGE / "config" / "default.json").read_text())
    c["data_kind"] = "SYNTHETIC"
    c["time"]["consumption_time_verified"] = True
    c["inputs"]["covariates"] = "covariates.csv"
    d = make_meals() if meals is None else meals.copy()
    participants = d.participant_id.unique()
    cov = pd.DataFrame({"participant_id": participants, "cohort": "10k", "cgm_research_stage": "00_00_visit",
                        "exclude_diabetes_or_a10": 0})
    for col in c["analysis"]["require_covariate_columns"]:
        cov[col] = "yes" if col in ["sex", "education_level", "smoking_status", "cgm_device_type"] else 1
    cov.loc[cov.index % 20 == 0, "exclude_diabetes_or_a10"] = 1
    cov.loc[cov.index % 20 == 1, "exclude_diabetes_or_a10"] = np.nan
    cov.to_csv(root / "covariates.csv", index=False)
    run = Run(root, source, c)
    run.table(d.drop(columns=["primary_eligible", "strict_eligible", "mean_glucose", "mean_increment"]), "02_meals.csv", private=True)
    run.table(repeat_pairs(d), "02_repeated_meal_pairs.csv", private=True)
    run.record(root / "covariates.csv", "covariates")
    run.status = {"audit": {"status": "PASS"}, "build": {"status": "PASS",
                  "meals_sha256": sha256(source / "private" / "02_meals.csv")}, "analyze": {"status": "PASS"}}
    run.save()
    write_json(source / "code_fingerprint.json", core_fingerprint())
    return root, source


def snapshot(path):
    return {str(p.relative_to(path)): sha256(p) for p in path.rglob("*") if p.is_file()}


def test_integer_weighted_quantile_matches_expanded_numpy():
    rng = np.random.RandomState(31)
    for _ in range(100):
        x = np.sort(rng.normal(size=30))
        w = rng.randint(0, 5, size=30)
        for q in [0.1, 0.5, 0.9]:
            assert integer_weighted_quantile(x, w, q) == pytest.approx(np.quantile(np.repeat(x, w), q))


def test_pair_regeneration_rejects_tampering_and_cross_identity():
    d = make_meals(4, 4)
    p = repeat_pairs(d)
    checked = checked_pairs(d, p)
    assert len(checked) == 16 and checked.same_connection.all()
    aligned = checked.merge(p[["meal_a", "meal_b", "delta"]], on=["meal_a", "meal_b"], validate="one_to_one")
    assert np.allclose(aligned.mean_increment, aligned.delta / 120)
    pd.testing.assert_frame_equal(checked, checked_pairs(d, p.sample(frac=1, random_state=3)))
    p.loc[0, "delta"] += 10
    with pytest.raises(ValueError, match="values differ"):
        checked_pairs(d, p)


def test_cluster_bootstrap_person_equal_is_not_meal_weighted():
    p = pd.DataFrame({"participant_id": ["a", "b", "c"], "complete": True,
                      "primary_eligible": True, "strict_eligible": True})
    for col in OUTCOMES:
        p[col] = [0.0, 2.0, 10.0]
    duplicated = pd.concat([p, pd.concat([p.iloc[[2]]] * 20)], ignore_index=True)
    first = difference_summary(p, "primary", 250, 9, minimum_people=3)
    second = difference_summary(duplicated, "primary", 250, 9, minimum_people=3)
    key = "person_mean_of_median_absolute_difference"
    a = first.loc[first.metric.eq(key), ["estimate", "ci_low", "ci_high"]]
    b = second.loc[second.metric.eq(key), ["estimate", "ci_low", "ci_high"]]
    assert np.allclose(a, b)
    assert first.loc[first.metric.eq("pair_median_absolute_difference"), "estimate"].iloc[0] != second.loc[second.metric.eq("pair_median_absolute_difference"), "estimate"].iloc[0]


def test_zero_differences_and_small_samples():
    d = make_meals(3, 4)
    p = checked_pairs(d, repeat_pairs(d))
    p[OUTCOMES] = 0.0
    estimated = difference_summary(p, "primary", 200, minimum_people=3)
    assert estimated[["estimate", "ci_low", "ci_high"]].eq(0).all().all()
    limited = difference_summary(p, "primary", 200)
    assert limited.status.eq("DESCRIPTIVE_ONLY").all()
    assert limited.ci_low.isna().all()
    empty = difference_summary(p.iloc[0:0], "primary", 200)
    assert empty.status.eq("NO_PAIRS").all()


def test_templates_keep_all_repeats_and_circular_clock():
    d = make_meals(1, 4)
    d.loc[d.signature.eq("food_0"), "local_hour"] = [23.8, 0.1, 23.9, 0.2]
    t = template_structure(d)
    row = t.loc[t.signature.eq("food_0")].iloc[0]
    assert row.meals == 4 and row.dates == 4
    assert row.clock_span_hours == pytest.approx(0.4)
    assert structure_summary(t).templates.max() == 2


def test_date_split_is_disjoint_and_templates_need_two_train_dates():
    d = make_meals(3, 9)
    train, test, flow, assignment = prepare_prediction(d)
    assert set(train.local_date).isdisjoint(set(test.local_date))
    assert train.local_date.max() < test.local_date.min()
    assert train.training_dates.min() >= 2
    late = temporal_split(d)
    unseen = d.copy()
    unseen.loc[late.split.eq("test"), "signature"] = "future_unseen"
    assert prepare_prediction(unseen)[1].empty


def test_future_outcomes_do_not_affect_fit_or_features():
    d = make_meals(4, 9)
    train, test, _, _ = prepare_prediction(d)
    before, coef_before = fit_predict(train, test, FEATURES)
    changed = d.copy()
    changed.loc[changed.meal_id.isin(test.meal_id), "mean_glucose"] += 10000
    train2, test2, _, _ = prepare_prediction(changed)
    after, coef_after = fit_predict(train2, test2, FEATURES)
    pd.testing.assert_frame_equal(train, train2)
    pd.testing.assert_frame_equal(coef_before, coef_after)
    assert np.allclose(before, after)
    # Realized meal duration is deliberately excluded from meal-start information.
    changed["duration_minutes"] += 500
    assert features(changed)[FEATURES].equals(features(d)[FEATURES])


def test_known_synthetic_premeal_signal_improves_holdout():
    info, flow, assignment, pred, metrics, gains, coef = predict_cohort(make_meals(), "primary", 200, 31)
    assert info["status"] == "ESTIMATED"
    metrics = metrics.set_index("model")
    assert metrics.loc["plus_premeal", "person_equal_mae"] < metrics.loc["template_history", "person_equal_mae"] / 2
    assert set(pred.participant_id).issubset(set(assignment.loc[assignment.split.eq("train"), "participant_id"]))


def test_not_enough_prediction_support_is_explicit():
    info = predict_cohort(make_meals(4, 2), "primary", 200, 31)[0]
    assert info["status"] == "NOT_ESTIMATED" and info["test_meals"] == 0


def test_end_to_end_preserves_original_run_and_separates_cohorts(tmp_path):
    root, source = make_source(tmp_path)
    before = snapshot(source)
    out = tmp_path / "followup outputs"
    assert main(["--run-dir", str(source), "--data-root", str(root), "--output-root", str(out), "--bootstrap", "200"]) == 0
    assert snapshot(source) == before
    result = next(out.iterdir())
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["data_kind"] == "SYNTHETIC"
    assert manifest["status"]["followup"]["status"] == "COMPLETE"
    flow = pd.read_csv(result / "tables" / "01_cohort_flow.csv").set_index("cohort")
    assert flow.loc["all_descriptive", "people"] == 60
    assert flow.loc["primary", "people"] == 57
    assert flow.loc["strict", "people"] == 54
    assert (result / "figures" / "03_repeated_differences.png").stat().st_size > 1000
    assert (result / "figures" / "04_prediction_gains.png").stat().st_size > 1000
    assert "SYNTHETIC" in (result / "SUMMARY.md").read_text()
    for p in (result / "tables").glob("*.csv"):
        assert "participant_id" not in pd.read_csv(p).columns


def test_no_repeats_is_valid_complete_with_limitations(tmp_path):
    d = make_meals(4, 4)
    d["signature"] = d.meal_id
    root, source = make_source(tmp_path, d)
    out = tmp_path / "out"
    assert main(["--run-dir", str(source), "--output-root", str(out), "--bootstrap", "200"]) == 0
    result = next(out.iterdir())
    m = json.loads((result / "manifest.json").read_text())
    assert m["status"]["followup"]["status"] == "COMPLETE_WITH_LIMITATIONS"
    assert pd.read_csv(result / "tables" / "03_pair_differences.csv").status.eq("NO_PAIRS").all()
    assert pd.read_csv(result / "tables" / "04_prediction_metrics.csv").empty


def test_changed_meal_rejected_before_output(tmp_path):
    root, source = make_source(tmp_path, make_meals(3, 3))
    with (source / "private" / "02_meals.csv").open("a") as f:
        f.write("\n")
    out = tmp_path / "out"
    assert main(["--run-dir", str(source), "--output-root", str(out)]) == 2
    assert not out.exists()


def test_changed_covariates_fail_without_touching_original(tmp_path):
    root, source = make_source(tmp_path, make_meals(3, 3))
    before = snapshot(source)
    with (root / "covariates.csv").open("a") as f:
        f.write("\n")
    out = tmp_path / "out"
    assert main(["--run-dir", str(source), "--output-root", str(out)]) == 2
    assert snapshot(source) == before
    result = next(out.iterdir())
    assert json.loads((result / "manifest.json").read_text())["status"]["followup"]["status"] == "FAILED"


def test_time_review_required_and_output_must_be_separate(tmp_path):
    root, source = make_source(tmp_path, make_meals(3, 3))
    before = snapshot(source)
    assert main(["--run-dir", str(source), "--output-root", str(source / "followup")]) == 2
    assert snapshot(source) == before
    m = json.loads((source / "manifest.json").read_text())
    m["config"]["time"]["consumption_time_verified"] = False
    write_json(source / "manifest.json", m)
    with pytest.raises(ValueError, match="time interpretation"):
        inspect_source(source)


def test_interrupt_is_not_reported_complete(tmp_path, monkeypatch):
    root, source = make_source(tmp_path, make_meals(3, 3))
    before = snapshot(source)
    import followup.reporting

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(followup.reporting, "figures", interrupt)
    out = tmp_path / "out"
    assert main(["--run-dir", str(source), "--output-root", str(out), "--bootstrap", "200"]) == 130
    assert snapshot(source) == before
    result = next(out.iterdir())
    assert json.loads((result / "manifest.json").read_text())["status"]["followup"]["status"] == "INTERRUPTED"


def test_invalid_bootstrap_does_not_create_outputs(tmp_path):
    assert main(["--run-dir", str(tmp_path), "--output-root", str(tmp_path / "out"), "--bootstrap", "1"]) == 2
    assert not (tmp_path / "out").exists()


def test_unsupported_test_people_do_not_acquire_training_information():
    d = make_meals(3, 9)
    train, test, _, _ = prepare_prediction(d)
    test.loc[test.index[0], "participant_id"] = "never_seen"
    with pytest.raises(ValueError, match="Unseen test template"):
        fit_predict(train, test, FEATURES)


def test_constant_training_features_are_dropped_without_using_test_variation():
    d = make_meals(3, 9)
    train, test, _, _ = prepare_prediction(d)
    train[FEATURES] = 1.0
    test[FEATURES] = 1000.0
    history, _ = fit_predict(train, test, [])
    full, coef = fit_predict(train, test, FEATURES)
    assert np.allclose(history, full)
    assert coef.status.eq("CONSTANT_IN_TRAIN").all()


def test_missing_covariate_participant_is_not_unknown_but_eligible(tmp_path):
    from followup.io_utils import read_inputs
    root, source = make_source(tmp_path, make_meals(4, 4))
    p = root / "covariates.csv"
    cov = pd.read_csv(p, dtype={"participant_id": str})
    cov = cov.loc[cov.participant_id.ne("00002")]
    cov.to_csv(p, index=False)
    old = json.loads((source / "manifest.json").read_text())
    old["inputs"][0]["sha256"] = sha256(p)
    write_json(source / "manifest.json", old)
    fresh = Run(root, tmp_path / "new", old["config"])
    meals, _ = read_inputs(fresh, source, inspect_source(source))
    assert not meals.loc[meals.participant_id.eq("00002"), "primary_eligible"].any()
    # The recorded but unknown flag is allowed in primary and excluded in strict.
    assert meals.loc[meals.participant_id.eq("00001"), "primary_eligible"].all()
    assert not meals.loc[meals.participant_id.eq("00001"), "strict_eligible"].any()


def test_core_fingerprint_mismatch_is_rejected(tmp_path):
    root, source = make_source(tmp_path, make_meals(3, 3))
    f = source / "code_fingerprint.json"
    value = json.loads(f.read_text())
    value["hppmeal/cgm.py"] = "different_version"
    write_json(f, value)
    with pytest.raises(ValueError, match="code differs"):
        inspect_source(source)


def test_input_change_during_followup_is_detected(tmp_path):
    from followup.io_utils import verify_unchanged
    root, source = make_source(tmp_path, make_meals(3, 3))
    old = inspect_source(source)
    fresh = Run(root, tmp_path / "new", old["config"])
    path = source / "private" / "02_meals.csv"
    fresh.record(path, "source_meals")
    with path.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="input changed"):
        verify_unchanged(fresh)


@pytest.mark.parametrize("name", ["manifest.json", "private/02_meals.csv", "code_fingerprint.json"])
def test_source_change_between_inspection_and_loading_is_rejected(tmp_path, name):
    from followup.io_utils import read_inputs
    root, source = make_source(tmp_path, make_meals(3, 3))
    old = inspect_source(source)
    fresh = Run(root, tmp_path / "new", old["config"])
    with (source / name).open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="source changed before loading"):
        read_inputs(fresh, source, old)


def test_core_files_are_recorded_for_final_integrity_check(tmp_path):
    from followup.io_utils import read_inputs
    root, source = make_source(tmp_path, make_meals(3, 3))
    old = inspect_source(source)
    fresh = Run(root, tmp_path / "new", old["config"])
    read_inputs(fresh, source, old)
    paths = {x["path"] for x in fresh.inputs}
    assert all(str(PACKAGE / name) in paths for name in core_fingerprint())


def test_empty_or_incomplete_upload_manifest_is_rejected(tmp_path, monkeypatch):
    import followup.io_utils as io
    (tmp_path / "followup").mkdir()
    manifest = tmp_path / "followup" / "MANIFEST.json"
    monkeypatch.setattr(io, "PACKAGE", tmp_path)
    from followup import VERSION
    for files in [{}, {"followup/run_repeated.sh": "0" * 64}]:
        manifest.write_text(json.dumps({"version": VERSION, "files": files}))
        with pytest.raises(ValueError, match="Incomplete followup manifest"):
            io.verify_bundle()


def test_show_missing_directory_returns_actionable_error(tmp_path, capsys):
    assert main(["--show", str(tmp_path / "missing")]) == 2
    assert "exact RUN_DIR" in capsys.readouterr().err


def test_covariate_change_between_hash_and_record_is_rejected(tmp_path, monkeypatch):
    from followup.io_utils import read_inputs
    root, source = make_source(tmp_path, make_meals(3, 3))
    old = inspect_source(source)
    fresh = Run(root, tmp_path / "new", old["config"])
    original = fresh.record

    def change_before_record(path, label):
        if label == "covariates":
            with Path(path).open("a") as f:
                f.write("\n")
        return original(path, label)

    monkeypatch.setattr(fresh, "record", change_before_record)
    with pytest.raises(ValueError, match="Covariate source changed before loading"):
        read_inputs(fresh, source, old)


def test_package_change_after_verification_is_rejected(tmp_path, monkeypatch):
    import followup.io_utils as io
    from hppmeal.common import sha256
    root, source = make_source(tmp_path, make_meals(3, 3))
    original = io.verify_bundle

    def changed_digest():
        # Emulate a verified version becoming stale before input recording.
        result = original()
        result["files"]["followup/repeats.py"] = "0" * 64
        return result

    monkeypatch.setattr(io, "verify_bundle", changed_digest)
    out = tmp_path / "out"
    assert main(["--run-dir", str(source), "--output-root", str(out)]) == 2
    result = next(out.iterdir())
    status = json.loads((result / "manifest.json").read_text())["status"]["followup"]
    assert status["status"] == "FAILED" and "package changed after inspection" in status["reason"]
