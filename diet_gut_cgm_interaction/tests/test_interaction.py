import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from interaction.common import InputError, norm_ids
from interaction.inputs import load_config, load_inputs, resolve_paths, resolve_taxa
from interaction.models import (ModelUnavailable, apply_fdr, bh_adjust, build_reference, contrast,
                                fit_interaction, make_design, ols_hc3)
from interaction.pipeline import run_models
from interaction.synthetic import create_fixture

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    config = load_config(PACKAGE / "config/analysis.json")
    config["minimum_model_n"] = 100
    root = tmp_path / "different root" / "renamed_data"
    create_fixture(root, config, n=360)
    paths = resolve_paths(config, root)
    return config, root, paths


def test_eligibility_and_corrected_scores(prepared):
    config, _, paths = prepared
    master, meta = load_inputs(config, paths)
    assert not master.iloc[:3].primary_eligible.any()
    assert master.iloc[3:30].primary_eligible.all()
    assert not master.iloc[3:30].strict_eligible.any()
    assert not master.iloc[30:45].primary_eligible.any()
    old = pd.read_csv(paths["current4"]).set_index("participant_id")
    np.testing.assert_allclose(master.hPDI, -old.hPDI_z)
    assert meta["summary"]["missing_covariate_record_n"] == 3
    assert meta["summary"]["primary_eligible_n"] == 342
    assert norm_ids(pd.Series(["001.0", "002"])).tolist() == ["001", "002"]


@pytest.mark.parametrize("issue", ["duplicate", "wrong_visit", "bad_nova", "bad_z", "device", "bad_flag", "bad_numeric"])
def test_invalid_input_fails(prepared, issue):
    config, _, paths = prepared
    label = {"duplicate": "microbiome", "wrong_visit": "cgm", "bad_nova": "candidate_master", "bad_z": "cgm", "device": "cgm", "bad_flag": "covariates", "bad_numeric": "new3"}[issue]
    frame = pd.read_csv(paths[label])
    if issue == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif issue == "wrong_visit":
        frame["research_stage"] = "wrong_visit"
    elif issue == "bad_nova":
        frame.loc[0, "nova4_total_energy_pct_cov80_z"] += 1
    elif issue == "bad_z":
        frame.loc[0, "cgm_cv_z"] += 1
    elif issue == "device":
        frame.loc[10, "cgm_device_type"] = "DifferentDevice"
    elif issue == "bad_flag":
        frame.loc[0, "exclude_diabetes_or_a10"] = 8
    else:
        frame["NOVA4_z"] = frame.NOVA4_z.astype(str)
        frame.loc[0, "NOVA4_z"] = "invalid"
    frame.to_csv(paths[label], index=False)
    with pytest.raises(InputError):
        load_inputs(config, paths)


def test_exact_taxon_matching():
    taxa = [{"id": "T", "species": "Test_species"}]
    with pytest.raises(InputError):
        resolve_taxa(["s__Test_species_subspecies"], taxa)
    with pytest.raises(InputError):
        resolve_taxa(["k__A|s__Test_species", "k__B|s__Test_species"], taxa)
    assert resolve_taxa(["k__A|s__Test_species"], taxa)[0]["column"] == "k__A|s__Test_species"


def test_hc3_and_contrast_against_independent_statsmodels():
    sm = pytest.importorskip("statsmodels.api", reason="Local reference dependency; not required on server")
    rng = np.random.RandomState(417)
    x = np.column_stack([np.ones(700), rng.normal(size=(700, 4))])
    x = np.column_stack([x, x[:, 1] * x[:, 2]])
    y = x @ np.array([.8, .5, -.2, .1, .3, .6]) + rng.normal(size=700) * (.5 + abs(x[:, 1]))
    actual, _ = ols_hc3(y, x)
    reference = sm.OLS(y, x).fit(cov_type="HC3", use_t=True)
    np.testing.assert_allclose(actual.params, reference.params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.cov_params(), reference.cov_params(), rtol=1e-9, atol=1e-12)
    weights = np.array([0, 1, 0, 0, 0, .7])
    estimate = contrast(actual, list("abcdef"), {"b": 1, "f": .7})
    expected = reference.t_test(weights)
    np.testing.assert_allclose([estimate["beta"], estimate["se"], estimate["p"]], [float(expected.effect[0]), float(expected.sd[0, 0]), float(expected.pvalue)], rtol=1e-9)
    np.testing.assert_allclose([estimate["ci_low"], estimate["ci_high"]], expected.conf_int()[0], rtol=1e-9)


def test_hc3_matches_direct_sandwich_without_reference_dependency():
    rng = np.random.RandomState(18)
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    y = rng.normal(size=200)
    fit, hat = ols_hc3(y, x)
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    adjusted = (y - x @ beta) ** 2 / (1-hat) ** 2
    covariance = bread @ (x.T @ (x * adjusted[:, None])) @ bread
    np.testing.assert_allclose(fit.cov_params(), covariance, rtol=1e-10, atol=1e-12)


def test_fdr_keeps_failed_tests_in_denominator():
    frame = pd.DataFrame(dict(model=["primary"]*4, family=["focus"]*4,
                              status=["OK", "OK", "NOT_ESTIMABLE", "SKIPPED"], interaction_p=[.01, .02, np.nan, np.nan]))
    result, audit = apply_fdr(frame)
    np.testing.assert_allclose(result.q_focus.iloc[:2], [.04, .04])
    assert result.q_focus.iloc[2:].isna().all()
    assert (audit.planned_tests == 4).all()
    assert (audit.nonestimable_in_denominator == 2).all()
    np.testing.assert_allclose(bh_adjust([.03, .001, .1]), [.045, .003, .1])


def test_nonlinear_main_effect_can_mimic_interaction():
    rng = np.random.RandomState(71)
    n = 1800
    d = rng.normal(size=n)
    m = .8*d + rng.normal(size=n)*.6
    frame = pd.DataFrame({"Ddiet": d, "Mtaxon": m, "Y": 3*d*d + m*m + rng.normal(size=n)})
    config = dict(minimum_model_n=100, minimum_residual_df=30, minimum_tail_cell_n=10)
    ref = build_reference(frame, "Ddiet", "Mtaxon")
    linear, _ = fit_interaction(frame, "Ddiet", "Mtaxon", "Y", [], ref, "linear", config, 1)
    quadratic, _ = fit_interaction(frame, "Ddiet", "Mtaxon", "Y", [], ref, "quadratic", config, 1)
    assert linear["interaction_p"] < 1e-8
    assert quadratic["interaction_p"] > .05
    assert abs(quadratic["interaction_beta"]) < .15


def test_rank_deficiency_not_silently_accepted():
    x = np.column_stack([np.ones(100), np.arange(100), np.arange(100)])
    with pytest.raises(ModelUnavailable):
        ols_hc3(np.arange(100), x)


@pytest.mark.parametrize("tied", ["diet", "microbe"])
def test_zero_iqr_keeps_estimable_interaction(tied):
    rng = np.random.RandomState(713)
    n = 500
    frame = pd.DataFrame({"diet": rng.normal(size=n), "microbe": rng.normal(size=n)})
    frame[tied] = 0.0
    frame.loc[:79, tied] = rng.uniform(.5, 2, 80)
    frame["outcome"] = frame.diet + frame.microbe + 2*frame.diet*frame.microbe + rng.normal(size=n)
    config = dict(minimum_model_n=100, minimum_residual_df=30, minimum_tail_cell_n=20)
    ref = build_reference(frame, "diet", "microbe")
    result, slopes = fit_interaction(frame, "diet", "microbe", "outcome", [], ref, "linear", config, 1)
    assert result["interaction_p"] < 1e-8
    assert result["limited_joint_support"]
    assert result["double_iqr_status"] == "NOT_INFORMATIVE_ZERO_IQR"
    assert np.isnan(result["double_iqr_beta_native"])
    assert np.isnan(result["support_min_tail_n"])
    assert len(slopes) == 3


@pytest.mark.parametrize("known,a10,combined", [(0, np.nan, 0), (0, 0, 1), (0, 0, np.nan)])
def test_exclusion_union_consistency(prepared, known, a10, combined):
    config, _, paths = prepared
    cov = pd.read_csv(paths["covariates"])
    cov["known_diabetes"] = cov.exclude_diabetes_or_a10
    cov["a10_medication_use"] = cov.exclude_diabetes_or_a10
    cov.loc[100, ["known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]] = [known, a10, combined]
    cov.to_csv(paths["covariates"], index=False)
    with pytest.raises(InputError, match="Exclusion flag"):
        load_inputs(config, paths)


def test_valid_three_state_exclusion_union(prepared):
    config, _, paths = prepared
    cov = pd.read_csv(paths["covariates"])
    cov["known_diabetes"] = cov.exclude_diabetes_or_a10
    cov["a10_medication_use"] = cov.exclude_diabetes_or_a10
    # A single known negative is still unknown; one positive is sufficient.
    cov.loc[100, ["known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]] = [0, np.nan, np.nan]
    cov.loc[101, ["known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]] = [1, np.nan, 1]
    cov.to_csv(paths["covariates"], index=False)
    master, _ = load_inputs(config, paths)
    first, second = cov.loc[[100, 101], "participant_id"]
    assert master.loc[first, "primary_eligible"]
    assert not master.loc[first, "strict_eligible"]
    assert not master.loc[second, "primary_eligible"]


def test_all_unavailable_writes_readable_diagnostics(prepared, tmp_path):
    config, root, _ = prepared
    config["minimum_model_n"] = 10000
    config_file = tmp_path / "unavailable_config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "all_unavailable"
    process = subprocess.run([sys.executable, str(PACKAGE / "run_analysis.py"), "--data-root", str(root),
                              "--config", str(config_file), "--output-dir", str(output)],
                             capture_output=True, text=True)
    assert process.returncode == 2, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "NO_ESTIMABLE_FOCUS_MODELS"
    results = pd.read_csv(output / "tables/all_interactions.csv")
    assert len(results) == 228
    assert results.status.eq("NOT_ESTIMABLE").all()
    assert results.interaction_beta_native.isna().all()
    slopes = pd.read_csv(output / "tables/conditional_diet_slopes.csv")
    assert slopes.empty and "beta_native" in slopes
    assert len(list((output / "figures").glob("*.png"))) == 4
    assert (output / "aggregate_results.zip").is_file()


def test_frozen_grid_and_missing_protocol(prepared):
    config, _, paths = prepared
    paths["protocol"].unlink()
    master, metadata = load_inputs(config, paths)
    results, slopes, flows, families = run_models(master, metadata, config)
    assert len(results) == 228
    assert results.status.eq("OK").sum() == 180
    assert results.status.eq("SKIPPED").sum() == 48
    assert len(results.loc[results.model.eq("primary")]) == 84
    assert len(slopes) == 180 * 3
    assert families.loc[families.model.eq("primary") & families.family.eq("focus"), "planned_tests"].iloc[0] == 24
    common = flows.loc[flows.model.eq("common_focus_sample")]
    assert common.sample_sha256.nunique() == 1
    assert flows.loc[flows.model.eq("strict"), "unknown_exclusion_in_model_n"].eq(0).all()
    config["protocol_mode"] = "required"
    with pytest.raises(InputError):
        load_inputs(config, paths)


def test_complete_package_relocates_without_data_root_flag(prepared, tmp_path):
    config, root, _ = prepared
    target = root / "diet_gut_cgm_interaction"
    shutil.copytree(PACKAGE, target, ignore=shutil.ignore_patterns("outputs", "__pycache__", ".pytest_cache"))
    (target / "config/analysis.json").write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "relocated_check"
    result = subprocess.run([sys.executable, str(target / "run_analysis.py"), "--check-only", "--output-dir", str(output)],
                            cwd=str(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "INPUT_CHECK_PASSED"
    assert Path(manifest["data_root"]) == root
    for item in manifest["inputs"]:
        assert not Path(item["path"]).relative_to(root).is_absolute()


def test_python38_syntax():
    for file in list((PACKAGE / "interaction").glob("*.py")) + list(PACKAGE.glob("*.py")):
        ast.parse(file.read_text(), filename=str(file), feature_version=(3, 8))
