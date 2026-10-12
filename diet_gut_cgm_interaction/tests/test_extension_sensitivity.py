import copy
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from extension_sensitivity.interaction.common import InputError, sha256, write_json
from extension_sensitivity.interaction.inputs import load_config, load_inputs, resolve_paths
from extension_sensitivity.interaction.models import bh_adjust
from extension_sensitivity.interaction.pipeline import run_models
from extension_sensitivity.interaction.synthetic import create_fixture
from extension_sensitivity.interaction.extension_sensitivity import (
    CORE_FILES, KEY, MODELS, compare_reference, load_reference, run_extended,
    stability_tables, verify_input_identity,
)

MODULE = Path(__file__).resolve().parents[1]
PACKAGE = MODULE / "extension_sensitivity"


def test_upload_contains_frozen_core_and_python38_syntax():
    for name in CORE_FILES + ["interaction/reporting.py", "interaction/synthetic.py",
                              "config/analysis.json", "requirements.txt", "METHODS.md"]:
        assert (PACKAGE / name).read_bytes() == (MODULE / name).read_bytes()
    for file in PACKAGE.rglob("*.py"):
        ast.parse(file.read_text(), filename=str(file), feature_version=(3, 8))
    result = subprocess.run(["bash", "-n", str(PACKAGE / "run_extension_sensitivity.sh")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("extension_data")
    config = load_config(PACKAGE / "config/analysis.json")
    create_fixture(root, config, n=500)
    paths = resolve_paths(config, root)
    # Make the seven-diet sample strictly smaller than the old focus sample.
    frame = pd.read_csv(paths["current4"])
    frame.loc[100:105, "AHEI_z"] = np.nan
    frame.to_csv(paths["current4"], index=False)
    master, metadata = load_inputs(config, paths)
    original, _, _, _ = run_models(master, metadata, config)
    extended, slopes, flows, families = run_extended(master, metadata, config)
    return config, master, metadata, original, extended, slopes, flows, families


def test_full_grid_and_fixed_fdr(case):
    config, _, _, original, result, slopes, flows, families = case
    assert len(original) == 228 and len(result) == 588
    assert len(result[result.family.eq("extension")]) == 420
    assert set(result.model) == {"primary"} | set(MODELS)
    assert len(slopes) == 3 * result.status.eq("OK").sum()
    assert len(flows) == 588
    assert families.loc[families.family.eq("all84"), "planned_tests"].tolist() == [84] * 7
    for _, group in result.groupby("model"):
        good = group.status.eq("OK")
        q = bh_adjust(group.interaction_p.where(good, 1).to_numpy())
        np.testing.assert_allclose(group.loc[good, "q_all84"], q[good.to_numpy()])
        assert group.loc[~good, "q_all84"].isna().all()
    assert result.loc[result.family.eq("extension"), "q_focus24"].isna().all()


def test_reference_reproduction_and_common_sample(case):
    config, master, _, original, result, _, flows, _ = case
    checks = compare_reference(original, result, config)
    assert len(checks) == 204 and checks.matches.all()
    common = flows[flows.model.eq("common_all7_sample")]
    assert common.sample_sha256.nunique() == 1
    old_n = original[original.model.eq("common_focus_sample")].n.iloc[0]
    assert common.nunique()["complete_case_n"] == 1
    assert common.complete_case_n.iloc[0] == old_n - 6
    assert flows[flows.model.eq("strict")].unknown_exclusion_in_model_n.eq(0).all()


def test_drift_in_coefficient_or_membership_is_detected(case):
    config, _, _, original, result, _, _, _ = case
    for column, value in [("interaction_beta_native", 100), ("sample_sha256", "wrong")]:
        changed = result.copy()
        changed.loc[changed.model.eq("primary").idxmax(), column] = value
        checks = compare_reference(original, changed, config)
        assert not checks.matches.all()


def test_input_identity_requires_exact_labels_and_hashes(case):
    _, _, metadata, _, _, _, _, _ = case
    reference = {"inputs": copy.deepcopy(metadata["files"])}
    verify_input_identity(reference, metadata)
    changed = copy.deepcopy(metadata)
    changed["files"][0]["sha256"] = "bad"
    with pytest.raises(InputError, match="Input identity"):
        verify_input_identity(reference, changed)
    changed = copy.deepcopy(metadata)
    changed["files"].pop()
    with pytest.raises(InputError, match="Input identity"):
        verify_input_identity(reference, changed)


def test_missing_protocol_is_not_counted_as_robust(case):
    config, master, metadata, original, _, _, _, _ = case
    metadata = copy.deepcopy(metadata)
    metadata["summary"]["protocol_available"] = False
    result, _, _, families = run_extended(master, metadata, config)
    assert result.status.eq("SKIPPED").sum() == 168
    protocol = result[result.model.str.startswith("protocol")]
    assert protocol.q_all84.isna().all()
    assert families.loc[families.model.str.startswith("protocol") & families.family.eq("all84"), "nonestimable_in_denominator"].eq(84).all()
    reference = original.copy()
    reference.loc[reference.model.eq("primary") & reference.family.eq("extension"), "q_all"] = .01
    stability, summary = stability_tables(reference, result, config["alpha"])
    assert len(stability) == 60
    assert not stability.all_six_estimable.any()
    assert not stability.retained_all_six.any()
    assert stability.max_sensitivity_q_all84.isna().all()


def test_reference_selection_frozen_even_when_new_results_are_significant(case):
    config, _, _, original, result, _, _, _ = case
    reference = original.copy()
    mask = reference.model.eq("primary") & reference.family.eq("extension")
    reference.loc[mask, "q_all"] = 1.0
    selected = reference.index[mask][0]
    reference.loc[selected, "q_all"] = .01
    result = result.copy()
    result.loc[result.family.eq("extension"), "q_all84"] = .001
    stability, summary = stability_tables(reference, result, config["alpha"])
    assert len(stability) == 1
    assert stability.exposure.iloc[0] == reference.loc[selected, "exposure"]
    assert summary.extension_q_all84_below_alpha.eq(60).all()
    assert summary.reference_hits_retained.le(1).all()
    assert summary.new_significant_outside_reference.eq(59).all()
    key = reference.loc[selected, KEY].tolist()
    target = result.model.eq("strict")
    for name, value in zip(KEY, key):
        target &= result[name].eq(value)
    result.loc[target, "interaction_beta_native"] = -reference.loc[selected, "interaction_beta_native"]
    stability, _ = stability_tables(reference, result, config["alpha"])
    assert not stability.same_direction_all_six.iloc[0]
    assert not stability.retained_all_six.iloc[0]


def test_tampered_reference_output_or_core_code_is_rejected(case, tmp_path):
    config, _, metadata, original, _, _, _, _ = case
    root = tmp_path / "reference"
    (root / "tables").mkdir(parents=True)
    write_json(root / "analysis_config.json", config)
    original.to_csv(root / "tables/all_interactions.csv", index=False)
    manifest = dict(version=config["version"], data_kind="SYNTHETIC", status="COMPLETE_WITH_LIMITATIONS",
                    config=config, inputs=metadata["files"],
                    code_sha256={name: sha256(PACKAGE / name) for name in CORE_FILES},
                    output_sha256={name: sha256(root / name) for name in ["analysis_config.json", "tables/all_interactions.csv"]})
    write_json(root / "manifest.json", manifest)
    load_reference(root, PACKAGE, expected_kind="SYNTHETIC")
    with pytest.raises(InputError, match="data kind"):
        load_reference(root, PACKAGE, expected_kind="SERVER_INPUTS")
    table = root / "tables/all_interactions.csv"
    original_bytes = table.read_bytes()
    table.write_bytes(original_bytes + b"\n")
    with pytest.raises(InputError, match="Reference output changed"):
        load_reference(root, PACKAGE, expected_kind="SYNTHETIC")
    table.write_bytes(original_bytes)
    manifest["code_sha256"][CORE_FILES[0]] = "bad"
    write_json(root / "manifest.json", manifest)
    with pytest.raises(InputError, match="Core analysis code"):
        load_reference(root, PACKAGE, expected_kind="SYNTHETIC")


def test_no_reference_hits_still_reports_all_extensions(case):
    config, _, _, reference, results, _, _, _ = case
    reference = reference.copy()
    reference.loc[:, "q_all"] = 1.0
    stability, summary = stability_tables(reference, results, config["alpha"])
    assert stability.empty and "retained_all_six" in stability
    assert len(summary) == 6 and summary.extension_planned.eq(60).all()
    assert summary.reference_hits.eq(0).all()


def test_cli_relocation_preserves_reference_and_rejects_input_drift(case, tmp_path):
    config, _, metadata, original, _, _, _, _ = case
    root = tmp_path / "renamed Data root"
    package = root / "diet_gut_cgm_interaction" / PACKAGE.name
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("outputs*", "__pycache__", ".pytest_cache"))
    for source in metadata["files"]:
        target = root / config["inputs"][source["label"]]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["path"], target)
    reference = package.parent / "outputs" / "original fixture"
    (reference / "tables").mkdir(parents=True)
    write_json(reference / "analysis_config.json", config)
    original.to_csv(reference / "tables/all_interactions.csv", index=False)
    # Synthetic test fixture labelled SERVER_INPUTS solely to exercise the real CLI branch.
    manifest = dict(version=config["version"], data_kind="SERVER_INPUTS", status="COMPLETE_WITH_LIMITATIONS",
                    fixture_notice="SYNTHETIC TEST FIXTURE ONLY", config=config, inputs=metadata["files"],
                    code_sha256={name: sha256(package / name) for name in CORE_FILES},
                    output_sha256={name: sha256(reference / name) for name in ["analysis_config.json", "tables/all_interactions.csv"]})
    write_json(reference / "manifest.json", manifest)
    frozen = {str(p.relative_to(reference)): sha256(p) for p in reference.rglob("*") if p.is_file()}
    assert not (package.parent / "interaction").exists()
    assert not (package.parent / "run_analysis.py").exists()
    command = ["bash", str(package / "run_extension_sensitivity.sh"), "--reference-run", "../outputs/original fixture", "--skip-plots"]
    env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    out = tmp_path / "followup"
    process = subprocess.run(command + ["--output-dir", str(out)], cwd=str(package), env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    result = pd.read_csv(out / "tables/all_interactions.csv")
    assert len(result) == 588
    current_manifest = json.loads((out / "manifest.json").read_text())
    assert Path(current_manifest["data_root"]) == root
    assert current_manifest["reference_reproduced_tests"] == 204
    assert all(sha256(reference / name) == digest for name,digest in frozen.items())
    for name,digest in current_manifest["output_sha256"].items():
        assert sha256(out / name) == digest
    pointer = package / "outputs_extension_sensitivity/LATEST_REAL.txt"
    assert pointer.read_text().strip() == str(out)
    screen = subprocess.run([sys.executable, str(package / "show_results.py"), "--screen", "1"],
                            cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert screen.returncode == 0, screen.stderr
    assert "Reference reproduction=204/204" in screen.stdout
    assert "EXTENSION 1/3" in screen.stdout
    for table in (out / "tables").glob("*.csv"):
        text = table.read_text(encoding="utf-8-sig")
        assert "participant_id" not in text and "SYNTH_000" not in text
    first = root / config["inputs"]["current4"]
    first.write_bytes(first.read_bytes() + b"\n")
    failed = tmp_path / "changed_input"
    process = subprocess.run(command + ["--output-dir", str(failed)], cwd=str(package), env=env, capture_output=True, text=True)
    assert process.returncode == 2
    assert "Input identity differs" in process.stderr
    assert json.loads((failed / "manifest.json").read_text())["status"] == "FAILED"
    assert pointer.read_text().strip() == str(out)
    assert all(sha256(reference / name) == digest for name,digest in frozen.items())
    nested = subprocess.run(command + ["--output-dir", str(reference / "child")], cwd=str(package), env=env, capture_output=True, text=True)
    assert nested.returncode == 2 and not (reference / "child").exists()


@pytest.mark.parametrize("unsafe_path", ["/do-not-write-real-input.csv", "../do-not-write-real-input.csv"])
def test_demo_refuses_unsafe_input_destinations(tmp_path, unsafe_path):
    package = tmp_path / "package"
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("outputs*", "__pycache__", ".pytest_cache"))
    config = load_config(PACKAGE / "config/analysis.json")
    config["inputs"]["current4"] = unsafe_path
    write_json(package / "config/analysis.json", config)
    result = subprocess.run([sys.executable, str(package / "run_extension_sensitivity.py"),
                             "--demo", "--output-dir", str(tmp_path / "out"), "--skip-plots"],
                            cwd=str(tmp_path), capture_output=True, text=True)
    assert result.returncode == 2
    assert "safe relative" in result.stderr
    assert not (tmp_path / "out/reference_demo").exists()
