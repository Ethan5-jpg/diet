"""Version and source checks. No confirmation outcome paths are accepted."""

import importlib
from importlib.metadata import version
from hashlib import sha256
import json
from pathlib import Path
import sys

from exp2_store import read_json, sha, verify_file


ROOT = Path(__file__).resolve().parent
V12 = "diet_effect_modification_20261109_v12"
# The version sits under the experiment directory, alongside V12 in TRE.
# The local archive keeps V12 under its experiments subfolder.
EXPERIMENT_PARENT = ROOT.parent.parent
EXPERIMENTS = (EXPERIMENT_PARENT if (EXPERIMENT_PARENT / V12).is_dir()
               else EXPERIMENT_PARENT / "experiments")
PACKAGE = EXPERIMENTS.parent
V4 = "diet_nested_stability_20261103_v4"
V1 = "diet_response_heterogeneity_20261103_v1_attempt02"
V3 = "diet_representation_screen_20261103_v3"
RUNTIME_FILES = {
    "config.json", "dependency_manifest.json", "server_run.sh",
    "exp2_design.py", "exp2_engine.py", "exp2_expansion.py", "exp2_followup.py", "exp2_inputs.py",
    "exp2_models.py", "exp2_native_fixture.py", "exp2_report.py", "exp2_run.py",
    "exp2_score.py", "exp2_store.py", "exp2_test.py",
}


def environment(cfg):
    expected = cfg["dependencies"]
    found = {name: version(name) for name in [*expected, "threadpoolctl"]}
    mismatch = [name for name in expected if found[name] != expected[name]]
    if mismatch:
        raise RuntimeError("Use the existing pinned HPP environment; version mismatch: " +
                           ", ".join("%s=%s (need %s)" % (m, found[m], expected[m]) for m in mismatch))
    return found


def preflight():
    manifest_bytes = (ROOT / "upload_manifest.json").read_bytes()
    upload = json.loads(manifest_bytes)
    if set(upload.get("files", {})) != RUNTIME_FILES:
        raise ValueError("Upload manifest must cover every runtime file exactly")
    for name, digest in upload["files"].items():
        verify_file(ROOT / name, digest)
    anchors = read_json(ROOT / "dependency_manifest.json")
    for name, digest in anchors["experiments"].items():
        verify_file(EXPERIMENTS / name, digest)
    config_bytes = (ROOT / "config.json").read_bytes()
    if sha256(config_bytes).hexdigest() != upload["files"]["config.json"]:
        raise ValueError("Configuration changed during release verification")
    return len(upload["files"]), sha256(manifest_bytes).hexdigest(), json.loads(config_bytes)


def adapters():
    folder = EXPERIMENTS / V12
    sys.path.insert(0, str(folder))
    try:
        result = tuple(importlib.import_module(n) for n in ("bridge", "features"))
    finally:
        sys.path.pop(0)
    if any(Path(m.__file__).resolve().parent != folder.resolve() for m in result):
        raise ImportError("Loaded the wrong historical adapter")
    return result


def synthetic_module():
    folder = EXPERIMENTS / V12
    sys.path.insert(0, str(folder))
    try:
        result = importlib.import_module("synthetic")
    finally:
        sys.path.pop(0)
    if Path(result.__file__).resolve().parent != folder.resolve():
        raise ImportError("Loaded the wrong synthetic adapter")
    return result


def native_paths():
    """Exact allowlist: old discovery inputs, role preprocessing, and V10 covariates."""
    inputs = PACKAGE / "inputs/discovery_v2"
    paths = [inputs / n for n in ("discovery_covariates_food_TRE_ONLY.parquet",
                                 "D_Glu_raw_Y_TRE_ONLY.parquet",
                                 "oral_raw_TRE_ONLY.npz", "oral_taxa.json",
                                 "gut_raw_TRE_ONLY.npz", "gut_taxa.json")]
    paths += [EXPERIMENTS / V4 / "results/TRE_ONLY/final_predictions.npz",
              EXPERIMENTS / V4 / "run_nested.py",
              EXPERIMENTS / V1 / "code/core.py", EXPERIMENTS / V1 / "code/run_pilot.py",
              EXPERIMENTS / V3 / "run_screen.py",
              PACKAGE / "preprocessing/code/preprocessing_v1_20261030.py",
              PACKAGE / "inputs/daily/food_days_discovery_TRE_ONLY.parquet",
              PACKAGE.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet",
              PACKAGE.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Required existing discovery source missing: " + str(path))
    anchors = read_json(ROOT / "dependency_manifest.json")
    verify_file(PACKAGE / "preprocessing/code/preprocessing_v1_20261030.py",
                anchors["preprocessing_sha"])
    return paths


def fingerprint(paths):
    return {str(path): sha(path) for path in paths}


def source_state(paths):
    """File identity and modification stamps for checks before fold commits."""
    result = {}
    for path in paths:
        info = path.stat()
        result[path] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    return result


def validate_population(dataset, cfg):
    import numpy as np

    rows = len(dataset.index)
    ref = dataset.reference
    outer = np.asarray(ref["outer"])
    household = dataset.index.split_unit_id
    if household.isna().any() or household.astype(str).str.strip().eq("").any():
        raise ValueError("Missing household identifier")
    group = household.astype(str).to_numpy()
    if rows != cfg["expected_discovery_rows"] or ref["y"].shape != (2, rows) or outer.shape != (2, rows):
        raise ValueError("Unexpected discovery shape")
    if not np.issubdtype(outer.dtype, np.integer) or np.asarray(dataset.y).shape != (rows,):
        raise ValueError("Unexpected raw outcome shape or outer assignment type")
    eligible = np.isfinite(ref["y"])
    if not np.array_equal(eligible[0], eligible[1]):
        raise ValueError("Different evaluation people across repeats")
    if not np.isfinite(np.asarray(dataset.y)[eligible[0]]).all():
        raise ValueError("A frozen eligible row has a nonfinite raw outcome")
    if eligible[0].sum() != cfg["expected_people"] or len(set(group[eligible[0]])) != cfg["expected_households"]:
        raise ValueError("Frozen eligible population changed")
    for rep in range(2):
        if set(outer[rep]) != set(range(cfg["outer_folds"])):
            raise ValueError("Unexpected outer split identifiers")
        for fold in range(cfg["outer_folds"]):
            valid = outer[rep] == fold
            dataset.api.separated(group, ~valid, valid)
            if not np.any(valid & eligible[rep]):
                raise ValueError("Empty outer fold")
