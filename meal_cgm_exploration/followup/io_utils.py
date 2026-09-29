"""Validate the completed source run without writing to it."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hppmeal.common import KEYS, Run, require, sha256, validate_config
from hppmeal.models import covariates
from . import VERSION

PACKAGE = Path(__file__).resolve().parents[1]
TEXT = KEYS + ["meal_id", "signature", "connection_id", "local_date"]
NUMERIC = ["start", "end", "local_hour", "duration_minutes", "calories_kcal",
           "carbohydrate_g", "previous_gap_minutes", "baseline_glucose",
           "pre_slope_per_min", "iauc_net_0_120", "peak_rise_0_120"]


def core_fingerprint():
    files = sorted((PACKAGE / "hppmeal").glob("*.py")) + sorted((PACKAGE / "scripts").glob("*.py"))
    return {str(p.relative_to(PACKAGE)): sha256(p) for p in files}


def verify_bundle():
    path = PACKAGE / "followup" / "MANIFEST.json"
    require(path.is_file(), "followup/MANIFEST.json missing; upload the complete followup folder")
    manifest_digest = sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"followup/" + name for name in ["__init__.py", "io_utils.py", "prediction.py",
                "repeats.py", "reporting.py", "run_repeated.py", "run_repeated.sh"]}
    require(isinstance(manifest.get("files"), dict) and required.issubset(manifest["files"]),
            "Incomplete followup manifest; upload the complete current followup folder")
    require(manifest.get("version") == VERSION, "Followup manifest/code version mismatch")
    for name, digest in manifest["files"].items():
        p = (PACKAGE / name).resolve()
        require(PACKAGE in p.parents and p.is_file(), "Invalid/missing followup package member: " + name)
        require(sha256(p) == digest, "Followup file changed/incomplete upload: " + name)
    require(sha256(path) == manifest_digest, "Followup manifest changed during inspection")
    manifest["_manifest_sha256"] = manifest_digest
    return manifest


def inspect_source(source):
    source = Path(source).resolve()
    manifest_path = source / "manifest.json"
    require(manifest_path.is_file(), "Source manifest missing")
    manifest_digest = sha256(manifest_path)
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(sha256(manifest_path) == manifest_digest, "Source manifest changed while being inspected")
    c = validate_config(old["config"])
    require(old["data_kind"] == c["data_kind"], "Source data-kind mismatch")
    for stage in ["audit", "build"]:
        require(old["status"].get(stage, {}).get("status") == "PASS", "Source {} must be PASS".format(stage))
    require(not any(v.get("status") in ["RUNNING", "FAILED", "INTERRUPTED"] for v in old["status"].values()),
            "Source run contains an incomplete or failed stage")
    require(c["time"]["consumption_time_verified"], "Use the reviewed analysis run with its recorded time interpretation")
    fingerprint = source / "code_fingerprint.json"
    require(fingerprint.is_file(), "Original code fingerprint missing")
    fingerprint_digest = sha256(fingerprint)
    core = json.loads(fingerprint.read_text())
    require(sha256(fingerprint) == fingerprint_digest, "Source fingerprint changed while being inspected")
    require(core == core_fingerprint(),
            "Original hppmeal/scripts code differs from source run; restore the original version")
    meals = source / "private" / "02_meals.csv"
    require(meals.is_file() and sha256(meals) == old["status"]["build"].get("meals_sha256"),
            "Source meal table is missing or changed since BUILD")
    # Preserve the digests actually inspected, rather than accepting a new version
    # if a file changes before Run.record/read_inputs gets to it.
    old["_validated_inputs"] = {
        str(manifest_path): manifest_digest,
        str(fingerprint): fingerprint_digest,
        str(meals): old["status"]["build"]["meals_sha256"]}
    old["_validated_inputs"].update({str(PACKAGE / name): digest for name, digest in core.items()})
    return old


def read_inputs(run, source, old):
    require("_validated_inputs" in old, "Source inspection must precede input loading")
    for name, expected in old["_validated_inputs"].items():
        path = run.record(Path(name), "verified_source_" + Path(name).name)
        observed = next(x["sha256"] for x in run.inputs if x["path"] == str(path))
        require(observed == expected, "Verified source changed before loading: " + Path(name).name)
    run.record(source / "private" / "02_repeated_meal_pairs.csv", "source_repeated_pairs")
    path = run.path("covariates")
    previous = [x for x in old["inputs"] if x["label"] == "covariates"]
    require(len(previous) == 1, "Source must record one covariate input; complete its analyze stage first")
    require(path is not None and path.is_file() and sha256(path) == previous[0]["sha256"],
            "Covariate table missing/changed since original analysis; --data-root must point to the same data")
    cov = covariates(run)  # All tables/records go to the NEW run via this adapter.
    captured = next(x["sha256"] for x in run.inputs if x["path"] == str(path.resolve()))
    require(captured == previous[0]["sha256"], "Covariate source changed before loading")
    d = pd.read_csv(source / "private" / "02_meals.csv", usecols=TEXT + NUMERIC + ["analysis_ready"],
                    dtype={k: str for k in TEXT + ["analysis_ready"]})
    require(not d.meal_id.isna().any() and d.meal_id.is_unique, "Missing/duplicate meal IDs")
    require(not d[KEYS].isna().any().any(), "Missing meal identity")
    flag = d.analysis_ready.str.strip().str.lower()
    require(flag.isin(["true", "false"]).all(), "Invalid analysis_ready values")
    d["analysis_ready"] = flag.eq("true")
    for col in NUMERIC:
        d[col] = pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    require(d.cohort.eq(run.c["cohort"]).all() and d.research_stage.eq(run.c["research_stage"]).all(),
            "Meal cohort/visit differs from configured source scope")
    d = d.merge(cov, on="participant_id", how="left", validate="many_to_one")
    d["primary_eligible"] = d.primary_eligible.eq(True)
    d["strict_eligible"] = d.strict_eligible.eq(True)
    d["mean_increment"] = d.iauc_net_0_120 / 120.0
    d["mean_glucose"] = d.mean_increment + d.baseline_glucose
    old_pairs = pd.read_csv(source / "private" / "02_repeated_meal_pairs.csv",
                            dtype={x: str for x in ["participant_id", "meal_a", "meal_b"]})
    return d, old_pairs


def verify_unchanged(run):
    for item in run.inputs:
        require(sha256(item["path"]) == item["sha256"], "An input changed during followup: " + item["label"])


def cohort_meals(d, cohort):
    keep = d.analysis_ready.copy()
    if cohort != "all_descriptive":
        keep &= d[cohort + "_eligible"]
    return d.loc[keep].copy()
