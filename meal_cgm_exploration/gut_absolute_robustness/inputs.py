"""Read-only, pinned original participant summaries; no raw CGM ingestion."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hppmeal.common import require, sha256
from seven_diet_meals.inputs import CATEGORIES
from seven_diet_meals.models import sample_hash
from gut_meal_bridge_v2.analysis import model_columns
from gut_meal_bridge_v2.inputs import contract as original_contract
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent
SOURCE_FILES = ["tables/models.csv", "tables/sample.csv",
                "private/person_inputs.csv", "private/meal_membership.csv"]
DEP_FILES = ["hppmeal/__init__.py", "hppmeal/common.py", "hppmeal/models.py",
             "seven_diet_meals/__init__.py", "seven_diet_meals/inputs.py",
             "seven_diet_meals/models.py", "seven_diet_meals/sources.json",
             "gut_meal_bridge_v2/__init__.py", "gut_meal_bridge_v2/inputs.py",
             "gut_meal_bridge_v2/analysis.py", "gut_meal_bridge_v2/CONTRACT.json"]
MODEL_KEYS = ["setting", "taxon", "outcome"]


def contract():
    return json.loads((MODULE / "CONTRACT.json").read_text(encoding="utf-8"))


def unchanged(pins):
    for name, digest in pins.items():
        path = Path(name)
        require(path.is_file() and sha256(path) == digest, "Missing/changed pinned file: " + str(path))


def bundle():
    path = MODULE / "MANIFEST.json"
    require(path.is_file(), "Upload the entire gut_absolute_robustness folder; MANIFEST.json missing")
    body = json.loads(path.read_text(encoding="utf-8"))
    required = {"__init__.py", "CONTRACT.json", "inputs.py", "analysis.py", "reporting.py", "run_analysis.py"}
    require(body.get("version") == VERSION and required.issubset(body.get("files", {})), "Incomplete module manifest")
    require(set(DEP_FILES) == set(body.get("dependencies", {})), "Incomplete dependency manifest")
    pins = {str(path): sha256(path)}
    for section, root in [("files", MODULE), ("dependencies", PACKAGE)]:
        for name, digest in body[section].items():
            f = (root / name).resolve()
            require(root in f.parents, "Invalid manifest path")
            pins[str(f)] = digest
    unchanged(pins)
    return pins


def booleans(series, label):
    values = series.astype(str).str.strip().str.lower()
    result = values.map({"true": True, "false": False, "1": True, "0": False,
                         "1.0": True, "0.0": False})
    require(result.notna().all(), "Invalid/missing boolean in " + label)
    return result.astype(bool)


def selected(persons, members, setting, minimum_meals=5, minimum_dates=3):
    require(setting in contract()["settings"], "Unknown sample setting")
    keep = persons.meals.ge(minimum_meals) & persons.dates.ge(minimum_dates)
    if setting == "strict_common":
        keep &= persons.strict_eligible
    p = persons.loc[keep].sort_values("participant_id").reset_index(drop=True).copy()
    m = members.loc[members.participant_id.isin(p.participant_id)].sort_values("meal_id").reset_index(drop=True).copy()
    return p, m


def validate_frames(persons, members, models, samples):
    c, old = contract(), original_contract()
    require(old["version"] == c["source_version"] and [t["id"] for t in old["taxa"]] == c["taxa"], "Original contract/taxa differ")
    require(old["minimum_meals"] == 5 and old["minimum_dates"] == 3, "Original meal support differs")
    require(old["minimum_people"] == c["minimum_people"] and old["minimum_residual_df"] == c["minimum_residual_df"], "Inference limits differ")
    for frame, ids, label in [(persons, ["participant_id"], "person inputs"),
                              (members, ["participant_id", "meal_id"], "meal membership")]:
        for name in ids:
            require(name in frame and frame[name].notna().all() and frame[name].astype(str).str.strip().ne("").all(), "Missing identity in " + label)
    require(not persons.participant_id.duplicated().any(), "Duplicate person inputs")
    require(not members.meal_id.duplicated().any(), "Duplicate meal membership")
    require(set(persons.participant_id) == set(members.participant_id), "Person/member identities do not match")
    for frame, label in [(persons, "persons"), (members, "members")]:
        require("strict_eligible" in frame, "Missing strict flag")
        frame["strict_eligible"] = booleans(frame.strict_eligible, label)
    flags = members.participant_id.map(persons.set_index("participant_id").strict_eligible)
    require(flags.eq(members.strict_eligible).all(), "Person/member strict flags differ")
    needed = set(c["taxa"] + old["meal_outcomes"] + old["reference_outcomes"] + ["meals", "dates"])
    for outcome in old["meal_outcomes"] + old["reference_outcomes"]:
        needed.update(model_columns(c["taxa"][0], outcome))
    require(needed.issubset(persons.columns), "Missing required original person fields: " + ",".join(sorted(needed - set(persons.columns))))
    for name in needed:
        require(persons[name].notna().all(), "Missing original common variable: " + name)
        if name in CATEGORIES:
            require(persons[name].astype(str).str.strip().ne("").all(), "Empty category: " + name)
        else:
            persons[name] = pd.to_numeric(persons[name], errors="raise")
            require(np.isfinite(persons[name]).all(), "Nonfinite original common variable: " + name)
    for name, lower in [("meals", 5), ("dates", 3)]:
        require((persons[name].ge(lower) & persons[name].mod(1).eq(0)).all(), "Invalid original support counts: " + name)
    require(persons.dates.le(persons.meals).all(), "More meal dates than meals")
    meal_counts = members.groupby("participant_id").size()
    require(persons.participant_id.map(meal_counts).eq(persons.meals).all(), "Frozen person meal counts do not match membership")
    expected = {(s, t, o) for s in c["settings"] for t in c["taxa"] for o in old["meal_outcomes"] + old["reference_outcomes"]}
    require(len(models) == 48 and not models.duplicated(MODEL_KEYS).any() and
            set(models[MODEL_KEYS].itertuples(index=False, name=None)) == expected, "Original 48 model specifications incomplete")
    require(set(models.status).issubset({"ESTIMATED", "NOT_ESTIMATED"}), "Unrecognized source model status")
    require(not samples.setting.duplicated().any() and set(samples.setting) == set(c["settings"]), "Original sample rows incomplete")
    for setting in c["settings"]:
        p, m = selected(persons, members, setting)
        expected_values = dict(people=len(p), meals=len(m), person_sha256=sample_hash(p.participant_id), meal_sha256=sample_hash(m.meal_id))
        row = samples.set_index("setting").loc[setting]
        part = models.loc[models.setting.eq(setting)]
        for name, value in expected_values.items():
            require(name in part and row[name] == value and part[name].eq(value).all(), "Original common membership/count mismatch: " + setting + "/" + name)
    persons["absolute_post_mean"] = persons.baseline_glucose + persons.mean_increment
    require(np.isfinite(persons.absolute_post_mean).all(), "Nonfinite absolute post-meal mean")
    return persons, members


def load_source(source):
    source = Path(source).expanduser().resolve()
    manifest = source / "manifest.json"
    require(manifest.is_file(), "Original gut run manifest missing: " + str(manifest))
    body = json.loads(manifest.read_text(encoding="utf-8"))
    state = body.get("status", {}).get("gut_meal_bridge_v2", {})
    require(state.get("status") in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"], "Original gut run must be complete")
    require(state.get("version") == contract()["source_version"], "Original gut version mismatch")
    require(body.get("data_kind") in ["REAL", "SYNTHETIC"], "Unverified source data label")
    pins = {str(manifest): sha256(manifest)}
    frames = {}
    strings = {x: str for x in CATEGORIES + ["participant_id", "meal_id", "reference_connection", "person_sha256", "meal_sha256"]}
    for name in SOURCE_FILES:
        path = source / name
        digest = state.get("result_sha256", {}).get(name)
        require(digest and path.is_file() and sha256(path) == digest, "Missing/changed original result: " + name)
        pins[str(path)] = digest
        frames[name] = pd.read_csv(path, dtype=strings, float_precision="round_trip")
    persons, members = validate_frames(frames["private/person_inputs.csv"], frames["private/meal_membership.csv"],
                                      frames["tables/models.csv"], frames["tables/sample.csv"])
    require(int(frames["tables/models.csv"].status.eq("ESTIMATED").sum()) == state.get("estimated"), "Original estimated count mismatch")
    unchanged(pins)
    return persons, members, frames["tables/models.csv"], frames["tables/sample.csv"], body["data_kind"], pins
