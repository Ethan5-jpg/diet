"""Read-only, hash-pinned interfaces to the existing HPP analysis."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from hppmeal.common import (KEYS, require, sha256, header, read_csv, norm,
                            unique, select_participant_scope)
from seven_diet_meals import inputs as seven
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent


def contract():
    return json.loads((MODULE / "CONTRACT.json").read_text(encoding="utf-8"))


def bundle():
    path = MODULE / "MANIFEST.json"
    require(path.is_file(), "Upload the complete gut_meal_bridge_v2 folder (MANIFEST.json missing)")
    body = json.loads(path.read_text(encoding="utf-8"))
    needed = {"__init__.py", "inputs.py", "analysis.py", "reporting.py", "run_analysis.py",
              "run_gut_meal_bridge_v2.sh", "CONTRACT.json", "sources.json", "PLAN.md", "README.md", "REVIEW.md"}
    require(body.get("version") == VERSION and needed.issubset(body.get("files", {})), "Incomplete upload manifest")
    dependencies = {"hppmeal/__init__.py", "hppmeal/common.py", "hppmeal/models.py",
                    "seven_diet_meals/__init__.py", "seven_diet_meals/inputs.py",
                    "seven_diet_meals/models.py", "seven_diet_meals/sources.json"}
    require(dependencies.issubset(body.get("dependencies", {})), "Incomplete dependency manifest")
    expected = {str(path): sha256(path)}
    for section, root in [("files", MODULE), ("dependencies", PACKAGE)]:
        for name, digest in body[section].items():
            p = (root / name).resolve()
            require(root in p.parents and p.is_file() and sha256(p) == digest,
                    "Missing/changed upload or dependency: " + str(p))
            expected[str(p)] = digest
    return expected


def resolve_species(columns, taxa):
    """Exact species token only; never select a genus, substring or first hit."""
    rows = []
    for taxon in taxa:
        token = taxon["species"]
        found = [c for c in columns if c == token or ("s__" + token) in c.split("|")]
        require(len(found) == 1, "{}: expected one exact species column, found {} ({})".format(
            taxon["id"], len(found), token))
        rows.append(dict(taxon=taxon["id"], species=token, column=found[0], scale="original CLR-Z"))
    require(len(set(x["column"] for x in rows)) == len(rows), "Taxa map to duplicate columns")
    return rows


def inspect(meal_run, nutrition_run, root_override, sources_config, score_config):
    # The seven-score interface verifies original meal BUILD and covariate hashes.
    root, config, paths, expected, audit = seven.inspect(meal_run, root_override, score_config)
    settings_path = Path(sources_config).expanduser().resolve()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    require(set(settings) == {"whole_cgm", "microbiome_dates"}, "Invalid sources configuration keys")
    expected[str(settings_path)] = sha256(settings_path)
    old = json.loads((Path(meal_run) / "manifest.json").read_text(encoding="utf-8"))
    micro = (root / config["inputs"]["microbiome"]).resolve()
    entries = [x for x in old["inputs"] if x["label"] == "microbiome"]
    require(len(entries) == 1, "Original meal run must record the microbiome source/hash")
    require(micro.is_file() and sha256(micro) == entries[0]["sha256"], "Microbiome differs from original meal run")
    require(set(KEYS).issubset(header(micro)), "Microbiome cohort/visit fields required")
    mapping = resolve_species(header(micro), contract()["taxa"])
    paths["microbiome"] = micro
    expected[str(micro)] = entries[0]["sha256"]

    nf = Path(nutrition_run).expanduser().resolve() / "manifest.json"
    require(nf.is_file(), "Completed nutrition_bridge manifest missing: " + str(nf))
    n = json.loads(nf.read_text(encoding="utf-8"))
    require(n["status"].get("nutrition_bridge", {}).get("status") in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"],
            "nutrition_bridge must have completed")
    require(not any(x.get("status") in ["FAILED", "RUNNING", "INTERRUPTED"] for x in n["status"].values()),
            "nutrition_bridge has an incomplete stage")
    require(n["data_kind"] == config["data_kind"] and n["config"] == config, "Source lineage configuration/data-kind differs")
    cg = (root / Path(settings["whole_cgm"]).expanduser()).resolve()
    # Relocate the old Data root only; require the exact recorded finalized table.
    pinned = {}
    for item in n["inputs"]:
        original = Path(item["path"])
        try:
            original = root / original.relative_to(Path(n["root"]))
        except ValueError:
            pass
        key = str(original.resolve())
        require(key not in pinned or pinned[key] == item["sha256"], "Conflicting historical input hashes: " + key)
        pinned[key] = item["sha256"]
    for label in ["current4", "new3", "corrected_hpdi", "covariates", "meals"]:
        key = str(paths[label])
        require(key in pinned and sha256(paths[label]) == pinned[key],
                "Input must match the finalized source recorded in nutrition_bridge: " + label)
        expected[key] = pinned[key]
    require(str(cg) in pinned and cg.is_file() and sha256(cg) == pinned[str(cg)],
            "Whole-CGM table must match the finalized input recorded in nutrition_bridge")
    required = KEYS + ["connection_id"] + contract()["reference_outcomes"]
    require(set(required).issubset(header(cg)), "Finalized CGM table missing identity or original Z columns")
    paths["whole_cgm"] = cg
    expected.update({str(nf): sha256(nf), str(cg): pinned[str(cg)]})

    date_value = settings["microbiome_dates"]
    if date_value == "auto":
        configured = config["inputs"].get("microbiome_dates")
        date_path = root / (configured or "Transfer/gut_microbiome/gut_microbiome.csv")
        if configured:
            require(date_path.is_file(), "Originally configured microbiome dates file is missing")
    elif date_value is None:
        date_path = None
    else:
        date_path = root / Path(date_value).expanduser()
        require(date_path.is_file(), "Explicit microbiome dates file is missing")
    paths["dates"] = date_path.resolve() if date_path is not None and date_path.is_file() else None
    if paths["dates"] is not None:
        expected[str(paths["dates"])] = sha256(paths["dates"])
    for path, digest in expected.items():
        require(Path(path).is_file() and sha256(path) == digest, "Input changed during inspection: " + path)
    return root, config, paths, expected, audit, mapping


def date_only(value, timezone):
    """Preserve local dates; convert explicitly offset timestamps to cohort timezone."""
    if pd.isna(value) or str(value).strip().lower() in ["", "nan", "none", "null"]:
        return pd.NaT
    text = str(value).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", text):
        return pd.NaT
    try:
        stamp = pd.Timestamp(text)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(timezone).tz_localize(None)
        return stamp.normalize()
    except (ValueError, TypeError):
        return pd.NaT


def timing_table(micro, metadata, config):
    """Audit dates without guessing which of repeated stool samples supplied CLR-Z."""
    out = micro[["participant_id"]].copy().set_index("participant_id")
    out["sample_date"] = pd.NaT
    out["timing_status"] = "UNKNOWN_NO_METADATA"
    if metadata is None:
        return out.reset_index()
    require(set(KEYS).issubset(metadata), "Date metadata must contain participant/cohort/visit keys")
    d = metadata.copy()
    for col in KEYS:
        d[col] = norm(d[col])
    d = d.loc[d.cohort.eq(config["cohort"]) & d.research_stage.eq(config["research_stage"]) & d.participant_id.notna()].copy()
    fields = [x for x in ["collection_date", "collection_timestamp"] if x in d]
    require(fields, "Date metadata has no collection_date/collection_timestamp")
    parsed = [d[x].apply(lambda v: date_only(v, config["time"]["local_timezone"])) for x in fields]
    dates = parsed[0]
    conflict = pd.Series(False, index=d.index)
    for other in parsed[1:]:
        conflict |= dates.notna() & other.notna() & dates.ne(other)
        dates = dates.fillna(other)
    d["sample_date"] = dates
    d["date_conflict"] = conflict
    if "sample_name" in d:
        d["sample_name"] = norm(d.sample_name)
        known = d.loc[d.sample_name.notna()]
        unique(known, KEYS + ["sample_name"], "date metadata sample identity")
    groups = {pid: g for pid, g in d.groupby("participant_id")}
    for _, person in micro.iterrows():
        pid = person.participant_id
        rows = groups.get(pid)
        if rows is None:
            out.loc[pid, "timing_status"] = "UNKNOWN_NO_MATCH"
            continue
        if "sample_name" in micro and "sample_name" in d and pd.notna(person.sample_name):
            rows = rows.loc[rows.sample_name.eq(person.sample_name)]
            if len(rows) == 0:
                out.loc[pid, "timing_status"] = "UNKNOWN_SAMPLE_MISMATCH"
                continue
        if len(rows) != 1:
            out.loc[pid, "timing_status"] = "AMBIGUOUS_MULTIPLE_SAMPLES"
            continue
        row = rows.iloc[0]
        if row.date_conflict:
            out.loc[pid, "timing_status"] = "UNKNOWN_DATE_CONFLICT"
        elif pd.isna(row.sample_date):
            out.loc[pid, "timing_status"] = "UNKNOWN_INVALID_DATE"
        else:
            out.loc[pid, "sample_date"] = row.sample_date
            out.loc[pid, "timing_status"] = ("MATCHED_SAMPLE" if "sample_name" in micro and
                "sample_name" in d and pd.notna(person.get("sample_name")) else "SINGLE_VISIT_RECORD")
    return out.reset_index()


def load(run, paths, mapping):
    meals, people = seven.load(run, paths)
    # Older loader versions may not retain connection_id: attach it explicitly.
    raw = read_csv(paths["meals"], ["meal_id", "connection_id"], usecols=["meal_id", "connection_id"])
    raw["meal_id"], raw["connection_id"] = norm(raw.meal_id), norm(raw.connection_id)
    unique(raw, ["meal_id"], "meal connection map")
    # The frozen BUILD intentionally retains rejected/no-CGM meals with blank
    # connection IDs. seven.load has already selected analysis_ready meals.
    ready = raw.meal_id.isin(meals.meal_id)
    run.table(pd.DataFrame([
        dict(scope=scope, meals=int(mask.sum()),
             missing_connection=int((mask & raw.connection_id.isna()).sum()))
        for scope, mask in [("analysis_ready", ready), ("not_analysis_ready", ~ready)]
    ]), "01_meal_connection_audit.csv")
    raw = raw.loc[ready].copy()
    require(len(raw) == len(meals), "Analysis-ready meal missing from connection map")
    missing = int(raw.connection_id.isna().sum())
    require(missing == 0, "Missing meal connection ID in analysis_ready meals: {}. See tables/01_meal_connection_audit.csv".format(missing))
    if "connection_id" in meals:
        comparison = meals[["meal_id", "connection_id"]].merge(raw, on="meal_id", how="left", validate="one_to_one")
        require(comparison.connection_id_x.eq(comparison.connection_id_y).all(), "Meal connection inconsistent")
    else:
        meals = meals.merge(raw, on="meal_id", how="left", validate="one_to_one")
    require(meals.groupby("participant_id").connection_id.nunique().le(1).all(),
            "Multiple meal CGM connections per participant: cannot average across connections")

    cols = KEYS + [x["column"] for x in mapping]
    if "sample_name" in header(paths["microbiome"]):
        cols.append("sample_name")
    micro = select_participant_scope(run, read_csv(paths["microbiome"], cols, usecols=cols), "microbiome")
    for item in mapping:
        micro[item["taxon"]] = seven.strict_number(micro[item["column"]], item["species"])
    if "sample_name" in micro:
        micro["sample_name"] = norm(micro.sample_name)
    metadata = read_csv(paths["dates"]) if paths["dates"] is not None else None
    timing = timing_table(micro, metadata, run.c)
    selected = micro[["participant_id"] + [x["taxon"] for x in mapping]].merge(timing, on="participant_id", validate="one_to_one")
    people = people.join(selected.set_index("participant_id"), how="outer")
    for flag in ["primary_eligible", "strict_eligible"]:
        people[flag] = people[flag].eq(True)
    refs = contract()["reference_outcomes"]
    cg = select_participant_scope(run, read_csv(paths["whole_cgm"], KEYS + ["connection_id"] + refs,
        usecols=KEYS + ["connection_id"] + refs), "whole_cgm")
    cg["connection_id"] = norm(cg.connection_id)
    require(cg.connection_id.notna().all(), "Missing reference CGM connection ID")
    for col in refs:
        cg[col] = seven.strict_number(cg[col], col)
    people = people.join(cg.set_index("participant_id")[["connection_id"] + refs].rename(
        columns={"connection_id": "reference_connection"}), how="outer")
    run.table(pd.DataFrame(mapping), "01_taxon_mapping.csv")
    return meals, people
