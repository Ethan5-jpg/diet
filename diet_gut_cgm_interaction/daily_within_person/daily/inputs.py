"""Read frozen participant sources; raw input adapters never guess IDs from filenames."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (BASE, CATEGORICAL, KEYS, NUTRIENTS, date_only, header, ids,
                     norm, number, read_csv, require, scoped, unique)


def paths_for(root, config):
    root = Path(root).resolve()
    paths = {k: (root / Path(v).expanduser()).resolve() if v else None for k, v in config["inputs"].items()}
    paths["cgm_files"] = [Path(p).resolve() for p in sorted(glob.glob(str(paths["cgm_glob"])))]
    return paths


def species_mapping(columns, taxa):
    rows = []
    for item in taxa:
        hits = [x for x in columns if x == item["species"] or ("s__" + item["species"]) in x.split("|")]
        require(len(hits) == 1, item["id"] + ": expected one exact species column, found " + str(len(hits)))
        rows.append(dict(taxon=item["id"], species=item["species"], column=hits[0], scale="original CLR-Z"))
    return rows


def preflight(paths, config):
    rows = []
    needs = {
        "diet": KEYS + ["collection_timestamp", "collection_date", "food_id"] + NUTRIENTS,
        "nova_mapping": ["food_id", "nova_group", "nova_mapping_status"],
        "microbiome": KEYS,
        "connections": KEYS + ["connection_id"],
        "covariates": ["participant_id", "cohort", "exclude_diabetes_or_a10"] + BASE
    }
    for label, required in needs.items():
        path = paths[label]
        issues = []
        if not path.is_file():
            issues = ["MISSING_FILE"]
        else:
            try:
                cols = header(path)
                issues = sorted(set(required) - set(cols))
                if label == "covariates" and not {"research_stage", "cgm_research_stage"}.intersection(cols):
                    issues.append("visit_identity")
                if label == "microbiome":
                    species_mapping(cols, config["taxa"])
            except ValueError as exc:
                issues.append(str(exc))
        rows.append(dict(source=label, status="MISSING_OR_INVALID" if issues else "OK", detail="; ".join(issues), path=str(path)))
    bad = 0
    for p in paths["cgm_files"]:
        try:
            require(set(KEYS + ["connection_id", "collection_timestamp", "glucose"]).issubset(header(p)), "identity/time/glucose columns missing")
        except ValueError:
            bad += 1
    rows.append(dict(source="cgm_files", status="OK" if paths["cgm_files"] and not bad else "MISSING_OR_INVALID",
                     detail="files={}; invalid_headers={}".format(len(paths["cgm_files"]), bad), path=str(paths["cgm_glob"])))
    p = paths["microbiome_dates"]
    status = "OPTIONAL_ABSENT" if p is None or not p.is_file() else "OPTIONAL_PRESENT"
    if status == "OPTIONAL_PRESENT" and not set(KEYS).issubset(header(p)):
        status = "OPTIONAL_UNUSABLE_IDENTITY"
    rows.append(dict(source="microbiome_dates", status=status, detail="No date means unknown timing, never inferred baseline", path=str(p or "")))
    return pd.DataFrame(rows)


def load_mapping(path):
    d = read_csv(path, ["food_id", "nova_group", "nova_mapping_status"])
    d["food_id"] = ids(d.food_id)
    unique(d, ["food_id"], "NOVA mapping")
    group = number(d.nova_group)
    keep = norm(d.nova_mapping_status).eq("mapped")
    require(group[keep].isin([1, 2, 3, 4]).all(), "Invalid mapped NOVA code")
    if "nova_analysis_status" in d:
        keep &= ~d.nova_analysis_status.str.startswith("excluded", na=False)
    return dict(zip(d.loc[keep, "food_id"], group.loc[keep]))


def timing_table(micro, path, config):
    out = micro[["participant_id"]].copy()
    out["sample_date"] = ""
    out["timing_status"] = "UNKNOWN_NO_METADATA"
    if path is None or not path.is_file():
        return out
    if not set(KEYS).issubset(header(path)):
        out["timing_status"] = "UNKNOWN_METADATA_IDENTITY"
        return out
    d = scoped(read_csv(path), config, "microbiome_dates", [], unique_person=False, allow_empty=True)
    if d.empty:
        out["timing_status"] = "UNKNOWN_NO_MATCHING_VISIT"
        return out
    fields = [x for x in ["collection_date", "collection_timestamp"] if x in d]
    if not fields:
        out["timing_status"] = "UNKNOWN_NO_DATE_FIELD"
        return out
    parsed = [date_only(d[x], config["time"]["local_timezone"]) for x in fields]
    d["sample_date"] = parsed[0]
    d["conflict"] = False
    for s in parsed[1:]:
        d["conflict"] |= d.sample_date.notna() & s.notna() & d.sample_date.ne(s)
        d["sample_date"] = d.sample_date.fillna(s)
    if "sample_name" in d:
        d["sample_name"] = norm(d.sample_name)
    groups = {pid: g for pid, g in d.groupby("participant_id")}
    for index, person in micro.iterrows():
        rows = groups.get(person.participant_id)
        mask = out.participant_id.eq(person.participant_id)
        if rows is None:
            out.loc[mask, "timing_status"] = "UNKNOWN_NO_MATCH"
            continue
        matched = "sample_name" in micro and "sample_name" in d and pd.notna(person.get("sample_name"))
        if matched:
            rows = rows.loc[rows.sample_name.eq(person.sample_name)]
        if len(rows) != 1:
            out.loc[mask, "timing_status"] = "UNKNOWN_AMBIGUOUS_OR_MISMATCHED_SAMPLE"
        elif rows.iloc[0].conflict or pd.isna(rows.iloc[0].sample_date):
            out.loc[mask, "timing_status"] = "UNKNOWN_INVALID_OR_CONFLICTING_DATE"
        else:
            out.loc[mask, "sample_date"] = rows.iloc[0].sample_date
            out.loc[mask, "timing_status"] = "MATCHED_SAMPLE" if matched else "SINGLE_VISIT_RECORD"
    return out


def load_people(paths, config):
    audit = []
    cg_cols = KEYS + ["connection_id"]
    cg_extra = [x for x in ["cgm_device_type"] if x in header(paths["connections"])]
    cg = scoped(read_csv(paths["connections"], cg_cols, usecols=cg_cols + cg_extra), config, "connections", audit)
    cg["connection_id"] = ids(cg.connection_id)
    unique(cg, ["participant_id", "connection_id"], "selected connections")
    cov = scoped(read_csv(paths["covariates"]), config, "covariates", audit)
    for col in BASE + ["exclude_diabetes_or_a10", "known_diabetes", "a10_medication_use"]:
        if col not in cov:
            require(col not in BASE, "Missing covariate: " + col)
            continue
        cov[col] = norm(cov[col]) if col in CATEGORICAL else number(cov[col], col, strict=True)
    for col in ["exclude_diabetes_or_a10", "known_diabetes", "a10_medication_use", "vitamin_use", "hormone_use"]:
        if col in cov:
            require(cov[col].dropna().isin([0, 1]).all(), "Nonbinary flag: " + col)
    flag = cov.exclude_diabetes_or_a10
    for col in ["known_diabetes", "a10_medication_use"]:
        if col in cov:
            require(not (cov[col].eq(1) & ~flag.eq(1)).any(), "Diabetes/A10 union conflict")
    if {"known_diabetes", "a10_medication_use"}.issubset(cov):
        expected = pd.Series(np.nan, index=cov.index)
        expected.loc[cov.known_diabetes.eq(1) | cov.a10_medication_use.eq(1)] = 1
        expected.loc[cov.known_diabetes.eq(0) & cov.a10_medication_use.eq(0)] = 0
        require((flag.eq(expected) | (flag.isna() & expected.isna())).all(), "Three-state exclusion union conflict")
    if "connection_id" in cov:
        cv = cov[["participant_id", "connection_id"]].copy()
        cv["connection_id"] = ids(cv.connection_id)
        pair = cg[["participant_id", "connection_id"]].merge(cv, on="participant_id", validate="one_to_one")
        require(not (pair.connection_id_y.notna() & pair.connection_id_x.ne(pair.connection_id_y)).any(), "Covariate/final CGM connection conflict")
    if "cgm_device_type" in cg:
        pair = cg[["participant_id", "cgm_device_type"]].merge(cov[["participant_id", "cgm_device_type"]], on="participant_id", validate="one_to_one")
        a, b = [norm(s).str.replace(r"\.0$", "", regex=True) for s in [pair.cgm_device_type_x, pair.cgm_device_type_y]]
        require(not (a.notna() & b.notna() & a.str.lower().ne(b.str.lower())).any(), "CGM device conflict")
    cov["covariates_complete"] = cov[BASE].notna().all(axis=1)
    cov["primary_eligible"] = ~flag.eq(1) & cov.covariates_complete
    cov["strict_eligible"] = flag.eq(0) & cov.covariates_complete
    cov["exclusion_unknown"] = flag.isna()
    cov["has_covariate_record"] = True
    mapping = species_mapping(header(paths["microbiome"]), config["taxa"])
    mc = KEYS + [r["column"] for r in mapping]
    if "sample_name" in header(paths["microbiome"]):
        mc.append("sample_name")
    micro = scoped(read_csv(paths["microbiome"], mc, usecols=mc), config, "microbiome", audit)
    for r in mapping:
        micro[r["taxon"]] = number(micro[r["column"]], r["taxon"], strict=True)
    if "sample_name" in micro:
        micro["sample_name"] = norm(micro.sample_name)
    dates = timing_table(micro, paths["microbiome_dates"], config)
    micro = micro[["participant_id", "T068", "T096"]].merge(dates, on="participant_id", validate="one_to_one")
    keep = BASE + ["primary_eligible", "strict_eligible", "exclusion_unknown", "has_covariate_record"]
    people = cg[KEYS + ["connection_id"]].merge(cov[["participant_id"] + keep], on="participant_id", how="left", validate="one_to_one")
    people = people.merge(micro, on="participant_id", how="left", validate="one_to_one")
    for col in ["primary_eligible", "strict_eligible", "exclusion_unknown", "has_covariate_record"]:
        people[col] = people[col].fillna(False).astype(bool)
    people["timing_status"] = people.timing_status.fillna("UNKNOWN_NO_MICROBIOME")
    people["sample_date"] = people.sample_date.fillna("")
    return people, pd.DataFrame(audit), pd.DataFrame(mapping)
