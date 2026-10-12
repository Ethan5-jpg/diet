"""Small, strict, Python 3.8-compatible interfaces shared by this module only."""
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["participant_id", "cohort", "research_stage"]
BASE = ["age_years", "sex", "education_level", "smoking_status",
        "sleep_duration_hours_day", "physical_activity_met_h_week", "vitamin_use",
        "hormone_use", "bmi", "cgm_device_type"]
CATEGORICAL = {"sex", "education_level", "smoking_status", "cgm_device_type"}
NUTRIENTS = ["calories_kcal", "carbohydrate_g", "lipid_g", "protein_g",
             "dietary_fiber_g", "alcohol_g"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), delete=False) as f:
        temp = f.name
        json.dump(clean_json(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, str(path))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        columns = next(csv.reader(f), [])
    require(columns and all(columns) and len(columns) == len(set(columns)), "Empty/duplicate CSV header: " + str(path))
    return columns


def read_csv(path, required=(), **kwargs):
    missing = sorted(set(required) - set(header(path)))
    require(not missing, "Missing columns {}: {}".format(missing, path))
    # No automatic string NA coercion of identifiers such as 'NA'.
    result = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, **kwargs)
    if isinstance(result, pd.DataFrame):
        require(isinstance(result.index, pd.RangeIndex), "Malformed CSV row width: " + str(path))
    return result


def norm(s):
    x = s.fillna("").astype(str).str.strip()
    return x.mask(x.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def ids(s):
    return norm(s).str.replace(r"\.0$", "", regex=True)


def number(s, label="", strict=False):
    original = norm(s)
    n = pd.to_numeric(original, errors="coerce").astype(float)
    if strict:
        require(not (original.notna() & ~np.isfinite(n)).any(), "Non-numeric/nonfinite values: " + label)
    return n.where(np.isfinite(n))


def unique(d, keys, label):
    require(not d[keys].isna().any().any(), label + ": missing identity")
    require(not d.duplicated(keys).any(), label + ": duplicate identity; no automatic row selection")


def scoped(d, config, label, audit, unique_person=True, allow_empty=False):
    d = d.copy()
    stage = "cgm_research_stage" if label == "covariates" and "cgm_research_stage" in d else "research_stage"
    require(set(["participant_id", "cohort", stage]).issubset(d), label + ": cohort/visit identity required")
    d["participant_id"] = ids(d.participant_id)
    d["cohort"], d[stage] = norm(d.cohort), norm(d[stage])
    if stage != "research_stage":
        if "research_stage" in d:
            other = norm(d.research_stage)
            require(not (other.notna() & d[stage].notna() & other.ne(d[stage])).any(), label + ": conflicting visit fields")
        d["research_stage"] = d[stage]
    keep = d.cohort.eq(config["cohort"]) & d.research_stage.eq(config["research_stage"])
    audit.append(dict(source=label, input_rows=len(d), scoped_rows=int(keep.sum())))
    d = d.loc[keep].copy()
    require(allow_empty or not d.empty, label + ": no rows in requested cohort/visit")
    unique(d, ["participant_id"], label) if unique_person else require(d.participant_id.notna().all(), label + ": missing ID")
    return d


def parse_times(s, naive_timezone=None):
    text = norm(s)
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False)
    kw = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
    out = pd.to_datetime(text.where(aware), utc=True, errors="coerce", **kw)
    if naive_timezone:
        local = pd.to_datetime(text.where(~aware), errors="coerce", **kw)
        local = local.dt.tz_localize(naive_timezone, ambiguous="NaT", nonexistent="NaT")
        out = out.fillna(local.dt.tz_convert("UTC"))
    return out


def epoch(s):
    # Timestamp.timestamp avoids pandas datetime unit changes across versions.
    # Series.map preserves datetime dtype for an empty datetime Series, whose
    # astype(float) raises. Build float values explicitly, including empty/NaT.
    values = np.fromiter((x.timestamp() if pd.notna(x) else np.nan for x in s),
                         dtype=np.float64, count=len(s))
    return pd.Series(values, index=s.index, name=s.name, dtype="float64")


def date_only(s, timezone):
    def one(v):
        if pd.isna(v) or not str(v).strip():
            return None
        text = str(v).strip()
        # Reject partial years/months and ambiguous slash dates instead of
        # manufacturing a specific sampling/logging date from them.
        if not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[Tt ])", text):
            return None
        try:
            x = pd.Timestamp(text)
            if x.tzinfo is not None:
                x = x.tz_convert(timezone)
            return x.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None
    return s.map(one)


def validate_config(c):
    require(c.get("version") == "2026-10-11.3", "Unsupported analysis configuration version")
    require([t["id"] for t in c["taxa"]] == ["T068", "T096"], "This version freezes two taxa")
    require([t["species"] for t in c["taxa"]] == ["Eubacterium_rectale", "Ruthenibacterium_lactatiformans"], "Taxon registry changed")
    require(c["cohort"] and c["research_stage"], "Empty cohort/visit")
    require(type(c["chunksize"]) is int and c["chunksize"] > 0, "Invalid chunk size")
    pd.Timestamp("2026-01-01", tz=c["time"]["local_timezone"])
    if c["time"]["naive_timezone"]:
        pd.Timestamp("2026-01-01", tz=c["time"]["naive_timezone"])
    for section in ["cgm", "diet", "analysis"]:
        for key, value in c[section].items():
            require(type(value) in (int, float) and np.isfinite(value) and value > 0, "Invalid " + section + "." + key)
    cg, diet, a = c["cgm"], c["diet"], c["analysis"]
    require(cg["bin_minutes"] == 15, "This version freezes 15-minute observed bins")
    require(0 < cg["minimum_coverage"] <= cg["strict_coverage"] <= 1, "Invalid CGM coverage")
    require(cg["minimum_quarter_coverage"] <= 1 and cg["strict_gap_minutes"] <= cg["maximum_gap_minutes"], "Invalid CGM gap/quarter rules")
    require(cg["minimum_glucose"] < cg["maximum_glucose"], "Invalid glucose bounds")
    require(diet["minimum_energy_kcal"] < diet["maximum_energy_kcal"], "Invalid energy bounds")
    require(diet["minimum_nova_coverage"] <= diet["strict_nova_coverage"] <= 1, "Invalid NOVA coverage")
    require(a["minimum_people"] >= 3 and a["minimum_days"] >= 2 and a["strict_days"] >= a["minimum_days"], "Invalid repeated-day requirements")
    require(0 < a["alpha"] < 1, "Invalid alpha")
    for key in ["minimum_people", "minimum_days", "strict_days", "minimum_residual_df", "minimum_tail_people"]:
        require(type(a[key]) is int, "Integer required: " + key)
    expected = {"diet", "cgm_glob", "nova_mapping", "covariates", "microbiome", "connections", "microbiome_dates"}
    require(set(c["inputs"]) == expected, "Invalid input registry")
    require(all(isinstance(v, str) and v for k, v in c["inputs"].items() if k != "microbiome_dates"), "Missing input path")
    return c
