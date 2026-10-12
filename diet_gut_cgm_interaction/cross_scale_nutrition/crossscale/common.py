"""Strict IO and audit helpers; compatible with Python 3.8."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

TAXA = ["T068", "T096"]
OUTCOMES = {"SD": "daily_sd", "Mean": "daily_mean"}
BASE_NUMERIC = ["age_years", "sleep_duration_hours_day", "physical_activity_met_h_week",
                "vitamin_use", "hormone_use", "bmi"]
BASE_CATEGORY = ["sex", "education_level", "smoking_status", "cgm_device_type"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), delete=False) as f:
        temp = f.name
        json.dump(clean_json(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, str(path))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        columns = next(csv.reader(f), [])
    require(columns and all(columns) and len(set(columns)) == len(columns), "Invalid CSV header")
    d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    require(isinstance(d.index, pd.RangeIndex), "Malformed CSV row width")
    return d


def numeric(s, label):
    missing = s.astype(str).str.lower().isin(["", "nan", "none", "null", "<na>"])
    n = pd.to_numeric(s.where(~missing), errors="coerce").astype(float)
    require(not ((~missing) & ~np.isfinite(n)).any(), "Malformed/nonfinite numeric field: " + label)
    return n


def boolean(s, label):
    v = s.astype(str).str.lower()
    require(v.isin(["true", "false"]).all(), "Malformed boolean field: " + label)
    return v.eq("true")


def sample_hash(d):
    require(d.day_id.is_unique and d.day_id.ne("").all(), "Invalid person-day identity")
    return hashlib.sha256("\n".join(sorted(d.day_id.astype(str))).encode()).hexdigest()


def people_hash(d):
    return hashlib.sha256("\n".join(sorted(d.participant_id.unique())).encode()).hexdigest()


def bh(pvalues):
    p = np.asarray(pvalues, float).copy()
    missing = ~np.isfinite(p)
    p[missing] = 1.0
    require(((p >= 0) & (p <= 1)).all(), "Invalid p-values")
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1].clip(0, 1)
    out = np.empty(len(p))
    out[order] = q
    out[missing] = np.nan
    return out


def code_hashes(here):
    paths = []
    for pattern in ["*.py", "*.sh", "requirements.txt", "crossscale/*.py", "config/*.json"]:
        paths.extend(Path(here).glob(pattern))
    return {str(p.relative_to(here)): sha256(p) for p in sorted(paths)}
