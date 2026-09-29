"""Strict IO and provenance; compatible with Python 3.8 syntax."""
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import glob
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["participant_id", "cohort", "research_stage"]
NUTRIENTS = ["weight_g", "calories_kcal", "carbohydrate_g", "lipid_g",
             "protein_g", "dietary_fiber_g"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.generic):
        return safe(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    path = Path(path)
    body = json.dumps(safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), delete=False) as f:
            name = f.name
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, str(path))
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        names = next(csv.reader(f), [])
    require(names and all(names) and len(names) == len(set(names)),
            "Invalid/duplicate CSV columns: " + str(path))
    return names


def read_csv(path, required=(), **kwargs):
    names = header(path)
    need = sorted(set(required) - set(names))
    require(not need, "Missing columns {} in {}".format(need, path))
    result = pd.read_csv(path, dtype=str, encoding="utf-8-sig", **kwargs)
    if isinstance(result, pd.DataFrame):
        require(isinstance(result.index, pd.RangeIndex), "Malformed CSV row width: " + str(path))
    return result


def norm(s):
    x = s.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return x.mask(x.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def number(s):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return x.where(np.isfinite(x))


def unique(df, keys, label):
    require(not df[keys].isna().any().any(), label + ": missing identity")
    require(not df.duplicated(keys).any(), label + ": duplicate identity; no automatic averaging")


def scope_columns(columns, label):
    """Require verifiable cohort/visit metadata before joining participant tables."""
    stage = "cgm_research_stage" if label == "covariates" and "cgm_research_stage" in columns else "research_stage"
    require("cohort" in columns and stage in columns,
            label + ": cohort and visit metadata required; do not infer visit from participant ID")
    return stage


def select_participant_scope(run, df, label):
    stage = scope_columns(df.columns, label)
    d = df.copy()
    for key in ["participant_id", "cohort", stage]:
        d[key] = norm(d[key])
    if stage != "research_stage" and "research_stage" in d:
        other = norm(d.research_stage)
        require(not (other.notna() & d[stage].notna() & other.ne(d[stage])).any(),
                label + ": conflicting visit metadata")
    missing = d[["participant_id", "cohort", stage]].isna().any(axis=1)
    keep = ~missing & d.cohort.eq(run.c["cohort"]) & d[stage].eq(run.c["research_stage"])
    run.table(pd.DataFrame([dict(source=label, input_rows=len(d), selected_rows=int(keep.sum()),
                                missing_identity_or_scope=int(missing.sum()),
                                other_cohort_or_visit=int((~missing & ~keep).sum()),
                                cohort=run.c["cohort"], research_stage=run.c["research_stage"],
                                visit_column=stage)]), "03_{}_scope.csv".format(label))
    d = d.loc[keep].copy()
    require(not d.empty, label + ": no rows for configured cohort/visit")
    unique(d, ["participant_id"], label + " selected cohort/visit")
    return d


def acquire_run_lock(out):
    """Advisory lock for the supported Linux/macOS servers; never unlink it."""
    import fcntl
    handle = (Path(out) / ".run.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        handle.close()
        raise ValueError("Run directory is locked by another process; wait for it to finish")
    return handle


def parse_times(series, naive_timezone=None):
    """Never silently interpret timezone-naive consumption times as UTC."""
    s = series.fillna("").astype(str).str.strip()
    aware = s.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, case=False)
    options = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
    out = pd.to_datetime(s.where(aware), utc=True, errors="coerce", **options)
    if naive_timezone:
        local = pd.to_datetime(s.where(~aware), errors="coerce", **options)
        local = local.dt.tz_localize(naive_timezone, ambiguous="NaT", nonexistent="NaT")
        out = out.fillna(local.dt.tz_convert("UTC"))
    return out


def epoch(times):
    out = times.astype("int64").astype(float) / 1e9
    return out.where(times.notna())


def validate_config(c):
    require(c["schema_version"] == 1, "Unsupported config schema")
    require(c["data_kind"] in ["REAL", "SYNTHETIC"], "Invalid data_kind")
    require(type(c["time"]["consumption_time_verified"]) is bool, "Time verification must be JSON true/false")
    for section in ["ingest", "meal", "cgm", "plots"]:
        for name, value in c[section].items():
            require(type(value) in [int, float] and math.isfinite(value), "Expected finite number: " + section + "." + name)
    for name in ["minimum_people", "minimum_meals_per_person", "minimum_nova_range_pp", "minimum_residual_df"]:
        value = c["analysis"][name]
        require(type(value) in [int, float] and math.isfinite(value), "Expected finite number: analysis." + name)
    for section, names in {"ingest": ["chunksize"], "plots": ["example_meals"],
                           "meal": ["signature_weight_decimals"],
                           "cgm": ["baseline_min_points", "minimum_post_points"],
                           "analysis": ["minimum_people", "minimum_meals_per_person", "minimum_residual_df"]}.items():
        for name in names:
            require(type(c[section][name]) is int, "Expected integer: " + section + "." + name)
    require(c["plots"]["example_meals"] >= 0, "Negative example count")
    require(0 <= c["meal"]["signature_weight_decimals"] <= 6, "Invalid signature rounding")
    require(c["cgm"]["baseline_min_points"] >= 2, "Baseline trend requires at least two points")
    require(c["analysis"]["minimum_meals_per_person"] >= 2, "Within-person models require repeated meals")
    pd.Timestamp("2026-01-01", tz=c["time"]["local_timezone"])
    if c["time"]["naive_timezone"]:
        pd.Timestamp("2026-01-01", tz=c["time"]["naive_timezone"])
    require(c["cgm"]["post_minutes"] == 120, "Version 1 supports 120-minute outcomes only")
    require(0 < c["meal"]["nova_min_coverage"] <= 1, "Coverage must be in (0,1]")
    require(c["cgm"]["plausible_min"] < c["cgm"]["plausible_max"], "Invalid glucose range")
    for section, names in {
        "ingest": ["chunksize"],
        "meal": ["max_span_minutes", "minimum_kcal", "minimum_carbohydrate_g",
                 "previous_event_gap_minutes", "next_event_gap_minutes"],
        "cgm": ["sampling_minutes", "baseline_minutes", "baseline_min_points",
                "baseline_nearest_minutes", "max_gap_minutes", "minimum_post_points"],
        "analysis": ["minimum_people", "minimum_meals_per_person", "minimum_nova_range_pp", "minimum_residual_df"]
    }.items():
        for name in names:
            require(c[section][name] > 0, "Must be positive: " + section + "." + name)
    require(c["cgm"]["warmup_minutes"] >= 0, "Negative warmup")
    require(c["meal"]["max_span_minutes"] <= 60, "Meal span >60 minutes not supported in version 1")
    require(c["meal"]["next_event_gap_minutes"] >= 120, "Next event gap must cover full outcome window")
    require(c["analysis"]["minimum_people"] >= 3, "At least three participant clusters required")
    labels = [t["label"] for t in c["analysis"]["taxa"]]
    require(len(labels) == len(set(labels)), "Duplicate taxon label")
    for t in c["analysis"]["taxa"]:
        require(isinstance(t["label"], str) and t["label"].strip(), "Empty taxon label")
        require(isinstance(t.get("column") or t.get("match"), str) and (t.get("column") or t.get("match")).strip(), "Empty taxon column/match")
    for name in ["diet", "cgm_glob"]:
        require(isinstance(c["inputs"][name], str) and c["inputs"][name].strip(), "Missing required input path: " + name)
    return c


def resolve_taxa(columns, taxa):
    mappings = []
    for taxon in taxa:
        if taxon["column"]:
            found = [taxon["column"]] if taxon["column"] in columns else []
        else:
            needle = taxon["match"].lower().replace(" ", "_")
            found = [name for name in columns if needle in name.lower().replace(" ", "_")]
        mappings.append(dict(label=taxon["label"], status="FOUND" if len(found) == 1 else "MISSING_OR_AMBIGUOUS_COLUMN",
                             matches=len(found), column=found[0] if len(found) == 1 else ""))
    return mappings


class Run:
    def __init__(self, root, out, config):
        self.root, self.out, self.c = Path(root).resolve(), Path(out).resolve(), config
        self.inputs = []
        self.status = {}
        for folder in ["tables", "private", "figures", "logs"]:
            (self.out / folder).mkdir(parents=True, exist_ok=True)

    def path(self, name):
        value = self.c["inputs"].get(name)
        if not value:
            return None
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.root / p

    def record(self, path, label):
        p = Path(path).resolve()
        require(p.is_file(), "Missing {}: {}".format(label, p))
        stat = p.stat()
        item = dict(label=label, path=str(p), bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, sha256=sha256(p))
        require(p.stat().st_mtime_ns == stat.st_mtime_ns and p.stat().st_size == stat.st_size,
                "Input changed while hashing: " + str(p))
        for previous in self.inputs:
            if previous["path"] == str(p):
                require(previous["sha256"] == item["sha256"], "Input changed within run: " + str(p))
                return p
        self.inputs.append(item)
        return p

    def table(self, df, name, private=False):
        folder = "private" if private else "tables"
        df.to_csv(self.out / folder / name, index=False, encoding="utf-8-sig")

    def save(self):
        write_json(self.out / "manifest.json", {
            "data_kind": self.c["data_kind"], "status": self.status,
            "root": str(self.root), "config": self.c, "inputs": self.inputs,
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
            "interpretation": "Exploratory observational associations; not causal mediation proof"
        })


def check_inputs_unchanged(items):
    for item in items:
        p = Path(item["path"])
        require(p.is_file() and sha256(p) == item["sha256"],
                "Input changed or missing since audit: " + str(p))


def check_source_inventory(run):
    if run.status.get("audit", {}).get("status") != "PASS":
        return
    recorded = {x["path"] for x in run.inputs if x["label"] == "cgm"}
    current = {str(Path(p).resolve()) for p in glob.glob(str(run.path("cgm_glob")))}
    require(current == recorded, "CGM file inventory changed; start a new run")
    if run.status["audit"]["mapping_status"] == "MISSING_NOVA_MAPPING":
        p = run.path("nova_mapping")
        require(p is None or not p.exists(), "NOVA mapping appeared after audit; rebuild meals in a new run")


def check_source_stats(run):
    for item in run.inputs:
        p = Path(item["path"])
        require(p.is_file(), "Input disappeared during run: " + str(p))
        stat = p.stat()
        require(stat.st_size == item["bytes"] and stat.st_mtime_ns == item["mtime_ns"],
                "Input changed during run: " + str(p))
    check_source_inventory(run)
