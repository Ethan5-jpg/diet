"""Frozen source validation; no inferred participant or visit identities."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hppmeal.common import KEYS, header, norm, read_csv, require, sha256, unique, parse_times, epoch
from gut_meal_bridge_v2 import inputs as gut
from seven_diet_meals.inputs import strict_number
from quality_timecourse.inputs import zone, load_diary
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent


def bundle():
    path = MODULE / "MANIFEST.json"
    require(path.is_file(), "Upload the complete window_qc_followup folder; MANIFEST.json missing")
    body = json.loads(path.read_text(encoding="utf-8"))
    needed = {"__init__.py", "inputs.py", "coverage.py", "analysis.py", "reporting.py",
              "run_analysis.py", "run_window_qc.sh", "README.md", "PLAN.md", "REVIEW.md"}
    require(body.get("version") == VERSION and needed.issubset(body.get("files", {})), "Incomplete upload manifest")
    expected = {str(path): sha256(path)}
    for section, root in [("files", MODULE), ("dependencies", PACKAGE)]:
        for name, digest in body[section].items():
            p = (root / name).resolve()
            require(root in p.parents and p.is_file() and sha256(p) == digest,
                    "Changed/missing upload or dependency: " + name)
            expected[str(p)] = digest
    require(body.get("dependencies"), "Missing dependency fingerprints")
    return expected


def completed(path, stage, config):
    path = Path(path).resolve()
    file = path / "manifest.json"
    require(file.is_file(), "Source manifest missing: " + str(file))
    body = json.loads(file.read_text(encoding="utf-8"))
    require(body["status"].get(stage, {}).get("status") in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"],
            "Source must be completed: " + stage)
    require(not any(s.get("status") in ["FAILED", "RUNNING", "INTERRUPTED"] for s in body["status"].values()),
            "Incomplete source stage: " + stage)
    require(body["config"] == config and body["data_kind"] == config["data_kind"], "Source configuration/data-kind differs")
    return body


def inspect(args):
    root, config, paths, expected, audit, mapping = gut.inspect(
        Path(args.meal_run).resolve(), Path(args.nutrition_run).resolve(), args.data_root,
        args.sources_config, args.score_sources)
    require(config["cgm"]["sampling_minutes"] == 15, "Requires original 15-minute CGM")
    meal_run = Path(args.meal_run).resolve()
    cache = meal_run / "private/cache.sqlite"
    old = json.loads((meal_run / "manifest.json").read_text(encoding="utf-8"))
    digest = old["status"]["audit"].get("cache_sha256")
    require(digest and cache.is_file() and sha256(cache) == digest, "Frozen original CGM/food cache missing or changed")
    require(not any(Path(str(cache) + s).exists() and Path(str(cache) + s).stat().st_size > 0
                    for s in ["-wal", "-journal"]), "Original cache has a live write journal")
    paths["cache"] = cache
    expected[str(cache)] = digest
    paths["diary"] = Path(args.diary).expanduser().resolve() if args.diary else root / "Transfer/diet_logging/diet_logging.csv"
    require(paths["diary"].is_file(), "Diet daily metadata missing: " + str(paths["diary"]))
    expected[str(paths["diary"])] = sha256(paths["diary"])

    for arg, stage, field, relative in [
        (args.seven_run, "seven_diets", "previous_diet", "tables/all_models.csv"),
        (args.gut_run, "gut_meal_bridge_v2", "previous_gut", "tables/models.csv")]:
        source = Path(arg).expanduser().resolve()
        body = completed(source, stage, config)
        for label in ["meals", "covariates", "current4", "new3", "corrected_hpdi"]:
            original = Path(paths[label])
            candidates = []
            for item in body["inputs"]:
                p = Path(item["path"])
                try:
                    p = root / p.relative_to(Path(body["root"]))
                except ValueError:
                    pass
                if p.resolve() == original.resolve():
                    candidates.append(item["sha256"])
            require(len(set(candidates)) == 1 and sha256(original) == candidates[0],
                    "Source lineage differs: " + stage + "/" + label)
        file = source / relative
        require(file.is_file(), "Source model table missing: " + str(file))
        digest_now = sha256(file)
        frozen = body["status"][stage].get("result_sha256", {}).get(relative)
        require(frozen is None or frozen == digest_now, "Source model table changed: " + relative)
        paths[field] = file
        expected[str(file)] = digest_now
        expected[str(source / "manifest.json")] = sha256(source / "manifest.json")

    paths["window_starts"] = Path(args.window_starts).expanduser().resolve() if args.window_starts else None
    paths["cgm_metadata"] = Path(args.cgm_metadata).expanduser().resolve() if args.cgm_metadata else root / "Transfer/cgm/cgm.csv"
    for field in ["window_starts", "cgm_metadata"]:
        file = paths[field]
        if file is not None and file.is_file():
            expected[str(file)] = sha256(file)
        elif field == "window_starts" or args.cgm_metadata:
            require(file is None, "Explicit optional source missing: " + str(file))
        else:
            paths[field] = None
    return root, config, paths, expected, audit, mapping


def windows(path, config):
    columns = KEYS + ["connection_id", "planned_start"]
    if path is None:
        return None
    d = read_csv(path, columns)
    for col in KEYS + ["connection_id"]:
        d[col] = norm(d[col])
    d = d.loc[d.cohort.eq(config["cohort"]) & d.research_stage.eq(config["research_stage"])].copy()
    unique(d, KEYS + ["connection_id"], "declared plan windows")
    d["window_start"] = epoch(parse_times(d.planned_start))
    require(d.window_start.notna().all(), "planned_start must be a valid explicitly timezone-aware timestamp")
    if "diet_start_date" in d:
        clean = norm(d.diet_start_date)
        valid = clean.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
        parsed = pd.to_datetime(clean, format="%Y-%m-%d", errors="coerce")
        require((clean.isna() | (valid & parsed.notna())).all(), "Invalid declared diet_start_date")
        d["diet_start_date"] = clean
    else:
        d["diet_start_date"] = None
    return d


def metadata_scope(path, config):
    required = KEYS + ["connection_id"]
    if path is None or not set(required).issubset(header(path)):
        return pd.DataFrame(columns=required), "UNAVAILABLE_OR_MISSING_IDENTITY_FIELDS"
    d = read_csv(path, required, usecols=required)
    for col in required:
        d[col] = norm(d[col])
    d = d.loc[d.cohort.eq(config["cohort"]) & d.research_stage.eq(config["research_stage"])].copy()
    unique(d, required, "CGM metadata scope")
    return d, "SCOPED_METADATA_PLUS_FROZEN_CACHE"


def numeric_models(path):
    d = read_csv(path)
    for col in ["people", "meals", "estimate", "se", "ci_low", "ci_high", "p", "q_family", "q_meal12", "q_family14"]:
        if col in d:
            d[col] = strict_number(d[col], "previous " + col)
    return d


def load(run, paths, mapping):
    meals, people = gut.load(run, paths, mapping)
    diary = load_diary(run, paths["diary"])
    plan = windows(paths["window_starts"], run.c)
    metadata, scope = metadata_scope(paths["cgm_metadata"], run.c)
    run.table(pd.DataFrame([dict(cgm_population_scope=scope,
        raw_qc="Frozen meal cache valid flag; not upstream percentage_of_cgm_datapoints_lost_in_qc",
        window_anchor="DECLARED_PLAN" if plan is not None else "FIRST_RECORDED_POINT_PROXY",
        missing_before_first_record_identifiable=plan is not None)]), "00_scope_and_anchor.csv")
    return meals, people, diary, plan, metadata
