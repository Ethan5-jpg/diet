#!/usr/bin/env python3
"""Extend the frozen CGM cohort with 17 candidates; pandas/NumPy only."""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import traceback
import uuid

import numpy as np
import pandas as pd

VERSION = "2026-09-18.1"
PACKAGE = Path(__file__).resolve().parent.parent
KEY = ["participant_id", "cohort", "research_stage", "array_index", "connection_id"]
OLD_SEVEN = ["cgm_mean", "cgm_cv", "cgm_above_140", "cgm_mage",
             "cgm_above_180", "cgm_below_70", "cgm_in_range_70_180"]
MISSING = {"", "na", "n/a", "nan", "null", "none", "<na>", "#n/a"}
DEFAULT_BASE = (PACKAGE.parent / "outputs/cgm_extension/data" /
                "finalize_20260915T074925_139680Z_b375d22f/06_cgm_extended_phenotypes.csv")


class Tee:
    def __init__(self, *handles):
        self.handles = handles

    def write(self, value):
        for handle in self.handles:
            handle.write(value)
            handle.flush()
        return len(value)

    def flush(self):
        for handle in self.handles:
            handle.flush()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(value, path):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_json(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_table(path):
    # Read cells as text so old columns survive unchanged, including leading zeros.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if not header or any(not h.strip() for h in header):
            raise ValueError("Empty CSV header: %s" % path)
        if any(n > 1 for n in Counter(header).values()):
            raise ValueError("Duplicate CSV column names: %s" % path)
        for row in reader:
            if row and len(row) != len(header):
                raise ValueError("Malformed CSV row %d: %s" % (reader.line_num, path))
    frame = pd.read_csv(str(path), dtype=str, keep_default_na=False, na_filter=False,
                        encoding="utf-8-sig", low_memory=False)
    if frame.empty or not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("Empty table or malformed CSV index: %s" % path)
    return frame


def require_columns(frame, columns, name):
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise ValueError("%s missing columns: %s" % (name, ", ".join(absent)))


def keys(frame, name):
    require_columns(frame, KEY, name)
    result = frame[KEY].copy().reset_index(drop=True)
    for column in KEY:
        if result[column].isna().any():
            raise ValueError("%s has null key %s" % (name, column))
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError("%s has empty key %s" % (name, column))
    normalized = []
    for value in result["array_index"]:
        try:
            number = Decimal(value)
            if (not number.is_finite() or number != number.to_integral_value()
                    or number < 0 or number > 9223372036854775807):
                raise ValueError()
            normalized.append(str(int(number)))
        except (InvalidOperation, ValueError, OverflowError):
            raise ValueError("%s array_index must be a finite nonnegative integer" % name)
    result["array_index"] = normalized
    if result.duplicated(KEY).any():
        raise ValueError("%s has duplicate full keys" % name)
    return result


def match_source(old, source, outcomes, expected_rows):
    if len(old) != expected_rows:
        raise ValueError("Base cohort row count %d != expected %d" % (len(old), expected_rows))
    require_columns(old, [m + s for m in OLD_SEVEN for s in ("_raw", "_clean", "_z")], "base")
    require_columns(source, [o["field"] for o in outcomes], "iglu")
    left, right = keys(old, "base"), keys(source, "iglu")
    if left["participant_id"].duplicated().any():
        raise ValueError("Base has duplicate participant_id")
    if not (left["cohort"].eq("10k") & left["research_stage"].eq("00_00_visit")).all():
        raise ValueError("Base is not the frozen 10k baseline cohort")
    left["_order"] = np.arange(len(left))
    right["_source_row"] = np.arange(len(right))
    joined = left.merge(right, on=KEY, how="left", validate="one_to_one", indicator=True)
    joined = joined.sort_values("_order").reset_index(drop=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("%d base connections do not match iglu" % int(joined["_merge"].ne("both").sum()))
    selected = source.iloc[joined["_source_row"].astype(int).values].reset_index(drop=True)
    return selected, {"base_rows": len(old), "iglu_rows": len(source), "matched_rows": len(old),
                      "unmatched_rows": 0, "unused_iglu_rows": len(source) - len(old)}


def numeric(series):
    text = series.astype(str).str.strip()
    missing = series.isna() | text.str.lower().isin(MISSING)
    value = pd.to_numeric(text.mask(missing), errors="coerce").astype(float)
    state = pd.Series("valid", index=series.index, dtype=object)
    state.loc[missing] = "source_missing"
    state.loc[~missing & value.isna()] = "non_numeric"
    state.loc[value.notna() & ~np.isfinite(value)] = "non_finite"
    return value.where(state.eq("valid")), state


def legal_values(series, domain):
    value, state = numeric(series)
    invalid = value.notna() & (value.lt(0) if domain == "nonnegative" else ~value.between(0, 100))
    state.loc[invalid] = "negative_value" if domain == "nonnegative" else "outside_0_100"
    return value.mask(invalid), state


def describe(series):
    value = series.dropna()
    n = len(value)
    result = {"valid_n": n, "missing_n": len(series) - n, "zero_n": int(value.eq(0).sum()),
              "zero_pct": 100 * value.eq(0).sum() / n if n else np.nan,
              "hundred_n": int(value.eq(100).sum()), "mean": value.mean(),
              "sample_sd": value.std(ddof=1), "min": value.min(), "max": value.max()}
    for q in (1, 5, 25, 50, 75, 95, 99):
        result["p%d" % q] = value.quantile(q / 100.0) if n else np.nan
    return result


def robust_clean(raw):
    """Same interval/tie/ddof algorithm as parent 04_clean_cgm_phenotypes.py."""
    values = np.sort(raw.dropna().to_numpy(dtype=float))
    n = len(values)
    actions = pd.Series("unchanged", index=raw.index, dtype=object)
    actions.loc[raw.isna()] = "missing_input"
    params = {"cleaning_status": "applied", "coverage_target": 0.95,
              "extreme_values_set_missing": 0, "values_winsorized": 0}
    if n < 2 or len(np.unique(values)) < 2:
        params["cleaning_status"] = "not_estimable_insufficient_or_constant"
        return raw.copy(), actions, params
    width_n = max(2, int(math.ceil(0.95 * n)))
    widths = values[width_n - 1:] - values[:n - width_n + 1]
    start = int(np.argmin(widths))
    low, high = values[start], values[start + width_n - 1]
    interval = values[(values >= low) & (values <= high)]
    mean, sd = float(np.mean(interval)), float(np.std(interval, ddof=1))
    params.update(window_target_n=width_n, window_actual_n_including_boundary_ties=len(interval),
                  interval_low=low, interval_high=high, center_mean=mean, center_sample_sd=sd)
    if not np.isfinite(sd) or sd <= 0 or not np.isfinite(mean):
        params["cleaning_status"] = "blocked_degenerate_95pct_interval"
        actions.loc[raw.notna()] = "cleaning_blocked"
        return pd.Series(np.nan, index=raw.index), actions, params
    lower8, upper8 = mean - 8 * sd, mean + 8 * sd
    lower5, upper5 = mean - 5 * sd, mean + 5 * sd
    extreme = raw.notna() & ((raw < lower8) | (raw > upper8))
    winsor = raw.notna() & ~extreme & ((raw < lower5) | (raw > upper5))
    clean = raw.mask(extreme).clip(lower=lower5, upper=upper5)
    actions.loc[extreme] = "set_missing_beyond_8sd"
    actions.loc[winsor] = "winsorized_at_5sd"
    params.update(extreme_lower_8sd=lower8, extreme_upper_8sd=upper8,
                  winsor_lower_5sd=lower5, winsor_upper_5sd=upper5,
                  extreme_values_set_missing=int(extreme.sum()), values_winsorized=int(winsor.sum()))
    return clean, actions, params


def standardize(clean):
    n, mean, sd = int(clean.notna().sum()), clean.mean(), clean.std(ddof=1)
    state = ("insufficient_n" if n < 2 else "constant" if clean.nunique() < 2 else
             "computed" if np.isfinite(sd) and sd > 0 and np.isfinite(mean) else "nonfinite_parameters")
    z = (clean - mean) / sd if state == "computed" else pd.Series(np.nan, index=clean.index)
    if state == "computed":
        if (not np.isfinite(z.loc[clean.notna()]).all()
                or not np.isclose(z.mean(), 0, atol=1e-10, rtol=0)
                or not np.isclose(z.std(ddof=1), 1, atol=1e-10, rtol=0)):
            raise ValueError("Z validation failed")
    return z, {"clean_valid_n": n, "clean_mean": mean, "clean_sample_sd": sd,
               "z_status": state, "z_valid_n": int(z.notna().sum()),
               "z_mean": z.mean(), "z_sample_sd": z.std(ddof=1), "sd_ddof": 1}


def equal_numeric(a, b):
    return bool(np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                            rtol=1e-10, atol=1e-10, equal_nan=True))


def process(old, selected, outcomes):
    audit_added, final_added = {}, {}
    distributions, parameters, issues, actions_rows, blockers = [], [], [], [], []
    for spec in outcomes:
        field, rule = spec["field"], spec["rule"]
        suffixes = ["_source", "_value_status"]
        if rule != "reuse_legacy":
            suffixes += ["_raw", "_clean", "_z", "_clean_action", "_z_status"]
        collisions = [field + s for s in suffixes if field + s in old.columns]
        if collisions:
            raise ValueError("Would overwrite existing columns: %s" % collisions)
        raw, value_state = legal_values(selected[field], spec["domain"])
        for frame in (audit_added, final_added):
            frame[field + "_source"] = selected[field]
            frame[field + "_value_status"] = value_state
        if rule != "reuse_legacy":
            audit_added[field + "_raw"] = raw
            final_added[field + "_raw"] = raw
        distribution = {"outcome": field, "domain": spec["domain"], "rule": rule}
        distribution.update(describe(raw))
        for reason in ("source_missing", "non_numeric", "non_finite", "negative_value", "outside_0_100"):
            distribution[reason + "_n"] = int(value_state.eq(reason).sum())
        distributions.append(distribution)
        for i in value_state.index[value_state.ne("valid")]:
            record = old.loc[i, KEY].to_dict()
            record.update(outcome=field, source_value=selected.at[i, field], reason=value_state.at[i])
            issues.append(record)
        params = {"outcome": field, "rule": rule, "label": spec["label"], "note": spec.get("note", ""),
                  "z_population": "valid_clean_values_in_full_frozen_cgm_cohort"}
        if rule == "reuse_legacy":
            require_columns(old, [field + s for s in ("_raw", "_clean", "_z")], "base reuse")
            old_raw, _ = legal_values(old[field + "_raw"], spec["domain"])
            clean, clean_state = numeric(old[field + "_clean"])
            old_z, z_state = numeric(old[field + "_z"])
            if not equal_numeric(raw, old_raw):
                raise ValueError("%s legacy raw differs from matched iglu; input versions differ" % field)
            if clean_state.isin(["non_numeric", "non_finite"]).any() or z_state.isin(["non_numeric", "non_finite"]).any():
                raise ValueError("%s invalid legacy clean/z" % field)
            legal_clean, _ = legal_values(old[field + "_clean"], spec["domain"])
            if not equal_numeric(legal_clean, clean) or (clean.notna() & raw.isna()).any():
                raise ValueError("%s legacy clean violates domain or missing raw" % field)
            expected_z, zparams = standardize(clean)
            if not equal_numeric(old_z, expected_z):
                raise ValueError("%s legacy Z inconsistent with full-cohort clean; refusing to recalculate" % field)
            params.update(cleaning_status="reused_unchanged", extreme_values_set_missing=None, values_winsorized=None)
            params.update(zparams)
        else:
            if rule == "robust95_8_5":
                clean, clean_actions, cparams = robust_clean(raw)
            else:
                clean = raw.copy()
                clean_actions = pd.Series(np.where(raw.notna(), "unchanged", "missing_input"), index=raw.index)
                cparams = {"cleaning_status": "legal_raw_unchanged", "extreme_values_set_missing": 0, "values_winsorized": 0}
            z, zparams = standardize(clean)
            if cparams["cleaning_status"].startswith("blocked"):
                blockers.append(field + ": " + cparams["cleaning_status"])
                zparams["z_status"] = "blocked_cleaning"
            if zparams["z_status"] == "nonfinite_parameters":
                blockers.append(field + ": nonfinite_standardization_parameters")
            final_added[field + "_clean"] = clean
            final_added[field + "_z"] = z
            final_added[field + "_clean_action"] = clean_actions
            final_added[field + "_z_status"] = np.where(clean.isna(), "missing_clean_input", zparams["z_status"])
            params.update(cparams)
            params.update(zparams)
            for i in clean_actions.index[clean_actions.isin(["set_missing_beyond_8sd", "winsorized_at_5sd"])]:
                record = old.loc[i, KEY].to_dict()
                record.update(outcome=field, raw_value=raw.at[i], clean_value=clean.at[i], action=clean_actions.at[i])
                actions_rows.append(record)
        parameters.append(params)
    audit = pd.concat([old.copy(deep=True), pd.DataFrame(audit_added, index=old.index)], axis=1)
    final = pd.concat([old.copy(deep=True), pd.DataFrame(final_added, index=old.index)], axis=1)
    pd.testing.assert_frame_equal(final[list(old.columns)], old, check_exact=True)
    reports = {
        "raw_distributions": pd.DataFrame(distributions),
        "processing_parameters": pd.DataFrame(parameters),
        "value_issues": pd.DataFrame(issues, columns=KEY + ["outcome", "source_value", "reason"]),
        "cleaning_actions": pd.DataFrame(actions_rows, columns=KEY + ["outcome", "raw_value", "clean_value", "action"]),
    }
    return audit, final, reports, blockers


def load_config(path):
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    outcomes = config["outcomes"]
    names = [o["field"] for o in outcomes]
    if len(names) != 17 or len(set(names)) != 17 or not {"cgm_sdb", "cgm_sddm"}.issubset(names):
        raise ValueError("Config must have 17 distinct outcomes including SDb and SDdm")
    reused = {o["field"] for o in outcomes if o["rule"] == "reuse_legacy"}
    if reused != {"cgm_gmi", "cgm_modd"}:
        raise ValueError("Config must reuse legacy GMI and MODD")
    for o in outcomes:
        if o["domain"] not in ("nonnegative", "bounded_0_100") or o["rule"] not in ("reuse_legacy", "robust95_8_5", "legal_unchanged"):
            raise ValueError("Unsupported outcome domain/rule")
    return config


def summary_text(run, report=None):
    lines = ["=== CGM 新增17指标处理摘要 ===", "STATUS=" + run["status"],
             "RUN_DIR=" + run["run_dir"], "INPUT_BASE=" + run["base_csv"],
             "INPUT_IGLU=" + run["iglu_csv"]]
    if "matching" in run:
        m = run["matching"]
        lines += ["人数=%d；匹配=%d；未匹配=%d" % (m["base_rows"], m["matched_rows"], m["unmatched_rows"])]
    lines += ["候选字段=17；复用GMI/MODD=2；新增raw/clean/z组=15；SDb与SDdm分别保留。"]
    if report is not None:
        cols = ["outcome", "clean_valid_n", "extreme_values_set_missing", "values_winsorized", "z_valid_n", "z_status"]
        lines += ["\n各项处理结果（复用项的异常处理计数为NA，不代表0）：", report[cols].to_string(index=False)]
    if "preservation_verified" in run:
        lines += ["旧列文本及行序保留=%s；输入文件未变化=%s" % (run["preservation_verified"], run["inputs_unchanged"])]
    if run.get("final_file"):
        lines += ["FINAL_FILE=" + run["final_file"]["path"]]
    else:
        lines += ["本次未生成正式最终表。"]
    for item in run.get("blockers", []):
        lines += ["BLOCKER=" + item]
    if run.get("error"):
        lines += ["ERROR=" + run["error"]]
    lines += ["未做饮食/菌群关联或新的糖尿病/药物排除；未按17项同时完整筛人。",
              "Z可计算不代表普通线性模型适用；原始曲线未重新计算。",
              "详细分布/参数/异常记录见reports；后续关联分析必须显式使用本次FINAL_FILE。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--iglu-csv", type=Path, default=PACKAGE.parent.parent / "Transfer/cgm/iglu.csv")
    parser.add_argument("--config", type=Path, default=PACKAGE / "config/outcomes.json")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs")
    parser.add_argument("--expected-rows", type=int, default=7493)
    parser.add_argument("--check-only", action="store_true", help="Audit and preview parameters; no final phenotype CSV")
    args = parser.parse_args(argv)
    if args.expected_rows < 1:
        parser.error("--expected-rows must be positive")
    base_path, source_path, config_path = [p.expanduser().resolve() for p in (args.base_csv, args.iglu_csv, args.config)]
    run_dir = args.output_dir.expanduser().resolve() / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("data", "reports", "logs"):
        (run_dir / folder).mkdir()
    run = {"status": "running", "version": VERSION, "run_dir": str(run_dir), "base_csv": str(base_path),
           "iglu_csv": str(source_path), "check_only": args.check_only, "expected_rows": args.expected_rows,
           "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    code, report = 1, None
    pending = run_dir / "data/07_cgm_paper_extended_phenotypes.pending.csv"
    final_path = run_dir / "data/07_cgm_paper_extended_phenotypes.csv"
    with (run_dir / "logs/run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print("Reading frozen cohort and 17 source fields...", flush=True)
                input_paths = [base_path, source_path, config_path, Path(__file__).resolve()]
                before = [fingerprint(p) for p in input_paths]
                run["input_fingerprints"] = before
                config = load_config(config_path)
                write_json(config, run_dir / "reports/outcomes_used.json")
                old, source = read_table(base_path), read_table(source_path)
                selected, matching = match_source(old, source, config["outcomes"], args.expected_rows)
                run["matching"] = matching
                audit, final, reports, blockers = process(old, selected, config["outcomes"])
                run["blockers"] = blockers
                audit.to_csv(str(run_dir / "data/01_extracted_audit.csv"), index=False, encoding="utf-8")
                for name, frame in reports.items():
                    frame.to_csv(str(run_dir / "reports" / (name + ".csv")), index=False, encoding="utf-8")
                report = reports["processing_parameters"]
                run["unavailable_z_outcomes"] = report.loc[report["z_status"].ne("computed"), "outcome"].tolist()
                run["inputs_unchanged"] = before == [fingerprint(p) for p in input_paths]
                if not run["inputs_unchanged"]:
                    raise ValueError("Input/config/script changed while processing")
                pd.testing.assert_frame_equal(read_table(run_dir / "data/01_extracted_audit.csv")[list(old.columns)], old, check_exact=True)
                run["preservation_verified"] = True
                if blockers:
                    run["status"] = "blocked_review_required"
                    code = 2
                elif args.check_only:
                    run["status"] = "checked_no_final_table"
                    code = 0
                else:
                    final.to_csv(str(pending), index=False, encoding="utf-8")
                    written = read_table(pending)
                    pd.testing.assert_frame_equal(written[list(old.columns)], old, check_exact=True)
                    for spec in config["outcomes"]:
                        for suffix in ("_raw", "_clean", "_z"):
                            col = spec["field"] + suffix
                            expected, _ = numeric(final[col])
                            actual, _ = numeric(written[col])
                            if not equal_numeric(expected, actual):
                                raise ValueError("Round-trip validation failed: " + col)
                    if before != [fingerprint(p) for p in input_paths]:
                        run["inputs_unchanged"] = False
                        raise ValueError("Input/config/script changed before finalization")
                    pending.rename(final_path)
                    run["final_file"] = fingerprint(final_path)
                    run["status"] = "completed_with_unavailable_outcomes" if run["unavailable_z_outcomes"] else "completed"
                    code = 0
            except Exception as error:
                run["status"] = "failed"
                run["error"] = "%s: %s" % (type(error).__name__, error)
                traceback.print_exc(file=log)
                if pending.exists():
                    pending.unlink()
            finally:
                run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                run["exit_code"] = code
                write_json(run, run_dir / "reports/manifest.json")
                text = summary_text(run, report)
                (run_dir / "reports/运行摘要.txt").write_text(text, encoding="utf-8")
                (run_dir / "reports/处理结果.md").write_text("# CGM 论文新增17指标处理结果\n\n```text\n" + text + "```\n", encoding="utf-8")
                print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
