#!/usr/bin/env python3
"""Audit three new CGM proportions against the fixed legacy cohort.

Reports only: no outlier cleaning, Z-scores, cohort reselection or associations.
CSV text in every legacy column is retained in audit_rows.csv.
Compatible with Python 3.6+; requires pandas and NumPy, not scipy.
"""
import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import pandas as pd

VERSION = "2026-09-15.1"
FULL_KEY = ["participant_id", "cohort", "research_stage", "array_index", "connection_id"]
CONNECTION_KEY = [c for c in FULL_KEY if c != "array_index"]
PROPORTIONS = ["cgm_above_180", "cgm_below_70", "cgm_in_range_70_180"]
REFERENCES = ["cgm_mean", "cgm_cv", "cgm_above_140", "cgm_mage"]
MAGE_COLUMNS = ["cgm_mage_" + v for v in ("raw", "clean", "z")]
QC_COLUMNS = ["cgm_days", "cgm_qc_loss_fraction", "cgm_datapoints"]
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "<na>", "#n/a"}


def default_paths(script_dir=None):
    script_dir = Path(script_dir) if script_dir is not None else Path(__file__).resolve().parent
    workflow = script_dir.parent
    return {
        "core_csv": workflow / "outputs/data/05_cgm_core_phenotypes.csv",
        "iglu_csv": workflow.parent.parent / "Data/Transfer/cgm/iglu.csv",
        "output_dir": workflow / "outputs/cgm_extension",
    }


def read_table(path):
    """Read literal CSV cells, preserving leading zeros and literal NA identifiers."""
    with open(str(path), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if not header or any(not c.strip() for c in header):
            raise ValueError("CSV has an empty header: %s" % path)
        duplicates = [c for c, n in Counter(header).items() if n > 1]
        if duplicates:
            raise ValueError("CSV has duplicate column headers: %s" % duplicates)
        for row in reader:
            if not row:  # Consistent with pandas' skip_blank_lines default.
                continue
            if len(row) != len(header):
                raise ValueError("CSV malformed row width at line %d: expected %d fields, got %d (%s)" %
                                 (reader.line_num, len(header), len(row), path))
    frame = pd.read_csv(str(path), dtype=str, keep_default_na=False,
                        na_filter=False, encoding="utf-8-sig", low_memory=False)
    # pandas may infer an index for over-wide malformed records. Never accept that.
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("CSV has malformed row widths / inferred index: %s" % path)
    return frame


def require_columns(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("%s missing required columns: %s" % (label, ", ".join(missing)))


def identity_keys(frame, label):
    """Normalize only a matching copy; do not mutate legacy CSV text."""
    require_columns(frame, FULL_KEY, label)
    keys = frame[FULL_KEY].copy().reset_index(drop=True)
    for column in FULL_KEY:
        original = keys[column]
        text = original.astype(str).str.strip()
        invalid = original.isna() | text.eq("")
        if invalid.any():
            raise ValueError("%s identity %s missing/empty in %d rows" %
                             (label, column, int(invalid.sum())))
        if column == "array_index":
            normalized = []
            for value in text:
                try:
                    number = Decimal(value)
                    if not number.is_finite() or number != number.to_integral_value():
                        raise ValueError()
                    if number < 0 or number > 9223372036854775807:
                        raise ValueError()
                    normalized.append(str(int(number)))
                except (InvalidOperation, ValueError, OverflowError):
                    raise ValueError("%s array_index must contain finite nonnegative integers" % label)
            keys[column] = normalized
        else:
            keys[column] = text
    return keys


def numeric_values(series):
    text = series.astype(str).str.strip()
    missing = series.isna() | text.str.lower().isin(MISSING_TOKENS)
    numeric = pd.to_numeric(text.mask(missing), errors="coerce").astype(float)
    status = pd.Series("valid", index=series.index, dtype=object)
    status.loc[missing] = "source_missing"
    status.loc[~missing & numeric.isna()] = "non_numeric"
    status.loc[numeric.notna() & ~np.isfinite(numeric)] = "non_finite"
    return numeric.where(status.eq("valid")), status


def percentage_values(series):
    numeric, status = numeric_values(series)
    outside = numeric.notna() & ~numeric.between(0, 100)
    status.loc[outside] = "outside_0_100"
    return numeric.mask(outside), status


def describe(series):
    values = series.loc[series.notna() & np.isfinite(series)].astype(float)
    n, total = len(values), len(series)
    sd = float(values.std(ddof=1)) if n >= 2 else np.nan
    feasibility = "insufficient_n" if n < 2 else (
        "constant" if values.nunique() == 1 else (
            "eligible_nonconstant" if np.isfinite(sd) and sd > 0 else "non_finite_sd"))
    result = {
        "rows_n": total, "valid_n": n, "missing_n": total - n,
        "missing_pct_of_rows": 100.0 * (total - n) / total if total else np.nan,
        "zero_n": int(values.eq(0).sum()), "hundred_n": int(values.eq(100).sum()),
        "zero_pct_of_valid": 100.0 * values.eq(0).sum() / n if n else np.nan,
        "hundred_pct_of_valid": 100.0 * values.eq(100).sum() / n if n else np.nan,
        "mean": float(values.mean()) if n else np.nan, "sample_sd": sd,
        "min": float(values.min()) if n else np.nan,
        "max": float(values.max()) if n else np.nan,
        "z_feasibility": feasibility,
    }
    for percentile in (1, 5, 25, 50, 75, 95, 99):
        result["p%d" % percentile] = float(values.quantile(percentile / 100.0)) if n else np.nan
    return result


def spearman_pair(left, right):
    """Pearson correlation of average ranks within each complete pair."""
    pairs = pd.DataFrame({"left": left, "right": right})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    result = {"pair_n": len(pairs), "rho": np.nan, "status": "insufficient_n"}
    if len(pairs) < 2:
        return result
    if any(pairs[c].nunique() < 2 for c in pairs.columns):
        result["status"] = "constant"
        return result
    ranks = pairs.rank(method="average")
    result.update(rho=float(ranks["left"].corr(ranks["right"])), status="computed")
    return result


def audit_tables(old, source, tolerance, summary):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("sum tolerance must be finite and nonnegative")
    old = old.reset_index(drop=True)
    source = source.reset_index(drop=True)
    summary.update(old_rows=len(old), iglu_rows=len(source))
    for name, frame in (("old", old), ("iglu", source)):
        summary[name + "_missing_key_columns"] = [c for c in FULL_KEY if c not in frame.columns]
    summary["mage_columns_present"] = {c: c in old.columns for c in MAGE_COLUMNS}
    for column in MAGE_COLUMNS:
        if column in old.columns:
            values, _ = numeric_values(old[column])
            summary[column + "_finite_n"] = int(values.notna().sum())
    require_columns(old, FULL_KEY + MAGE_COLUMNS, "old final table")
    require_columns(source, FULL_KEY + PROPORTIONS, "iglu")
    if old.empty or source.empty:
        raise ValueError("old final table or iglu is empty")
    left, right = identity_keys(old, "old final table"), identity_keys(source, "iglu")
    for name, keys in (("old", left), ("iglu", right)):
        summary[name + "_unique_participants"] = int(keys["participant_id"].nunique())
        summary[name + "_unique_connections"] = len(keys.drop_duplicates(CONNECTION_KEY))
        summary[name + "_unique_full_keys"] = len(keys.drop_duplicates(FULL_KEY))
        summary[name + "_duplicate_full_key_rows"] = int(keys.duplicated(FULL_KEY, keep=False).sum())
    summary["old_duplicate_participant_rows"] = int(left.duplicated(["participant_id"], keep=False).sum())
    if summary["old_duplicate_full_key_rows"] or summary["iglu_duplicate_full_key_rows"]:
        raise ValueError("duplicate full keys; see summary counts")
    if summary["old_duplicate_participant_rows"]:
        raise ValueError("old final table has duplicate participant_id values")
    if not (left["cohort"].eq("10k") & left["research_stage"].eq("00_00_visit")).all():
        raise ValueError("old final table contains rows outside the agreed 10k baseline cohort")
    right["_source_row"] = np.arange(len(right))
    left["_old_row"] = np.arange(len(left))
    joined = left.merge(right, on=FULL_KEY, how="left", sort=False, validate="one_to_one", indicator=True)
    joined = joined.sort_values("_old_row").reset_index(drop=True)
    matched = joined["_merge"].eq("both")
    summary["matched_rows"] = int(matched.sum())
    summary["unmatched_rows"] = int((~matched).sum())
    summary["iglu_rows_not_selected"] = len(source) - int(matched.sum())
    if not matched.all():
        summary["unmatched_key_examples"] = joined.loc[~matched, FULL_KEY].head(10).to_dict("records")
        raise ValueError("%d old connections unmatched in iglu; cannot continue" % int((~matched).sum()))
    selected = source.iloc[joined["_source_row"].astype(int).values].reset_index(drop=True)
    appended = [metric + suffix for metric in PROPORTIONS for suffix in ("_source", "_raw", "_audit_status")]
    collisions = [column for column in appended if column in old.columns]
    if collisions:
        raise ValueError("old table already contains extension columns: %s" % collisions)
    rows = old.copy(deep=True)
    proportions = pd.DataFrame(index=old.index)
    distribution, issues = [], []
    for metric in PROPORTIONS:
        raw, status = percentage_values(selected[metric])
        proportions[metric] = raw
        rows[metric + "_source"] = selected[metric]
        rows[metric + "_raw"] = raw
        rows[metric + "_audit_status"] = status
        report = {"outcome": metric, "version": "legal_raw_percent"}
        report.update(describe(raw))
        for reason in ("source_missing", "non_numeric", "non_finite", "outside_0_100"):
            report[reason + "_n"] = int(status.eq(reason).sum())
        distribution.append(report)
        for i in status.index[~status.eq("valid")]:
            item = rows.loc[i, FULL_KEY].to_dict()
            item.update(outcome=metric, source_value=selected.at[i, metric],
                        status=status.at[i], action="audit_raw_missing_only")
            issues.append(item)
    complete = proportions.notna().all(axis=1)
    sums = proportions.sum(axis=1, min_count=3)
    deviation = sums - 100.0
    # Only accommodates binary floating-point rounding at the stated boundary.
    within = complete & deviation.abs().le(tolerance + 1e-12)
    complement = old[FULL_KEY].copy()
    complement["sum_percent"] = sums
    complement["deviation_percentage_points"] = deviation
    complement["tolerance_percentage_points"] = tolerance
    complement["status"] = np.where(~complete, "not_checkable", np.where(within, "within_tolerance", "outside_tolerance"))
    summary.update(
        complement_tolerance_percentage_points=tolerance,
        complement_checked_n=int(complete.sum()),
        complement_not_checkable_n=int((~complete).sum()),
        complement_within_tolerance_n=int(within.sum()),
        complement_outside_tolerance_n=int((complete & ~within).sum()),
        complement_max_abs_deviation=float(deviation.abs().max()) if complete.any() else None,
    )
    reference_reports, references = [], {}
    for metric in REFERENCES:
        versions = ("raw", "clean", "z") if metric == "cgm_mage" else ("raw", "clean")
        for version in versions:
            column = metric + "_" + version
            report = {"column": column, "present": column in old.columns}
            if column in old.columns:
                numeric, status = numeric_values(old[column])
                references[column] = numeric
                report.update(describe(numeric))
                report.update(source_missing_n=int(status.eq("source_missing").sum()),
                              non_numeric_n=int(status.eq("non_numeric").sum()),
                              non_finite_n=int(status.eq("non_finite").sum()))
            reference_reports.append(report)
    correlations = []
    for metric in PROPORTIONS:
        for reference in REFERENCES:
            for version in ("raw", "clean"):
                column = reference + "_" + version
                report = {"outcome": metric, "outcome_version": "legal_raw_percent", "reference": column}
                report.update(spearman_pair(proportions[metric], references[column]) if column in references
                              else {"pair_n": 0, "rho": np.nan, "status": "reference_column_absent"})
                correlations.append(report)
    qc_reports = []
    for column in QC_COLUMNS:
        report = {"column": column, "source": "old_final_table", "status": "column_absent"}
        if column in old.columns:
            numeric, status = numeric_values(old[column])
            report.update(describe(numeric))
            report.update(status="available", source_missing_n=int(status.eq("source_missing").sum()),
                          non_numeric_n=int(status.eq("non_numeric").sum()),
                          non_finite_n=int(status.eq("non_finite").sum()))
            invalid_domain = numeric.lt(0)
            if column == "cgm_qc_loss_fraction":
                invalid_domain = invalid_domain | numeric.gt(1)
            report["outside_expected_domain_n"] = int(invalid_domain.sum())
        qc_reports.append(report)
    pd.testing.assert_frame_equal(rows[old.columns], old, check_exact=True)
    summary.update(old_columns_preserved=True, old_row_order_preserved=True,
                   old_mage_recomputed=False, proportion_cleaning="pending_server_audit_review",
                   new_z_scores_computed=False, final_analysis_table_generated=False,
                   missing_numeric_tokens=sorted(MISSING_TOKENS),
                   qc_missing_columns=[c for c in QC_COLUMNS if c not in old.columns])
    return {
        "audit_rows": rows,
        "proportion_summary": pd.DataFrame(distribution),
        "value_issues": pd.DataFrame(issues, columns=FULL_KEY + ["outcome", "source_value", "status", "action"]),
        "complementarity": complement,
        "legacy_reference_summary": pd.DataFrame(reference_reports),
        "correlations": pd.DataFrame(correlations),
        "qc_summary": pd.DataFrame(qc_reports),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fingerprint(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def print_reports(reports):
    print("\n=== PROPORTION DISTRIBUTIONS / 新比例分布（每项独立有效样本） ===")
    print("缺失率分母=旧表总行数；0/100比例分母=该项合法有效人数；SD ddof=1。")
    for report in reports["proportion_summary"].to_dict("records"):
        print("\n[" + report["outcome"] + "]")
        for key, value in report.items():
            if key != "outcome":
                print("%s=%s" % (key, value))
    for name, title in [
        ("legacy_reference_summary", "LEGACY REFERENCE / 旧指标及 MAGE"),
        ("correlations", "SPEARMAN / 每对独立完整样本，含并列秩"),
        ("qc_summary", "AVAILABLE QC / 仅旧表已有字段"),
    ]:
        print("\n=== %s ===" % title)
        print(reports[name].to_string(index=False, float_format=lambda x: "%.6g" % x))
    print("\nNOTE: z_feasibility 仅描述数值可计算性；未生成新 Z，也不代表适合连续结局模型。")
    print("NOTE: 三比例合计检查不修改源值；TIR 为辅助指标，不能视作独立结局。")
    print("NOTE: QC 字段缺失不推算覆盖率；旧药物排除状态原样保留。")


def parse_args(argv=None):
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-csv", default=os.environ.get("CGM_CORE_CSV", str(defaults["core_csv"])))
    parser.add_argument("--iglu-csv", default=os.environ.get("CGM_IGLU_CSV", str(defaults["iglu_csv"])))
    parser.add_argument("--output-dir", default=os.environ.get("CGM_EXTENSION_OUTPUT_DIR", str(defaults["output_dir"])))
    parser.add_argument("--sum-tolerance", type=float, default=0.1,
                        help="Absolute percentage-point tolerance around 100 (default: 0.1); no rescaling")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("audit_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    output = Path(args.output_dir).expanduser().resolve()
    report_dir, log_dir = output / "reports" / run_id, output / "logs"
    report_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (run_id + ".log")
    manifest = {
        "run_id": run_id, "script_version": VERSION, "status": "running",
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "core_csv": str(Path(args.core_csv).expanduser().resolve()),
        "iglu_csv": str(Path(args.iglu_csv).expanduser().resolve()),
        "report_dir": str(report_dir), "log_file": str(log_path),
        "sum_tolerance_percentage_points": args.sum_tolerance,
        "summary": {},
    }
    exit_code = 1
    with open(str(log_path), "x", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("=== CGM EXTENSION AUDIT %s ===" % VERSION)
            for key in ("core_csv", "iglu_csv", "report_dir", "log_file", "python", "pandas", "numpy"):
                print("%s=%s" % (key.upper(), manifest[key]))
            try:
                if not np.isfinite(args.sum_tolerance) or args.sum_tolerance < 0:
                    raise ValueError("sum tolerance must be finite and nonnegative")
                paths = [Path(manifest[key]) for key in ("core_csv", "iglu_csv")]
                manifest["input_files"] = [fingerprint(path) for path in paths]
                manifest["script"] = fingerprint(Path(__file__).resolve())
                old, source = [read_table(path) for path in paths]
                print("INPUT_ROWS old=%d iglu=%d" % (len(old), len(source)))
                reports = audit_tables(old, source, args.sum_tolerance, manifest["summary"])
                # Detect input changes during the audit before publishing reports.
                if manifest["input_files"] != [fingerprint(path) for path in paths]:
                    raise ValueError("input files changed during audit; rerun with stable inputs")
                print_reports(reports)
                manifest["report_files"] = []
                for name, frame in reports.items():
                    target = report_dir / (name + ".csv")
                    frame.to_csv(str(target), index=False, encoding="utf-8")
                    manifest["report_files"].append(str(target))
                manifest["status"] = "audit_complete_review_required"
                exit_code = 0
            except Exception as error:
                manifest["status"] = "failed"
                manifest["error"] = "%s: %s" % (type(error).__name__, error)
                print("ERROR=" + manifest["error"])
            finally:
                print("\n=== INPUT / MATCH / COMPLEMENT SUMMARY ===")
                print(json.dumps(json_safe(manifest["summary"]), ensure_ascii=False, indent=2, allow_nan=False))
                print("STATUS=" + manifest["status"])
                print("MANIFEST=" + str(report_dir / "manifest.json"))
                print("LOG_FILE=" + str(log_path))
                manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                with open(str(report_dir / "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
