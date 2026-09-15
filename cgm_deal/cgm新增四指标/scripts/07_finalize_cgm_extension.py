#!/usr/bin/env python3
"""Append legality-only proportion phenotypes to the unchanged legacy table.

Requires sibling 06_audit_cgm_extension.py and a completed audit manifest.
No outlier removal, winsorization, transformations, or downstream associations.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import pandas as pd

VERSION = "2026-09-15.1"
RULE_ID = "2026-09-15_legal_percent_no_winsor_per_outcome_z_ddof1"
AUDIT_PATH = Path(__file__).resolve().with_name("06_audit_cgm_extension.py")
_spec = importlib.util.spec_from_file_location("cgm_extension_audit", str(AUDIT_PATH))
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def standardize(series):
    """Use all finite values for one outcome; leave degenerate Z undefined."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.where(np.isfinite(values))
    n = int(values.notna().sum())
    mean = float(values.mean()) if n else np.nan
    sd = float(values.std(ddof=1)) if n >= 2 else np.nan
    status = "insufficient_n" if n < 2 else (
        "constant" if values.nunique() == 1 else (
            "computed" if np.isfinite(sd) and sd > 0 else "non_finite_sd"))
    z = pd.Series(np.nan, index=series.index, dtype=float)
    if status == "computed":
        z = (values - mean) / sd
        if not np.isfinite(z.loc[values.notna()]).all():
            raise ValueError("Non-finite Z despite a nondegenerate input")
    parameters = {
        "clean_valid_n": n, "clean_mean": mean, "clean_sample_sd": sd,
        "z_status": status, "z_valid_n": int(z.notna().sum()),
        "z_mean": float(z.mean()) if z.notna().any() else np.nan,
        "z_sample_sd": float(z.std(ddof=1)) if z.notna().sum() >= 2 else np.nan,
    }
    if status == "computed":
        if not np.isclose(parameters["z_mean"], 0, atol=1e-10, rtol=0):
            raise ValueError("Z validation failed: mean is not approximately zero")
        if not np.isclose(parameters["z_sample_sd"], 1, atol=1e-10, rtol=0):
            raise ValueError("Z validation failed: sample SD is not approximately one")
    return z, parameters


def build_extension(old, audited_rows):
    """Retain all legacy CSV text and append only new outcome columns."""
    pd.testing.assert_frame_equal(audited_rows[list(old.columns)], old, check_exact=True)
    appended = [metric + suffix for metric in audit.PROPORTIONS
                for suffix in ("_source", "_raw", "_clean", "_z", "_value_status", "_z_status")]
    collisions = [c for c in appended if c in old.columns]
    if collisions:
        raise ValueError("Existing extension columns would be overwritten: %s" % collisions)
    final = old.copy(deep=True)
    reports = []
    for metric in audit.PROPORTIONS:
        raw, value_status = audit.percentage_values(audited_rows[metric + "_source"])
        expected = audited_rows[metric + "_raw"].astype(float)
        pd.testing.assert_series_equal(raw, expected, check_names=False, check_exact=True)
        z, parameters = standardize(raw)
        final[metric + "_source"] = audited_rows[metric + "_source"]
        final[metric + "_raw"] = raw
        final[metric + "_clean"] = raw.copy()
        final[metric + "_z"] = z
        final[metric + "_value_status"] = value_status
        final[metric + "_z_status"] = np.where(raw.isna(), "missing_input", parameters["z_status"])
        report = {"outcome": metric, "unit": "percent_0_to_100", "rule_id": RULE_ID,
                  "clean_rule": "legal_raw_unchanged", "z_population": "all_legacy_rows_valid_for_this_outcome",
                  "sd_ddof": 1, "missing_n": int(raw.isna().sum()),
                  "zero_n": int(raw.eq(0).sum()), "min": raw.min(), "max": raw.max()}
        report.update(parameters)
        reports.append(report)
    pd.testing.assert_frame_equal(final[list(old.columns)], old, check_exact=True)
    return final, pd.DataFrame(reports)


def verify_input_fingerprints(manifest):
    """Require exactly the two source files recorded by the reviewed audit."""
    recorded = manifest.get("input_files", [])
    if len(recorded) != 2:
        raise ValueError("Audit manifest lacks the two input fingerprints")
    paths = [Path(manifest[key]) for key in ("core_csv", "iglu_csv")]
    current = [audit.fingerprint(path) for path in paths]
    if current != recorded:
        raise ValueError("Source input fingerprints changed since audit; rerun and review the audit")
    return paths


def load_reviewed_manifest(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "audit_complete_review_required":
        raise ValueError("A completed audit manifest is required")
    if manifest.get("script_version") != audit.VERSION:
        raise ValueError("Audit script version differs from the reviewed run")
    if manifest.get("script", {}).get("sha256") != audit.fingerprint(AUDIT_PATH)["sha256"]:
        raise ValueError("Audit script contents differ from the reviewed run")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, help="Exact completed audit directory containing manifest.json")
    parser.add_argument("--output-dir", help="Extension root; defaults to that audit's extension root")
    args = parser.parse_args(argv)
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else audit_dir.parent.parent
    run_id = datetime.now(timezone.utc).strftime("finalize_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    report_dir = output / "reports" / run_id
    data_dir = output / "data" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (run_id + ".log")
    run = {
        "run_id": run_id, "script_version": VERSION, "rule_id": RULE_ID, "status": "running",
        "reviewed_audit_manifest": str(audit_dir / "manifest.json"),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "rule": {"clean": "legal raw percentage unchanged", "transformation": "none",
                 "winsorization": "none", "z_population": "per_outcome_valid_full_legacy_cohort",
                 "z_sd_ddof": 1, "mage": "reuse_legacy_raw_clean_z",
                 "clinical_trace_validation": "not_performed", "association_models": "not_selected"},
    }
    exit_code = 1
    with open(str(log_path), "x", encoding="utf-8") as log:
        tee = audit.Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("=== CGM EXTENSION FINALIZATION %s ===" % VERSION)
            print("RULE_ID=" + RULE_ID)
            try:
                manifest_path = audit_dir / "manifest.json"
                manifest_fingerprint = audit.fingerprint(manifest_path)
                reviewed = load_reviewed_manifest(manifest_path)
                paths = verify_input_fingerprints(reviewed)
                run["input_files"] = reviewed["input_files"]
                run["audit_manifest_fingerprint"] = manifest_fingerprint
                run["script"] = audit.fingerprint(Path(__file__).resolve())
                run["audit_script"] = audit.fingerprint(AUDIT_PATH)
                old, source = [audit.read_table(path) for path in paths]
                summary = {}
                reports = audit.audit_tables(old, source, reviewed["sum_tolerance_percentage_points"], summary)
                if audit.json_safe(summary) != reviewed["summary"]:
                    raise ValueError("Recomputed audit summary differs from the reviewed run")
                final, parameters = build_extension(old, reports["audit_rows"])
                # Recheck source snapshots before writing any final phenotype table.
                verify_input_fingerprints(reviewed)
                if audit.fingerprint(manifest_path) != manifest_fingerprint:
                    raise ValueError("Audit manifest changed during finalization")
                data_dir.mkdir(parents=True, exist_ok=False)
                final_path = data_dir / "06_cgm_extended_phenotypes.csv"
                pending = data_dir / "06_cgm_extended_phenotypes.pending.csv"
                final.to_csv(str(pending), index=False, encoding="utf-8")
                written = audit.read_table(pending)
                pd.testing.assert_frame_equal(written[list(old.columns)], old, check_exact=True)
                pending.rename(final_path)
                parameters.to_csv(str(report_dir / "standardization_parameters.csv"), index=False, encoding="utf-8")
                complete4 = final["cgm_mage_clean"].pipe(audit.numeric_values)[0].notna()
                for metric in audit.PROPORTIONS:
                    complete4 &= final[metric + "_clean"].notna()
                run["summary"] = {
                    "participants_retained": len(final), "legacy_columns_preserved": True,
                    "legacy_row_order_preserved": True, "mage_recomputed": False,
                    "mage_raw_finite_n": summary["cgm_mage_raw_finite_n"],
                    "mage_clean_finite_n": summary["cgm_mage_clean_finite_n"],
                    "mage_z_finite_n": summary["cgm_mage_z_finite_n"],
                    "four_clean_outcomes_complete_n_descriptive_only": int(complete4.sum()),
                    "complete_case_filter_applied": False,
                    "new_outcomes_with_computable_z": int(parameters["z_status"].eq("computed").sum()),
                }
                run["standardization_parameters"] = parameters.to_dict("records")
                run["final_file"] = audit.fingerprint(final_path)
                run["status"] = "finalized"
                print(json.dumps(audit.json_safe(run["summary"]), ensure_ascii=False, indent=2))
                print("\n=== 每项独立标准化参数（SD ddof=1） ===")
                for row in parameters.to_dict("records"):
                    print("\n[" + row["outcome"] + "]")
                    for key in ("clean_valid_n", "missing_n", "zero_n", "min", "max", "clean_mean",
                                "clean_sample_sd", "z_status", "z_valid_n", "z_mean", "z_sample_sd"):
                        print("%s=%s" % (key, row[key]))
                print("FINAL_FILE=" + str(final_path))
                print("NOTE: 四项同时完整人数仅供描述，不用于筛人；饮食/菌群合并后的 N 另算。")
                print("NOTE: Z 不改变偏态或零值集中；关联模型未确定，原始曲线尚未核验。")
                exit_code = 0
            except Exception as error:
                run["status"] = "failed"
                run["error"] = "%s: %s" % (type(error).__name__, error)
                print("ERROR=" + run["error"])
            finally:
                print("STATUS=" + run["status"])
                print("LOG_FILE=" + str(log_path))
                run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                with open(str(report_dir / "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump(audit.json_safe(run), handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
