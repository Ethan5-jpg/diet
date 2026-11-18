"""Read saved aggregate scores; never fit models or read participant records.

Run with Python 3.9+ and --root pointing to collaborator_handoff_20261102_v1.
This fills gaps in the local archive; it is not a fresh validation experiment.
No files are written. Only allowlisted summary JSON files and logs are read.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def paired_row(label, repeat, base, full, delta, source, stage, ci=None):
    base, full, delta = map(float, (base, full, delta))
    if not all(map(math.isfinite, (base, full, delta))):
        raise ValueError("NONFINITE_SCORE")
    # Permit independent rounding of displayed R2 and delta to six decimals.
    if abs((full - base) - delta) > 2.1e-6:
        raise ValueError("UNMATCHED_R2_AND_DELTA")
    return dict(label=label, repeat=repeat, base_r2=base, full_r2=full,
                delta_r2=delta, delta_r2_percentage_points=100 * delta,
                ratio_over_full_pct=100 * delta / full if full else None,
                ratio_over_base_pct=100 * delta / base if base else None,
                positive_r2_denominators=base > 0 and full > 0,
                original_delta_ci95=ci, stage=stage, source=source)


def read_json(path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("AGGREGATE_SIZE_LIMIT")
    return json.loads(path.read_text(encoding="utf-8"))


def scan(root):
    exp2 = root / "experiment2_Effect_Modification"
    rows, issues = [], []

    def report_error(path, exc):
        issues.append(dict(source=str(path.relative_to(root)), error=type(exc).__name__))

    def add(label, repeat, base, full, delta, path, stage, ci=None):
        rows.append(paired_row(label, repeat, base, full, delta,
                               str(path.relative_to(root)), stage, ci))

    # Native v1 summaries contain all saved routes, including ones absent from screenshots.
    for path in sorted((exp2 / "v1/runs_TRE_ONLY").glob("*/summary.json")):
        try:
            done = read_json(path.parent / "COMPLETE.json")
            if hashlib.sha256(path.read_bytes()).hexdigest() != done["summary_sha"]:
                raise ValueError("SUMMARY_HASH_CHANGED")
            data = read_json(path)
            if data["exposure"] != "fiber_density" or data["outcome"] != "D_Glu":
                raise ValueError("UNEXPECTED_TARGET")
            for repeat, scores in enumerate(data["scores"]):
                base = scores["common_main"]["r2"]
                for name, item in scores.items():
                    if name != "common_main" and "r2" in item and "delta" in item:
                        add("exp2_v1:" + name, repeat, base, item["r2"], item["delta"],
                            path, "discovery_family_oof", item.get("ci"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report_error(path, exc)

    # v3/v4 print complete score dictionaries to logs, without saving native rows here.
    for version in (3, 4):
        prefix = f"EXP2_V{version}_PREDICT "
        completed = f"EXP2_V{version}_COMPLETE "
        for path in sorted((exp2 / f"v{version}").glob("*.log")):
            try:
                if path.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError("LOG_SIZE_LIMIT")
                text = path.read_text(encoding="utf-8")
                if not re.search(r"(?m)^" + re.escape(completed), text):
                    continue
                for line in text.splitlines():
                    if not line.startswith(prefix):
                        continue
                    data = json.loads(line[len(prefix):])
                    background = data.get("background", "shared")
                    for name, item in data["routes"].items():
                        add(f"exp2_v{version}:{background}:{name}", data["repeat"],
                            data["baseline_r2"], item.get("full_r2", item.get("r2")),
                            item["delta_r2"], path, "discovery_family_oof",
                            item.get("ci95", item.get("delta_ci")))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                report_error(path, exc)

    path = exp2 / "v5/runs/TRE_ONLY/v5_reserved/reserved_summary.json"
    if path.is_file():
        try:
            done = read_json(path.parent / "COMPLETE.json")
            if hashlib.sha256(path.read_bytes()).hexdigest() != done["summary_sha256"]:
                raise ValueError("SUMMARY_HASH_CHANGED")
            data = read_json(path)
            add("exp2_v5:fermentation_smooth_curve@0.1", None, data["base_r2"],
                data["full_r2"], data["delta_r2"], path, "frozen_HPP_reserved",
                data.get("delta_r2_ci95"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report_error(path, exc)

    # Legacy nested v5 stores MSEs, MSE gain and delta, sufficient to recover variance.
    legacy = root / "experiments/diet_gut_fiber_modules_20261104_v5"
    for kind in ("pc", "module"):
        for repeat in (0, 1):
            path = legacy / f"{kind}_TRE_ONLY/round{repeat}.json"
            if not path.is_file():
                continue
            try:
                data = read_json(path)
                delta, gain = float(data["delta_R2"]), float(data["mse_gain"])
                if delta == 0 or gain == 0:
                    raise ValueError("OUTCOME_VARIANCE_NOT_RECOVERABLE_FROM_ZERO_GAIN")
                variance = gain / delta
                if not math.isfinite(variance) or variance <= 0:
                    raise ValueError("INVALID_RECOVERED_VARIANCE")
                base = 1 - float(data["MSE_null"]) / variance
                full = 1 - float(data["MSE_model"]) / variance
                add("legacy_v5:" + kind, repeat, base, full, delta, path,
                    "discovery_nested_family_oof")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                report_error(path, exc)

    # Identical repeated logs are one numerical result, not independent replications.
    unique = {}
    for row in rows:
        key = tuple(row[k] for k in ("label", "repeat", "base_r2", "full_r2", "delta_r2"))
        if key not in unique:
            unique[key] = row
    return list(unique.values()), issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--all", action="store_true", help="Print every paired result as aggregate JSON.")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "experiment2_Effect_Modification").is_dir():
        raise SystemExit("RATIO_AUDIT_STOP ROOT_NOT_FOUND")
    rows, issues = scan(root)
    eligible = [r for r in rows if r["positive_r2_denominators"] and r["delta_r2"] > 0]
    eligible.sort(key=lambda r: r["ratio_over_full_pct"], reverse=True)
    for row in (rows if args.all else eligible[:10]):
        print("RATIO_AUDIT_RESULT " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    for row in rows:
        if row["stage"] == "frozen_HPP_reserved" and not args.all:
            print("RATIO_AUDIT_RESERVED " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    for issue in issues:
        print("RATIO_AUDIT_ISSUE " + json.dumps(issue, ensure_ascii=False))
    print("RATIO_AUDIT_COMPLETE " + json.dumps(dict(
        paired_results=len(rows), positive_results_with_positive_r2=len(eligible),
        issues=len(issues), individual_records_read=False, models_fitted=False,
        files_written=False, scope="allowlisted saved summaries/logs only; missing logs are not zero results")))


if __name__ == "__main__":
    main()
