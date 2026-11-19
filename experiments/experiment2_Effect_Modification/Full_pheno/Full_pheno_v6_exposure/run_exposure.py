"""Compare alternative exposures or reselect diet models within discovery households."""

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v6"))

from run_diet_audit import ANCHOR, archive, v1_run
from run_v2 import directory_id, require


def emit(event, fields):
    print("FULL_PHENO_V6_" + event, json.dumps(fields, allow_nan=False), flush=True)


def summarise(study, fits, methods, baseline, draws):
    import numpy as np
    from diet_audit import METRICS, household_moments, simultaneous

    _, groups = np.unique(study.family, return_inverse=True)
    count = int(groups.max()) + 1
    moments = np.stack([household_moments(parts, groups, count) for parts in fits.values()])
    r2, points, radius = simultaneous(moments, draws)
    rows = []
    for i, ((exposure, endpoint, feature), parts) in enumerate(fits.items()):
        people = int(moments[i, :, 0].sum())
        families = int((moments[i, :, 0] > 0).sum())
        total = baseline.get((exposure, endpoint, feature), people)
        for j, method in enumerate(methods):
            a, b, c = map(float, r2[i, j])
            row = dict(exposure=exposure, endpoint=endpoint, feature=feature, method=method,
                       people=people, households=families, legacy_people=total, coverage=people/total,
                       r2_no_diet=a, r2_off=b, r2_on=c)
            for k, label in enumerate(METRICS):
                label = "vs_reference_pp" if label == "vs_pc5_half_pp" else label
                center, width = float(points[i, j, k]), float(radius[i, j, k])
                row[label] = center
                row[label + "_95"] = [center-width, center+width]
            row["positive_interaction_folds"] = sum(
                int(np.sum((x.outcome - x.off[:, j])**2 - (x.outcome - x.on[:, j])**2) > 0.) for x in parts)
            rows.append(row)
    return rows


def report(result):
    from statistics import median

    phase, rows = result["phase"], result["rows"]
    emit("SCOPE", {"phase": phase, "rows": len(rows), "bootstrap_draws": result["bootstrap_draws"],
                   "reserved_outcomes_read": False})
    if phase == "features":
        emit("FOOD_SUPPORT", result["food_support"])
        emit("SPLIT_HALF", result["split_half"])
        for name in sorted({r["feature"] for r in rows}):
            group = [r for r in rows if r["feature"] == name and r["method"] == "new_pc5_full"]
            emit("FEATURE_SUMMARY", dict(feature=name, evaluated=len(group),
                                         median_coverage=median(r["coverage"] for r in group),
                                         median_total_pp=median(r["diet_total_pp"] for r in group),
                                         improved_on_reference=sum(r["vs_reference_pp"] > 0 for r in group),
                                         interaction_ge_010=sum(r["interaction_pp"] >= .10 for r in group)))
        candidates = [r for r in rows if r["method"] != "matched_reference"]
        winners = {}
        for r in sorted(candidates, key=lambda r: r["vs_reference_pp"], reverse=True):
            winners.setdefault((r["exposure"], r["endpoint"]), r)
        for r in list(winners.values())[:8]:
            keys = ("exposure", "endpoint", "feature", "method", "coverage", "diet_total_pp", "interaction_pp",
                    "vs_reference_pp", "vs_reference_pp_95")
            emit("EXPLORATORY_PEAK", {k: r[k] for k in keys})
    else:
        for method in dict.fromkeys(r["method"] for r in rows):
            all_rows = [r for r in rows if r["method"] == method]
            group = [r for r in all_rows if (r["exposure"], r["endpoint"]) != ANCHOR]
            increments = [r["interaction_pp"] for r in group]
            pos, reach, drop = (sum(x > 0. for x in increments), sum(x >= .1 for x in increments),
                                 sum(x <= -.1 for x in increments))
            emit("NESTED_WORKFLOW", dict(method=method, pos17=pos, ge010=reach, le_neg010=drop,
                                        passes_development_gate=pos >= 9 and reach >= 2 and drop == 0,
                                        adjusted_positive17=sum(r["interaction_pp_95"][0] > 0 for r in group),
                                        adjusted_beats_reference=sum(r["vs_reference_pp_95"][0] > 0 for r in group),
                                        negative_total18=sum(r["diet_total_pp"] < 0 for r in all_rows),
                                        median_interaction_pp=median(increments),
                                        min_interaction_pp=min(increments), max_interaction_pp=max(increments)))
    emit("COMPLETE", {"phase": phase, "registered_comparisons": len(rows), "skipped": len(result.get("skipped", []))})


def execute(args, snapshot, identity, result_dir):
    from exposure_models import FEATURE_METHODS, NESTED_METHODS, feature_fits, nested_fits
    import protocol as p
    from source import Study

    study = Study(snapshot, with_upf=True)
    phase_file = result_dir / "summary.json"
    if args.phase == "features":
        from food_features import read_features

        features, paths = read_features(snapshot.root, study)
        paths = tuple(paths.values())
        details = dict(food_support=features.flow, split_half=features.reliability)
        emit("FEATURE_INPUTS", features.flow)
        methods = FEATURE_METHODS
    else:
        from upf_refit import load

        calibration, paths = load(snapshot.root, study)
        methods = NESTED_METHODS
        details = {}
    identity["extra_inputs"] = {str(path): v1_run.file_sha(path) for path in paths}
    meta = result_dir / "registration.json"
    if meta.exists():
        require(v1_run.read_json(meta) == identity, "V6_EXPOSURE_REGISTRATION_DIFF")
    else:
        v1_run.atomic_write(meta, v1_run.encode(identity))
    if phase_file.exists():
        return v1_run.read_json(phase_file)

    outcomes = {e: study.outcome(e) for e in p.ENDPOINTS}
    if args.phase == "features":
        fits, skipped, coverage = feature_fits(study, outcomes, features.values, emit)
        details["skipped"] = skipped
    else:
        fits, audit, choices = nested_fits(study, outcomes, calibration, emit)
        details.update(upf_inner_audit=audit, choices=choices)
        coverage = {}
    require(bool(fits), "V6_EXPOSURE_NO_VALID_FITS")
    rows = summarise(study, fits, methods, coverage, p.BOOTSTRAP_DRAWS)
    require(set(snapshot.paths) <= snapshot.verified, "V6_EXPOSURE_UNVERIFIED_INPUTS")
    snapshot.verify_manifest()
    require(all(v1_run.file_sha(Path(path)) == digest for path, digest in identity["extra_inputs"].items()),
            "V6_EXPOSURE_SOURCE_BYTES_CHANGED")
    result = dict(phase=args.phase, scope="discovery only", rows=rows, reserved_outcomes_read=False,
                  bootstrap_draws=p.BOOTSTRAP_DRAWS, **details)
    v1_run.atomic_write(phase_file, v1_run.encode(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("features", "nested"), required=True)
    parser.add_argument("--run-id", type=directory_id, required=True)
    parser.add_argument("--commit", required=False)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    out = HERE / "runs/TRE_ONLY" / args.run_id
    if args.report:
        result = v1_run.read_json(out / "summary.json")
        require(result["phase"] == args.phase, "V6_EXPOSURE_WRONG_PHASE")
        report(result)
        return
    require(args.commit and len(args.commit) == 40 and all(x in "0123456789abcdef" for x in args.commit),
            "V6_EXACT_COMMIT_REQUIRED")
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from source import Snapshot, required_files

    root = HERE.parents[2]
    require(root.name == p.HANDOFF, "V6_EXPOSURE_HANDOFF_ROOT")
    snapshot = Snapshot(root, required_files(p.MODES["all"]))
    release, _ = archive.verify_release()
    previous = HERE.parent / "Full_pheno_v5/runs/TRE_ONLY/discovery-1"
    archive.read_complete(previous, release)
    registered = v1_run.read_json(previous / "registration.json")
    require(registered["snapshot_sha"] == snapshot.manifest_sha and registered["sources"] == snapshot.report(),
            "V6_EXPOSURE_V5_SNAPSHOT_DIFFERENT")
    files = sorted(HERE.glob("*.py")) + [HERE.parent / "Full_pheno_v6/diet_audit.py"]
    identity = dict(commit=args.commit, phase=args.phase, runtime=runtime, snapshot=snapshot.manifest_sha,
                    source_code={path.name: v1_run.file_sha(path) for path in files})
    with v1_run.run_lock(out):
        emit("START", dict(phase=args.phase, run_id=args.run_id, scope="discovery only"))
        result = execute(args, snapshot, identity, out)
    report(result)


if __name__ == "__main__":
    main()
