"""Run the V8 discovery experiment or show a completed aggregate report."""
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parent
V6, V7 = ROOT.parent / "v6", ROOT.parent / "v7"
sys.path[1:1] = [str(V7), str(V6)]
import run_v6 as store
import run_v7


VERSION = "effect_modification_v8_ecology_maximin_20261118"
RUN = ROOT / "runs/TRE_ONLY" / VERSION
V7_FAMILY = "gut_pc10_ecology_both"
ARTIFACTS = tuple(run_v7.artifacts())


def emit(tag, value):
    store.emit("EXP2_V8_" + tag, value)


def release_check():
    manifest = store.read_json(ROOT / "upload_manifest.json")
    if manifest["version"] != VERSION or set(manifest["packages"]) != {"v6", "v7", "v8"}:
        raise ValueError("Different or incomplete V8 upload manifest")
    for version, entries in manifest["packages"].items():
        for filename, expected in entries.items():
            path = ROOT.parent / version / filename
            if not path.is_file() or store.digest(path) != expected:
                raise ValueError(f"Missing or changed upload file {version}/{filename}")
    return manifest


def protocol():
    import v8_models as models
    import v8_scores as scores
    import v8_selection as selection
    config = dict(version=VERSION, scope="discovery", exposure="fiber_density", outcome="D_Glu",
                  outer_repeats=2, outer_household_folds=5, inner_repeats=2, inner_household_folds=3,
                  first_inner="archived V7 family folds",
                  second_inner_seed_base=selection.SECOND_INNER_SEED,
                  inner_objective="maximize the smaller of the two paired inner ratios",
                  outer_objective="smaller of the two outer OOF percentage gains for the same procedure",
                  target_pct=scores.TARGET_PCT, bootstrap_draws=1999, bootstrap_seed=scores.BOOT_SEED,
                  historical_parent=V7_FAMILY, settings_count=len(models.KEYS),
                  family_count=len(models.FAMILIES), routes=selection.NAMES, primary=selection.PRIMARY,
                  reserved_outcomes_read=False, **models.design())
    return json.loads(json.dumps(config, allow_nan=False))


def check_v7_predictions(prior_run, folds):
    """Read the archived predictions only after all V8 selection and prediction."""
    import numpy as np
    from v8_selection import REFERENCE
    if prior_run is None or not (prior_run / "COMPLETE.json").is_file():
        return dict(status="v7_archive_unavailable_anchor_recomputed")
    receipt_sha = store.digest(prior_run / "COMPLETE.json")
    # Validate the V7 completion receipt and all hashes before reading a column.
    run_v7.read_complete(prior_run, store.read_json(V7 / "upload_manifest.json"))
    maximum = np.zeros(2)
    for r in range(2):
        for f in range(5):
            old, meta7 = store.read_fold(prior_run, r, f)
            new, meta8 = folds[r, f]
            if not all(np.array_equal(old[key], new[key]) for key in ("rows", "groups", "y")):
                raise ValueError(f"V7 anchor row/household/target mismatch at r{r} f{f}")
            choice7, choice8 = meta7["choices"][V7_FAMILY], meta8["choices"][REFERENCE]
            expected8 = "pc10_ecology@" + choice7["setting"].rsplit("@", 1)[-1]
            if choice7["enabled"] != choice8["enabled"] or choice8["settings"] != [expected8]:
                raise ValueError(f"V7 anchor penalty or activation mismatch at r{r} f{f}")
            i, j = meta7["names"].index(V7_FAMILY), meta8["names"].index(REFERENCE)
            diff = [np.max(np.abs(old[key][:, i] - new[key][:, j])) for key in ("off", "on")]
            if not np.isfinite(diff).all() or max(diff) > 2e-8:
                raise ValueError(f"V7 anchor prediction mismatch at r{r} f{f}: {diff}")
            maximum = np.maximum(maximum, diff)
    if store.digest(prior_run / "COMPLETE.json") != receipt_sha:
        raise ValueError("V7 receipt changed during replay check")
    return dict(status="v7_predictions_match", folds_checked=10, max_abs_off_on=maximum.tolist(),
                tolerance_mg_dl=2e-8, v7_receipt_sha256=receipt_sha)


def write_csv(path, summary):
    """One self-contained aggregate row per route per repeat; replace in one step."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     newline="", delete=False) as stream:
        temp = Path(stream.name)
        try:
            writer = None
            for name, result in summary["routes"].items():
                common = {k: v for k, v in result.items() if k != "repeats"}
                for score in result["repeats"]:
                    row = dict(route=name, **score, **common)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v
                                     for k, v in row.items()})
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def completed(run, release):
    receipt = store.read_json(run / "COMPLETE.json")
    if receipt["version"] != VERSION or set(receipt["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Incomplete V8 run receipt")
    for filename, expected in receipt["artifacts"].items():
        if store.digest(run / filename) != expected:
            raise ValueError("Completed artifact changed: " + filename)
    reg = store.read_json(run / "registration.json")
    if reg["release"] != release or reg["protocol"]["version"] != VERSION:
        raise ValueError("Completed results belong to another V8 release")
    return store.read_json(run / "summary.json")


def execute(source, run, release, threads, *, draws=1999, prior_run=None):
    import v8_scores
    import v8_selection
    registration = dict(protocol=protocol(), release=release, draws=draws,
                        runtime=store.runtime_identity(threads), cohort=store.cohort_identity(source),
                        source_hashes={str(k): v for k, v in source.source_hashes.items()})
    with store.lock(run):
        path = run / "registration.json"
        if path.exists():
            if store.read_json(path) != registration:
                raise ValueError("V8 registration differs from this run; preserve existing checkpoints")
        else:
            if any(run.glob("r*_f*")) or (run / "COMPLETE.json").exists():
                raise ValueError("Unregistered V8 checkpoint")
            store.write_json(path, registration)
        if (run / "COMPLETE.json").is_file():
            return completed(run, release)
        folds = {}
        for r in range(2):
            for f in range(5):
                if (run / f"r{r}_f{f}").is_dir():
                    folds[r, f] = store.read_fold(run, r, f)
                    emit("REUSED", dict(repeat=r, fold=f))
                    continue
                start = time.monotonic()
                emit("START", dict(repeat=r, fold=f))
                arrays, meta = v8_selection.fold_predictions(source, r, f)
                source.verify_sources()
                store.save_fold(run, r, f, arrays, meta)
                folds[r, f] = arrays, meta
                emit("OUTER_COMPLETE", dict(repeat=r, fold=f, people=meta["people"],
                                            seconds=round(time.monotonic()-start, 1)))
        result = v8_scores.summarize(source, folds, draws)
        result["v7_replay"] = check_v7_predictions(prior_run, folds)
        source.verify_sources()
        store.write_json(run / "summary.json", result)
        write_csv(run / "all_routes.csv", result)
        store.write_json(run / "COMPLETE.json", dict(version=VERSION,
                         artifacts={name: store.digest(run / name) for name in ARTIFACTS}))
        return completed(run, release)


def compact(summary):
    def number(x):
        return "NA" if x is None else f"{x:+.3f}"
    def ci(bounds):
        return "NA" if bounds is None else "[" + ",".join(map(number, bounds)) + "]"
    emit("REPLAY", summary["v7_replay"])
    print(f"EXP2_V8_SCREEN_BEGIN | N={summary['people']} HH={summary['households']} "
          f"| target > {summary['benchmark_pct']:.2f}% in BOTH splits")
    names = ["v7_ecology_replay", "pc10_ecology", "nested_maximin", "nested_maximin_guarded", "nested_top3_blend",
             *summary["exploratory_ranking_by_minimum"][:3], summary["best_stable_family"]]
    print("  route | ratio% r0/r1 | min% | min minus V7 (pp) | delta R2 r0/r1 (pp) | both > target point/adjusted")
    for name in dict.fromkeys(n for n in names if n):
        row = summary["routes"][name]
        r0, r1 = row["repeats"]
        print(f"  {name} | {number(r0['ratio_pct'])}/{number(r1['ratio_pct'])} "
              f"| {number(row['minimum_ratio_pct'])} | {number(row['minimum_ratio_vs_v7_pp'])} "
              f"| {number(r0['delta_r2_pp'])}/{number(r1['delta_r2_pp'])} "
              f"| {row['above_target_both_point']}/{row['above_target_both_adjusted']}")
    best = summary["exploratory_ranking_by_minimum"][:1]
    for name in dict.fromkeys(["nested_maximin", *best]):
        for s in summary["routes"][name]["repeats"]:
            print(f"  r{s['repeat']} {name} | off/on {s['off_r2']:.5f}/{s['on_r2']:.5f} "
                  f"| ratio CI% {ci(s['ratio_ci95_pct'])} | vs V7 {number(s['ratio_vs_v7_pp'])} "
                  f"pp CI {ci(s['ratio_vs_v7_ci95_pp'])} | off/on change vs V7 "
                  f"{number(s['off_r2_vs_v7_pp'])}/{number(s['on_r2_vs_v7_pp'])} R2 pp | {s['ratio_status']}")
    print(f"EXP2_V8_BOTH | point={len(summary['both_above_point'])} "
          f"adjusted={len(summary['both_above_adjusted'])} / {summary['screened_route_count']} routes "
          f"| best stable family={summary['best_stable_family'] or 'none'}")
    print("EXP2_V8_SCREEN_END | same discovery households across repeats", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    release = release_check()
    if args.report:
        compact(completed(RUN, release))
        return
    store.environment()
    from adapters.person_inputs import load_discovery
    from v7_source import wrap_source
    from threadpoolctl import threadpool_limits
    import sklearn.decomposition  # Load the native pools used by the archived adapters before capping threads.
    config = protocol()  # Import numerical libraries before limiting thread pools.
    if config != store.read_json(ROOT / "design.json"):
        raise ValueError("design.json does not match executable V8 protocol")
    emit("DESIGN", dict(families=config["family_count"], settings=config["settings_count"],
                        reported=len(config["routes"]), target=config["target_pct"], threads=args.threads))
    with threadpool_limits(limits=args.threads):
        if args.preflight:
            suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_v8*.py")
            with open(os.devnull, "w") as muted, contextlib.redirect_stdout(muted):
                outcome = unittest.TextTestRunner(verbosity=1).run(suite)
            if not outcome.wasSuccessful() or not outcome.testsRun or outcome.skipped:
                raise RuntimeError("V8 generated preflight failed")
            emit("PREFLIGHT_DONE", dict(tests=outcome.testsRun, scope="generated only"))
        else:
            source = wrap_source(load_discovery(ROOT))
            emit("SOURCE_READY", store.cohort_identity(source))
            compact(execute(source, RUN, release, args.threads, draws=config["bootstrap_draws"],
                            prior_run=run_v7.RUN))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("STOP", {"error": type(error).__name__, "detail": str(error)})
        raise
