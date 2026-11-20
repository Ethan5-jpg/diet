"""Evaluate the once-frozen 1.37 modifier on two further discovery partitions.

The strength is the rounded center of the common feasibility interval on the
three earlier partitions. Earlier held outcomes informed that choice. Neither
of the two new partition outputs informs the recipe or the other partition.
"""

import argparse
from pathlib import Path

import run_precision as previous
from run_interaction import ANCHOR, coverage_summary, emit, score, v1_run
from run_v2 import directory_id, require
from scale_precision import rescale


FIXED = {"pc5_half": ("pc5_half", 1.), "precision_pc2_x1.37": ("host_clip_dayvariance_pc2", 1.37)}
SEEDS = (20261223, 20261267)


def finish_report(result):
    for row in result["coverage"]:
        emit("V7_FIXED_COVERAGE", row)
    for row in result["rows"]:
        key = row["exposure"], row["endpoint"]
        if (row["method"] != "pc5_half" and key != ANCHOR
                and (row["interaction_pp"] >= .1 or key in (("fiber_density", "D_Liver_viscosity"), ("UPF", "D_TotalFat")))):
            labels = ("exposure", "endpoint", "n", "interaction_pp", "interaction_pp_95", "diet_total_pp")
            emit("V7_FIXED_PAIR", {k:row[k] for k in labels})
    emit("V7_FIXED_COMPLETE", dict(seed=result["partition_seed"], rows=len(result["rows"]),
                                  positive_total_pairs=result["positive_total_pairs"], reserved_outcomes_read=False))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--run-id", type=directory_id, required=True)
    cli.add_argument("--commit")
    cli.add_argument("--report", action="store_true")
    args = cli.parse_args()
    home = previous.HERE / "runs/TRE_ONLY"
    out = home / args.run_id
    summary = out / "summary.json"
    if args.report:
        cached = v1_run.read_json(summary)
        require(cached["partition_seed"]==args.seed, "V7_FIXED_REPORT_SEED")
        finish_report(cached)
        return
    require(args.commit and len(args.commit)==40 and all(c in "0123456789abcdef" for c in args.commit), "V7_FIXED_COMMIT")
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from interaction_fit import Repartition
    from source import Snapshot, Study, required_files
    from upf_refit import load
    from v3_inputs import read_sites

    root = previous.HERE.parents[2]
    snapshot = Snapshot(root, required_files(p.MODES["all"]))
    origin = home / "precision-1/registration.json"
    base_reg = v1_run.read_json(origin)
    discovery = [home / name / "summary.json" for name in ("scale-1", "scale-partition-1", "scale-partition-2")]
    require(base_reg["snapshot"]==snapshot.manifest_sha, "V7_FIXED_SNAPSHOT_CHANGE")
    for registered in discovery:
        require(v1_run.read_json(registered)["partition_seed"]!=args.seed, "V7_FIXED_SEED_USED_IN_SELECTION")
    with v1_run.run_lock(out):
        require(not summary.exists(), "V7_FIXED_ALREADY_COMPLETE")
        study = Study(snapshot, with_upf=True)
        calibration, paths = load(root, study)
        food = previous.day_sources(root)
        sources = {str(path):v1_run.file_sha(path) for path in (*food, *paths)}
        require(sources==base_reg["source"], "V7_FIXED_FOOD_SOURCE_CHANGED")
        registration = dict(commit=args.commit, seed=args.seed, runtime=runtime, snapshot=snapshot.manifest_sha,
                            sources=sources, discovery={f.parent.name:v1_run.file_sha(f) for f in discovery},
                            methods={name:list(pair) for name,pair in FIXED.items()},
                            code=v1_run.file_sha(Path(__file__)))
        metadata = out / "registration.json"
        if metadata.exists():
            require(v1_run.read_json(metadata)==registration, "V7_FIXED_RUN_REGISTRATION_CHANGED")
        else:
            v1_run.atomic_write(metadata, v1_run.encode(registration))
        emit("V7_FIXED_START", {"seed":args.seed,"run_id":args.run_id})
        work = Repartition(study, read_sites(study), calibration, args.seed, emit)
        days = previous.read_days(food, study, calibration)
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
        fits, _ = previous.fit_all(work, outcomes, days)
        selected = tuple(FIXED)
        rows = score(work, rescale(fits,FIXED), selected, selected, p.BOOTSTRAP_DRAWS)
        coverage = coverage_summary(rows, selected)
        guard = [[r["exposure"],r["endpoint"]] for r in rows if r["method"] != "pc5_half"
                 and (r["exposure"],r["endpoint"]) != ANCHOR and r["interaction_pp"]>=.1 and r["diet_total_pp"]>0.]
        require(set(snapshot.paths)<=snapshot.verified, "V7_FIXED_UNVERIFIED_INPUTS")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(f))==sha for f,sha in sources.items()), "V7_FIXED_FOOD_CHANGED")
        result = dict(rows=rows, coverage=coverage, positive_total_pairs=guard, partition_seed=args.seed,
                      commit=args.commit, reserved_outcomes_read=False)
        v1_run.atomic_write(summary, v1_run.encode(result))
    finish_report(result)


if __name__=="__main__":
    main()
