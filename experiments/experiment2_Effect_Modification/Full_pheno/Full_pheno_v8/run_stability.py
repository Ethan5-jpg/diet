"""Compare candidate rules across partitions, then score a frozen shortlist."""

import argparse
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/"Full_pheno_v7_precision"))
import run_precision as previous
from run_interaction import ANCHOR, archive, coverage_summary, emit, score, v1_run
from run_v2 import directory_id, require
from stability_rules import METHODS


SCREEN = (("archive",None),("prior-1",20261223),("prior-2",20261267))
FRESH = (("fresh-1",20261431),("fresh-2",20261493))
REFERENCE = ("precision_137","pc5_half")


def summarize(rows, selected):
    groups = coverage_summary(rows,selected)
    for c in groups:
        rest = [r for r in rows if r["method"]==c["method"] and (r["exposure"],r["endpoint"])!=ANCHOR]
        c["joint_pairs"] = [[r["exposure"],r["endpoint"]] for r in rest if r["interaction_pp"]>=.1 and r["diet_total_pp"]>0.]
        c["joint_pass"] = c["positive"]>=9 and len(c["joint_pairs"])>=2 and c["harmed"]==0
        c["median_total_pp"] = median(r["diet_total_pp"] for r in rest)
        c["median_vs_precision_total_pp"] = median(r["vs_ref_total_pp"] for r in rest)
    return groups


def compare_partitions(results, names):
    output = []
    for name in names:
        records = [next(c for c in r["coverage"] if c["method"]==name) for r in results]
        common = set.intersection(*(set(tuple(x) for x in c["joint_pairs"]) for c in records))
        output.append(dict(method=name,joint_passes=sum(c["joint_pass"] for c in records),
                           interaction_passes=sum(c["passes"] for c in records),common_joint_pairs=sorted(common),
                           min_joint=min(len(c["joint_pairs"]) for c in records),
                           min_positive=min(c["positive"] for c in records),max_harmed=max(c["harmed"] for c in records),
                           worst_total_median=min(c["median_total_pp"] for c in records),
                           worst_vs_precision_total_median=min(c["median_vs_precision_total_pp"] for c in records),
                           worst_interaction_median=min(c["median_interaction_pp"] for c in records)))
    return sorted(output,key=lambda c:(c["joint_passes"],len(c["common_joint_pairs"]),c["min_joint"],
                                      -c["max_harmed"],c["min_positive"],c["worst_total_median"],
                                      c["worst_interaction_median"]),reverse=True)


def emit_partition(name, result):
    for c in result["coverage"]:
        emit("V8_COVERAGE",dict(partition=name,method=c["method"],pos=c["positive"],hit=c["reached"],
                                harm=c["harmed"],joint=c["joint_pairs"],adjusted=c["adjusted_interaction_positive"],
                                adjusted_incumbent=c["adjusted_total_beats_reference"],
                                negative_total=c["total_negative"],median_interaction_pp=c["median_interaction_pp"]))


def replay(partition, rows):
    base = HERE.parent/"Full_pheno_v7_precision/runs/TRE_ONLY"
    if partition=="archive":
        path = HERE.parent/"Full_pheno_v7/runs/TRE_ONLY/interaction-1/summary.json"
        pair = ("pc5_half","pc5_half")
    else:
        path = base/f"fixed-{partition[-1]}"/"summary.json"
        pair = ("precision_137","precision_pc2_x1.37")
    reference = v1_run.read_json(path)
    old = {(r["exposure"],r["endpoint"]):r for r in reference["rows"] if r["method"]==pair[1]}
    curve = [r for r in rows if r["method"]==pair[0]]
    require(len(old)==len(curve)==18,"V8_REPLAY_PAIR_COUNT")
    errors = []
    for a in curve:
        b = old[a["exposure"],a["endpoint"]]
        require(a["n"]==b["n"] and a["families"]==b["families"],"V8_REPLAY_SUPPORT")
        errors.extend(abs(a[k]-b[k]) for k in ("r2_no_diet","r2_off","r2_on"))
    require(max(errors)<=2e-8,"V8_MISMATCH_V7")
    return {"summary_sha256":v1_run.file_sha(path),"max_r2_error":max(errors)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id",type=directory_id, required=True)
    cli.add_argument("--commit")
    cli.add_argument("--report",action="store_true")
    args = cli.parse_args()
    out = HERE/"runs/TRE_ONLY"/args.run_id
    if args.report:
        print_report(v1_run.read_json(out/"summary.json"))
        return
    require(args.commit and len(args.commit)==40 and all(x in "0123456789abcdef" for x in args.commit),"V8_CODE_COMMIT")
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from source import Snapshot, Study, required_files
    from stability_fit import run_partition
    from upf_refit import load

    root = HERE.parents[2]
    require(root.name==p.HANDOFF,"V8_WRONG_HANDOFF")
    snapshot = Snapshot(root,required_files(p.MODES["all"]))
    release,_ = archive.verify_release()
    v5 = HERE.parent/"Full_pheno_v5/runs/TRE_ONLY/discovery-1"
    archive.read_complete(v5,release)
    reg = v1_run.read_json(v5/"registration.json")
    require(reg["snapshot_sha"]==snapshot.manifest_sha and reg["sources"]==snapshot.report(),"V8_SOURCE_SNAPSHOT_CHANGED")
    with v1_run.run_lock(out):
        study = Study(snapshot,with_upf=True)
        calibration,cal_paths = load(root,study)
        feeds = previous.day_sources(root)
        files = {str(f):v1_run.file_sha(f) for f in (*feeds,*cal_paths)}
        source = HERE.parent/"Full_pheno_v7_precision/runs/TRE_ONLY/precision-1/registration.json"
        require(files==v1_run.read_json(source)["source"],"V8_DAILY_SOURCE_CHANGED")
        external = [HERE.parent/name for name in ("Full_pheno_v7_precision/run_precision.py","Full_pheno_v7/run_interaction.py",
                    "Full_pheno_v7/interaction_fit.py","Full_pheno_v6/diet_audit.py","Full_pheno_v6_exposure/upf_refit.py")]
        identity = dict(commit=args.commit,runtime=runtime,snapshot=snapshot.manifest_sha,inputs=files,
                        code={f.name:v1_run.file_sha(f) for f in (*sorted(HERE.glob("*.py")),*external)},
                        screen=[list(x) for x in SCREEN],fresh=[list(x) for x in FRESH])
        meta = out/"registration.json"
        if meta.exists():
            require(v1_run.read_json(meta)==identity,"V8_REGISTRATION_MISMATCH")
        else:
            v1_run.atomic_write(meta,v1_run.encode(identity))
        if (out/"summary.json").is_file():
            print_report(v1_run.read_json(out/"summary.json"))
            return
        days = previous.read_days(feeds,study,calibration)
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}

        def execute(name,seed,methods,frozen_hash=None):
            path = out/f"{name}.json"
            if path.exists():
                result = v1_run.read_json(path)
                require(result["seed"]==seed and result["methods"]==list(methods) and result["frozen_sha"]==frozen_hash,
                        "V8_PARTITION_CHECKPOINT_MISMATCH")
            else:
                emit("V8_PARTITION_START",dict(name=name,seed=seed,methods=len(methods)))
                fits,accepted = run_partition(study,outcomes,days,calibration,seed,emit,
                                              adaptive=any(x.startswith("inner_") for x in methods))
                rows = score(study,fits,METHODS,methods,p.BOOTSTRAP_DRAWS)
                replay_info = replay(name,rows) if frozen_hash is None else None
                result = dict(seed=seed,methods=list(methods),replay=replay_info,rows=rows,frozen_sha=frozen_hash,
                              coverage=summarize(rows,methods),accepted_inner_updates=accepted)
                v1_run.atomic_write(path,v1_run.encode(result))
            emit_partition(name,result)
            return result

        screen = [execute(n,s,METHODS) for n,s in SCREEN]
        ranked = compare_partitions(screen,METHODS)
        names = [c["method"] for c in ranked if c["method"] not in REFERENCE][:2]
        advance = dict(methods=list(REFERENCE)+names,
                       screen_sha={n:v1_run.file_sha(out/f"{n}.json") for n,_ in SCREEN})
        freeze = out/"frozen.json"
        if freeze.exists():
            require(v1_run.read_json(freeze)==advance,"V8_FROZEN_MISMATCH")
        else:
            v1_run.atomic_write(freeze,v1_run.encode(advance))
        emit("V8_FROZEN",advance["methods"])
        fresh = [execute(n,s,advance["methods"],v1_run.file_sha(freeze)) for n,s in FRESH]
        require(set(snapshot.paths)<=snapshot.verified,"V8_UNVERIFIED_DISCOVERY_INPUTS")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(path))==h for path,h in files.items()),"V8_DAILY_SOURCE_CHANGE_DURING_RUN")
        result = dict(commit=args.commit,run_id=args.run_id,frozen_methods=advance["methods"],
                      screen=ranked,fresh=compare_partitions(fresh,advance["methods"]),
                      row_counts={n:len(r["rows"]) for (n,_),r in zip((*SCREEN,*FRESH),(*screen,*fresh),strict=True)},
                      scope="discovery-only",reserved_outcomes_read=False)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(result))
    print_report(result)


def print_report(result):
    for c in result["fresh"]:
        emit("V8_FRESH_SUMMARY",c)
    emit("V8_COMPLETE",result)


if __name__=="__main__":
    main()
