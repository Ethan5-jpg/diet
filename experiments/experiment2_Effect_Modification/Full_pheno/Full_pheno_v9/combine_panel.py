"""Assess the expanded diet panel combined with the previously fixed negative-pair policy."""

import argparse
from pathlib import Path

from feature_policies import METHODS, adjusted
from run_expanded import HERE, HISTORY, aggregate_history, summarize
from run_interaction import emit, score, v1_run
from run_v2 import directory_id, require


SEEDS = (20261607,20261649)
COMBINED = ("v8_reference","expanded","expanded_half","expanded_stop_diet","expanded_stop_both")


def combine(fits,gates):
    import numpy as np
    from diet_audit import Fit
    from expanded_fit import PROTECTED

    result = {}
    j = METHODS.index("all_main_and_interaction")
    for key,parts in fits.items():
        fixed = []
        history = gates.get(key,"keep")
        for f in parts:
            b,c = f.off[:,j],f.on[:,j]
            if key in PROTECTED:
                options = [(f.off[:,0],f.on[:,0])]*3
            else:
                diet = "diet" if history=="diet" else "keep"
                options = [adjusted(f.no_diet,b,c,diet,.5),adjusted(f.no_diet,b,c,diet,0.),
                           adjusted(f.no_diet,b,c,history,0.)]
            u,v = zip(*options,strict=True)
            fixed.append(Fit(f.rows,f.outcome,f.no_diet,np.column_stack((f.off[:,0],b,*u)),
                             np.column_stack((f.on[:,0],c,*v))))
        result[key] = fixed
    return result


def report(result):
    for partition in result["partitions"]:
        for r in partition["coverage"]:
            emit("V9_COMBINED_ASSESSMENT",dict(seed=partition["seed"],**r))
    emit("V9_COMBINED_COMPLETE",dict(methods=COMBINED,rows=result["row_counts"],reserved_outcomes_read=False))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id",type=directory_id,required=True)
    cli.add_argument("--commit")
    cli.add_argument("--report",action="store_true")
    args = cli.parse_args()
    out = HERE/"runs/TRE_ONLY"/args.run_id
    summary = out/"summary.json"
    if args.report:
        report(v1_run.read_json(summary))
        return
    require(args.commit and len(args.commit)==40 and all(x in "0123456789abcdef" for x in args.commit), "V9_COMBINE_COMMIT")
    runtime = v1_run.init_runtime(2)
    from diet_inventory import read_inventory
    from expanded_fit import run_partition
    import protocol as p
    import run_precision as previous
    from source import Snapshot, Study, required_files
    from upf_refit import load

    root = HERE.parents[2]
    reference = HERE/"runs/TRE_ONLY/expanded-1"
    snapshot = Snapshot(root,required_files(p.MODES["all"]))
    past = v1_run.read_json(reference/"registration.json")
    complete = v1_run.read_json(reference/"summary.json")
    freeze = v1_run.read_json(reference/"frozen.json")
    require(past["snapshot"]==snapshot.manifest_sha and complete["reserved_outcomes_read"] is False
            and "all_main_and_interaction" in freeze["methods"],"V9_COMBINE_BASE_SOURCE")
    require(not set(SEEDS)&{seed for _,seed in past["screen"]+past["assessment"]},"V9_COMBINE_REPEAT_SEED")
    with v1_run.run_lock(out):
        study = Study(snapshot,with_upf=True)
        calibration,_ = load(root,study)
        sources = {name:v1_run.file_sha(Path(name)) for name in past["inputs"]}
        require(sources==past["inputs"],"V9_COMBINE_DIET_SOURCE_CHANGED")
        for name,h in past["previous"].items():
            require(v1_run.file_sha(HISTORY/f"{name}.json")==h,"V9_COMBINE_V8_HISTORY_CHANGED")
        identity = dict(commit=args.commit,seed=list(SEEDS),snapshot=snapshot.manifest_sha,runtime=runtime,sources=sources,
                        v9_frozen_sha256=v1_run.file_sha(reference/"frozen.json"),methods=list(COMBINED),
                        code={f.name:v1_run.file_sha(f) for f in (Path(__file__),HERE/"expanded_fit.py",HERE/"diet_inventory.py")})
        reg = out/"registration.json"
        if reg.exists():
            require(v1_run.read_json(reg)==identity,"V9_COMBINE_REGISTRATION_CHANGE")
        else:
            v1_run.atomic_write(reg,v1_run.encode(identity))
        if summary.exists():
            report(v1_run.read_json(summary))
            return
        gates,history = aggregate_history()
        require(all([list(v) for v in history[k]]==freeze["history"][k] for k in history),"V9_COMBINE_GATE_HISTORY")
        values,_,paths = read_inventory(root,study)
        require(all(str(path) in sources for path in paths.values()),"V9_COMBINE_UNREGISTERED_SOURCE")
        days = previous.read_days(previous.day_sources(root),study,calibration)
        y = {e:study.outcome(e) for e in p.ENDPOINTS}
        output = []
        for seed in SEEDS:
            path = out/f"partition-{seed}.json"
            if path.exists():
                cached = v1_run.read_json(path)
                require(cached["seed"]==seed and len(cached["rows"])==18*len(COMBINED),"V9_COMBINE_CHECKPOINT")
                output.append(cached)
                continue
            emit("V9_COMBINED_START",{"seed":seed})
            fits,_,_ = run_partition(study,y,days,values,calibration,gates,seed,emit,adaptive=False)
            rows = score(study,combine(fits,gates),COMBINED,COMBINED,p.BOOTSTRAP_DRAWS)
            partition = dict(seed=seed,coverage=summarize(rows,COMBINED),rows=rows)
            v1_run.atomic_write(path,v1_run.encode(partition))
            output.append(partition)
        require(set(snapshot.paths)<=snapshot.verified,"V9_COMBINE_INCOMPLETE_INPUTS")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(k))==h for k,h in sources.items()),"V9_COMBINE_INPUT_CHANGE_DURING_RUN")
        result = dict(commit=args.commit,partitions=output,row_counts=[len(r["rows"]) for r in output],
                      scope="discovery-only",reserved_outcomes_read=False)
        v1_run.atomic_write(summary,v1_run.encode(result))
    report(result)


if __name__=="__main__":
    main()
