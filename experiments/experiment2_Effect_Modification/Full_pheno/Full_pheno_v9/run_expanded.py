"""Measure expanded dietary features and abstention on consistently adverse pairs."""

import argparse
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/"Full_pheno_v8"))
import run_stability as previous
from feature_policies import BLOCKS, FIXED, METHODS
from run_interaction import ANCHOR, archive, emit, score, v1_run
from run_v2 import directory_id, require


SCREEN = (("archive",None),("replay-1",20261431),("replay-2",20261493))
NEW = (("assessment-1",20261531),("assessment-2",20261579))
HISTORY = (HERE.parent/"Full_pheno_v8/runs/TRE_ONLY/stability-1").resolve()
REFERENCE = "v8_reference"
DEVICE = ("train_select","train_shrink","train_stop","historical_stop")


def aggregate_history():
    partitions = [v1_run.read_json(HISTORY/f"{name}.json") for name in ("archive","prior-1","prior-2","fresh-1","fresh-2")]
    subsets = {}
    for label in ("diet_main_pp","diet_total_pp","interaction_pp"):
        subsets[label] = set.intersection(*({(r["exposure"],r["endpoint"]) for r in a["rows"]
                                              if r["method"]=="blended_main" and r[label]<0.} for a in partitions))
    gates = {k:"diet" for k in subsets["diet_total_pp"]}
    for k in subsets["interaction_pp"]-gates.keys():
        gates[k] = "interaction"
    return gates, {name:sorted(k) for name,k in subsets.items()}


def summarize(rows,names):
    out = []
    for method in names:
        rs = [r for r in rows if r["method"]==method and (r["exposure"],r["endpoint"])!=ANCHOR]
        require(len(rs)==17,"V9_TASK_DENOMINATOR")
        j = [(r["exposure"],r["endpoint"]) for r in rs if r["interaction_pp"]>=.1 and r["diet_total_pp"]>0.]
        out.append(dict(method=method,positive=sum(r["interaction_pp"]>1e-11 for r in rs),
                        reached=sum(r["interaction_pp"]>=.1 for r in rs),harmed=sum(r["interaction_pp"]<=-.1 for r in rs),
                        negative_total=sum(r["diet_total_pp"]<-1e-11 for r in rs),
                        adjusted_total_over_reference=sum(r["vs_ref_total_pp_95"][0]>0. for r in rs),
                        adjusted_interaction=sum(r["interaction_pp_95"][0]>0. for r in rs),
                        median_total=median(r["diet_total_pp"] for r in rs),
                        median_vs_reference=median(r["vs_ref_total_pp"] for r in rs),joint=sorted(j)))
    return out


def freeze_expanded(results):
    candidates = []
    for name in FIXED[1:]:
        cs = [next(c for c in r["coverage"] if c["method"]==name) for r in results]
        pairs = set.intersection(*(set(map(tuple,c["joint"])) for c in cs))
        rank = (-max(c["negative_total"] for c in cs), len(pairs),
                min(c["median_vs_reference"] for c in cs), min(c["positive"] for c in cs))
        candidates.append((rank,name))
    return max(candidates,key=lambda pair:pair[0])[1]


def replay(name, rows):
    f = HISTORY/(dict(archive="archive", **{"replay-1":"fresh-1","replay-2":"fresh-2"})[name]+".json")
    reference = v1_run.read_json(f)
    old = {(r["exposure"],r["endpoint"]):r for r in reference["rows"] if r["method"]=="blended_main"}
    cs = [r for r in rows if r["method"]==REFERENCE]
    require(len(cs)==len(old)==18,"V9_REPLAY_COUNT")
    errors = []
    for r in cs:
        o = old[r["exposure"],r["endpoint"]]
        require(r["n"]==o["n"] and r["families"]==o["families"],"V9_REPLAY_SUPPORT")
        errors.extend(abs(o[k]-r[k]) for k in ("r2_no_diet","r2_off","r2_on"))
    require(max(errors)<=2e-8,"V9_REPLAY_V8_DIFFERENCE")
    return max(errors)


def print_report(result):
    for v in result["assessment"]:
        for row in v["coverage"]:
            emit("V9_ASSESSMENT",dict(partition=v["partition"],**row))
    emit("V9_COMPLETE",dict(run_id=result["run_id"],methods=result["frozen_methods"],rows=result["row_counts"],
                            scope=result["scope"],reserved_outcomes_read=False))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id",type=directory_id, required=True)
    cli.add_argument("--commit")
    cli.add_argument("--report",action="store_true")
    args = cli.parse_args()
    out = HERE/"runs/TRE_ONLY"/args.run_id
    summary = out/"summary.json"
    if args.report:
        print_report(v1_run.read_json(summary))
        return
    require(args.commit and len(args.commit)==40 and all(x in "0123456789abcdef" for x in args.commit),"V9_COMMIT")
    runtime = v1_run.init_runtime(2)
    from diet_inventory import read_inventory
    from expanded_fit import run_partition
    from food_features import SOURCES
    import protocol as p
    from source import Snapshot, Study, required_files
    from upf_refit import load

    root = HERE.parents[2]
    require(root.name==p.HANDOFF,"V9_ROOT")
    snapshot = Snapshot(root,required_files(p.MODES["all"]))
    release,_ = archive.verify_release()
    old = HERE.parent/"Full_pheno_v5/runs/TRE_ONLY/discovery-1"
    archive.read_complete(old,release)
    receipt = v1_run.read_json(old/"registration.json")
    require(receipt["snapshot_sha"]==snapshot.manifest_sha and receipt["sources"]==snapshot.report(),"V9_V5_SNAPSHOT")
    hreg = v1_run.read_json(HISTORY/"registration.json")
    require(hreg["snapshot"]==snapshot.manifest_sha,"V9_V8_SNAPSHOT")
    with v1_run.run_lock(out):
        if summary.is_file():
            old_result = v1_run.read_json(summary)
            require(old_result["commit"]==args.commit,"V9_COMPLETION_COMMIT")
            print_report(old_result)
            return
        study = Study(snapshot,with_upf=True)
        calibration,food_paths = load(root,study)
        previous_precision = previous.previous
        source_paths = {*food_paths,*previous_precision.day_sources(root),*((root/x).resolve(strict=True) for x in SOURCES.values())}
        hashes = {str(path):v1_run.file_sha(path) for path in sorted(source_paths)}
        gates, historical = aggregate_history()
        prior = {name:v1_run.file_sha(HISTORY/f"{name}.json") for name in ("archive","prior-1","prior-2","fresh-1","fresh-2")}
        code = {x.name:v1_run.file_sha(x) for x in sorted(HERE.glob("*.py"))}
        dependencies = (HERE.parent/"Full_pheno_v7_precision/run_precision.py",HERE.parent/"Full_pheno_v8/run_stability.py",
                        HERE.parent/"Full_pheno_v7/interaction_fit.py",HERE.parent/"Full_pheno_v6_exposure/food_features.py")
        code.update({str(x.relative_to(HERE.parent)):v1_run.file_sha(x) for x in dependencies})
        identity = dict(commit=args.commit,runtime=runtime,snapshot=snapshot.manifest_sha,inputs=hashes,
                        previous=prior,code=code,screen=[list(x) for x in SCREEN],assessment=[list(x) for x in NEW])
        registry = out/"registration.json"
        if registry.exists():
            require(v1_run.read_json(registry)==identity,"V9_REGISTRATION_CHANGE")
        else:
            v1_run.atomic_write(registry,v1_run.encode(identity))
        emit("V9_START",dict(run_id=args.run_id,historical=historical))
        values,inventory,paths = read_inventory(root,study)
        require(set(paths.values())<=source_paths,"V9_UNREGISTERED_INVENTORY_SOURCE")
        emit("V9_INVENTORY",dict(blocks={k:len(inventory[k]) for k in BLOCKS},days=inventory["matched_days"]))
        days = previous_precision.read_days(previous_precision.day_sources(root),study,calibration)
        y = {e:study.outcome(e) for e in p.ENDPOINTS}

        def execute(name,seed,methods,frozen=None):
            path = out/f"{name}.json"
            if path.exists():
                r = v1_run.read_json(path)
                require(r["seed"]==seed and r["methods"]==list(methods) and r["frozen_sha"]==frozen,"V9_CHECKPOINT_CONFLICT")
            else:
                emit("V9_PARTITION_START",dict(name=name,seed=seed,methods=len(methods)))
                fits,support,decisions = run_partition(study,y,days,values,calibration,gates,seed,emit,
                                                       adaptive=any(x.startswith("train_") for x in methods))
                rows = score(study,fits,METHODS,methods,p.BOOTSTRAP_DRAWS)
                r = dict(partition=name,seed=seed,methods=list(methods),frozen_sha=frozen,rows=rows,
                         coverage=summarize(rows,methods),decisions=decisions,support=support,
                         replay=replay(name,rows) if frozen is None else None)
                v1_run.atomic_write(path,v1_run.encode(r))
            emit("V9_PARTITION_END",dict(name=name,rows=len(r["rows"]),reference_replay=r["replay"]))
            return r

        screen = [execute(name,seed,METHODS) for name,seed in SCREEN]
        family = freeze_expanded(screen)
        methods = tuple(dict.fromkeys((REFERENCE,"all_main","all_main_and_interaction",family,*DEVICE)))
        freeze = out/"frozen.json"
        advance = dict(methods=list(methods),selected_feature_rule=family,history=historical,
                       screen_sha={k:v1_run.file_sha(out/f"{k}.json") for k,_ in SCREEN})
        if freeze.exists():
            require(v1_run.read_json(freeze)==advance,"V9_FREEZE_CONFLICT")
        else:
            v1_run.atomic_write(freeze,v1_run.encode(advance))
        emit("V9_FROZEN",dict(methods=methods,feature_rule=family))
        tests = [execute(n,s,methods,v1_run.file_sha(freeze)) for n,s in NEW]
        require(set(snapshot.paths)<=snapshot.verified,"V9_INCOMPLETE_SNAPSHOT")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(path))==digest for path,digest in hashes.items()),"V9_DIET_SOURCE_CHANGE")
        result = dict(run_id=args.run_id,commit=args.commit,scope="discovery-only",reserved_outcomes_read=False,
                      inventory=inventory,historical=historical,frozen_methods=methods,
                      assessment=[dict(partition=r["partition"],coverage=r["coverage"]) for r in tests],
                      row_counts={r["partition"]:len(r["rows"]) for r in (*screen,*tests)})
        v1_run.atomic_write(summary,v1_run.encode(result))
    print_report(result)


if __name__=="__main__":
    main()
