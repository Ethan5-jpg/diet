"""Compare fixed and training-only calibrated category interactions on new folds."""

from pathlib import Path
from statistics import median
import sys


HERE=Path(__file__).resolve().parent
sys.path[1:1]=[str(HERE),str(HERE.parent/"Full_pheno_v12")]
from review_grid import REFERENCE, emit, parse, register, require, v1_run
from run_locked import context
from taxonomy import end_group, map_diets, summarize


PREVIOUS=HERE.parent/"Full_pheno_v12/runs/TRE_ONLY/categories-1"
FRESH=(("fresh-1",20261901),("fresh-2",20261937))
METRICS=("diet_main_pp","interaction_pp","diet_total_pp")


def check_replay(rows, archive):
    old={r["endpoint"]:r for r in archive["rows"] if r["diet"]=="whole_diet"}
    require(set(old)=={r["endpoint"] for r in rows},"V13_ARCHIVE_OUTCOME_MISMATCH")
    require(all((r["people"],r["families"])==(old[r["endpoint"]]["people"],old[r["endpoint"]]["families"])
                for r in rows),"V13_ARCHIVE_HOUSEHOLDS_MISMATCH")
    difference=max(abs(r[k]-old[r["endpoint"]][k]) for r in rows for k in METRICS)
    require(difference<1e-8,"V13_REFERENCE_SCORE_CHANGED")
    return difference


def aggregate(parts, classes, methods, previous):
    from batch_fit import stable_combinations
    by_class=summarize(parts,classes,{m:m for m in methods})
    stable=stable_combinations(parts)
    prior={x["endpoint"] for x in previous["stable"] if x["diet"]=="whole_diet" and
           x["min_gain"]>=.1 and x["min_total"]>0.}
    for arm,info in by_class["categories"].items():
        repeats=[r for r in stable["stable"] if r["diet"]==arm and r["min_gain"]>=.1 and r["min_total"]>0.]
        info["prior_joint_retained"]=sum(r["endpoint"] in prior for r in repeats)
        info["versus_fixed_median_by_split"]=[median(r["vs_fixed_pp"] for r in p["rows"] if r["diet"]==arm)
                                              for p in parts]
        info["one_split_large_loss_counts"]=[sum(r["interaction_pp"]<=-.1 for r in p["rows"] if r["diet"]==arm)
                                              for p in parts]
        info["one_split_joint_counts"]=[sum(r["interaction_pp"]>=.1 and r["diet_total_pp"]>0. and
                                           r["positive_folds"]>=3 for r in p["rows"] if r["diet"]==arm) for p in parts]
    return dict(**by_class,stable=stable["stable"],prior_joint_count=len(prior))


def write_report(out, summary):
    keys=("pairs","interaction_positive","interaction_reached","interaction_loss","joint",
          "total_positive","total_negative","prior_joint_retained","adjusted")
    lines=["# V13 training-household calibration of diet category interactions","",
           "The reference main predictions replay the V12 archive. Every recipe uses the same no-diet and whole-diet main predictions and identical held-out households. The original 18 models remain fixed.",
           "Two inner validation folds inside each outer training set select nonnegative updates capped at the original magnitude. Smaller weights or shutting off the update are allowed. Sources receive equal weight within each outcome domain.",
           "Both new outer partitions reassign the same discovery cohort. Confirmation outcomes stay sealed.","",
           "| Method | Shared outcomes | Positive interactions | At least +0.10 | At most −0.10 | Joint gains | Positive diet total | Negative diet total | Prior joint retained | Holm |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m,s in summary["categories"].items():
        lines.append("| "+m+" | "+" | ".join(str(s[k]) for k in keys)+" |")
    lines.extend(["","Joint gains require +0.10 R² percentage points, positive diet total, and at least three positive folds in each outer partition.",
                  "Reference comparisons use paired differences in R² on the same held-out predictions. The Holm test evaluates gain from adding the interaction.","",
                  "## Diet category selections by outcome domain",""])
    for seed, by_group in summary["category_choices"].items():
        for name,selection in by_group.items():
            lines.append(f"- {seed} | {name} | {selection}")
    lines.extend(["","Positive and negative outcome-domain results and all retained pairs are in the summary.json file.",""])
    v1_run.atomic_write(out/"stage_record_v13.md","\n".join(lines).encode())


def main():
    args=parse(__doc__)
    runtime=v1_run.init_runtime(1)
    import numpy as np
    import protocol as p
    from adapt_interactions import METHODS, collect, evaluate, score_partition, selection_counts
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    from interaction_fit import Repartition

    out=HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snapshot,study,calibration,sites,cal_files=context()
        v11=v1_run.read_json(REFERENCE/"registration.json")
        inv=v1_run.read_json(REFERENCE/"inventory.json")
        prior=v1_run.read_json(PREVIOUS/"summary.json")
        require(v11["snapshot"]==snapshot.manifest_sha and prior["reserved_outcomes_read"] is False
                and not (PREVIOUS/"RUNNING.lock").exists(),"V13_INPUT_REFERENCE_MISMATCH")
        require(not {seed for _,seed in FRESH}&{seed for _,seed in v11["seeds"]},"V13_REUSE_SEED")
        require(all(v1_run.file_sha(Path(k))==v for k,v in v11["sources"].items()),"V13_SOURCE_CHANGED")
        outcome_paths=paths(root)
        require(set(map(str,(*outcome_paths.values(),*cal_files)))<=v11["sources"].keys(),"V13_UNREGISTERED_SOURCE")
        outcome, metadata=read_outcomes(outcome_paths,study)
        diets,_,diet_paths=read_diets(root,study)
        require(set(map(str,diet_paths.values()))<=v11["sources"].keys(),"V13_UNREGISTERED_DIET")
        endpoints=inv["endpoint_ids"]
        require(list(outcome)==endpoints and [d[0] for d in diets]==inv["diet_ids"],"V13_FEATURE_CATALOG_CHANGED")
        map_diets(inv["diet_ids"])
        domain={x:end_group(x,metadata["domains"][x]) for x in endpoints}
        require(domain==prior["endpoint_groups"],"V13_DOMAIN_CLASSIFICATION_CHANGED")
        deps=(PREVIOUS/"summary.json", PREVIOUS/"archive.json", PREVIOUS/"registration.json", REFERENCE/"registration.json")
        proof={str(x):v1_run.file_sha(x) for x in deps}
        register(out,dict(commit=args.commit,runtime=runtime,reference=proof,seeds=[list(x) for x in FRESH],
                          methods=list(METHODS),code={f.name:v1_run.file_sha(f) for f in sorted(HERE.glob("*.py"))}))
        values=np.column_stack([outcome[x] for x in endpoints])
        d=np.array([domain[x] for x in endpoints])
        y_source=np.array([metadata["domains"][x] for x in endpoints])
        archive_path=out/"archive_replay.json"
        if archive_path.exists():
            replay=v1_run.read_json(archive_path)
        else:
            groups=outcome_groups(values)
            old_fits=[]
            for fold in p.FOLDS:
                old_fits.append(collect(study,fold,diets,values,groups))
                emit("V13_ARCHIVE_FOLD",{"fold":fold})
            old_rows=score_partition(np.stack(old_fits,axis=2),endpoints,("fixed_whole",))
            gap=check_replay(old_rows,v1_run.read_json(PREVIOUS/"archive.json"))
            replay=dict(outcomes=len(old_rows),score_max_difference=gap)
            v1_run.atomic_write(archive_path,v1_run.encode(replay))
        require(replay["outcomes"]>1400 and replay["score_max_difference"]<1e-8,"V13_REPLAY_FAILED")
        emit("V13_ARCHIVE_REPLAY",replay)
        results,choices=[],{}
        for name,seed in FRESH:
            path=out/f"{name}.json"
            if path.exists():
                part=v1_run.read_json(path)
            else:
                work=Repartition(study,sites,calibration,seed,emit)
                scores,plan=evaluate(work,sites,calibration,diets,values,d,y_source,seed,emit)
                rows=score_partition(scores,endpoints,METHODS)
                part=dict(seed=seed,rows=rows,policy=plan)
                v1_run.atomic_write(path,v1_run.encode(part))
            require(part["seed"]==seed and len(part["rows"])>1400*len(METHODS),"V13_PARTITION_CHECKPOINT")
            results.append(part)
            choices[str(seed)]=selection_counts(part["policy"])
            emit("V13_PARTITION",dict(seed=seed,scored=len(part["rows"])))
        require(set(snapshot.paths)<=snapshot.verified,"V13_SOURCE_NOT_VERIFIED")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(k))==v for k,v in {**v11["sources"],**proof}.items()),
                "V13_SOURCE_CHANGED_DURING_RUN")
        analysis=aggregate(results,domain,METHODS,prior)
        summary=dict(commit=args.commit,methods=list(METHODS),archive_replay=replay,category_choices=choices,
                     **analysis,reserved_outcomes_read=False)
        write_report(out,summary)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(summary))
    emit("V13_COMPLETE",{"pairs":summary["shared_pairs"],"methods":len(METHODS)})


if __name__=="__main__":
    main()
