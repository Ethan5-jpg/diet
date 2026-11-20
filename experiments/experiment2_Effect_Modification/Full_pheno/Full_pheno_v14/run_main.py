"""Audit persistent losses and test alternative main effects in the expanded matrix."""

from pathlib import Path
from statistics import median
import sys


HERE=Path(__file__).resolve().parent
sys.path[1:1]=[str(HERE),str(HERE.parent/"Full_pheno_v13")]
from run_adaptive import PREVIOUS, REFERENCE, check_replay, context, emit, parse, register, require, v1_run
from taxonomy import end_group, map_diets, summarize


V13=HERE.parent/"Full_pheno_v13/runs/TRE_ONLY/adaptive-1"
FRESH=(("fresh-1",20262005),("fresh-2",20262043))


def summarize_fits(results,classes,names,prior):
    from batch_fit import stable_combinations
    from diagnose import shared_rows

    summary=summarize(results,classes,{m:m for m in names})
    stable=stable_combinations(results)["stable"]
    older={z["endpoint"] for z in prior["stable"] if z["diet"]=="fixed_half" and z["min_gain"]>=.1 and z["min_total"]>0.}
    current={z["endpoint"] for z in stable if z["diet"]=="reference:half" and z["min_gain"]>=.1 and z["min_total"]>0.}
    for name,stats in summary["categories"].items():
        here={z["endpoint"] for z in stable if z["diet"]==name and z["min_gain"]>=.1 and z["min_total"]>0.}
        source=shared_rows(results,name)
        stats["prior_half_joint_retained"]=len(here&older)
        stats["matched_reference_half_joint_retained"]=len(here&current)
        stats["better_than_matched_reference_both"]=sum(all(x["vs_reference_pp"]>1e-10 for x in pair) for pair in source.values())
        stats["large_total_loss_both"]=sum(all(x["diet_total_pp"]<=-.1 for x in pair) for pair in source.values())
        stats["split_total_negative"]=[sum(r[i]["diet_total_pp"]<0. for r in source.values()) for i in range(len(results))]
        stats["median_paired_vs_reference"]=[median(r[i]["vs_reference_pp"] for r in source.values()) for i in range(len(results))]
    return dict(**summary,stable=stable,prior_half_joint_count=len(older),reference_half_joint_count=len(current))


def report(out,summary):
    diag=summary["diagnostics"]
    lines=["# V14 expanded diet baseline audit and representation comparison","",
           "Alternative main-effect representations are confined to the expanded matrix. The original 18 pair models keep their archived diet predictions.",
           "The no-diet arm and held households match across all recipes. For each main-effect recipe the interaction is off, fixed at the V12 multiplier, or half that multiplier.",
           "The interaction uses the same diet-by-gut design with residuals of the indicated main-effect model.",
           "Both new partitions reassign discovery households. Confirmation outcomes stay sealed.","",
           "## Outcome losses under the V13 whole-diet main model","",
           "| Outcome domain | Common outcomes | Main-effect losses in both partitions |",
           "|---|---:|---:|"]
    for group,x in diag["previous_full_by_domain"].items():
        lines.append(f"| {group} | {x['outcomes']} | {x['main_negative']} |")
    lines.extend(["","## Held-out diary coverage and split-half agreement",""])
    for group,x in diag["diary_strata"].items():
        lines.append(f"- {group} | {x['all']['outcomes']} common outcomes | {x['all']['main_negative']} persistently negative main effects")
    lines.extend(["", "Diary strata are observational subgroups. The complete group and family-coverage diagnostics are in summary.json.", ""])
    for key,x in diag["food_reliability"].items():
        lines.append(f"- {key} | {x['people_ge4']} people | split-half correlation {x['split_half_r']}")
    lines.extend(["","## Matched expanded model comparisons","",
                  "| Method | Common outcomes | Both main negative | Both total negative | Both total ≤−0.10 | Joint | Prior half joint retained |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    cols=("pairs","main_negative","total_negative","large_total_loss_both","joint","prior_half_joint_retained")
    for k,row in summary["categories"].items():
        lines.append("| "+k+" | "+" | ".join(str(row[a]) for a in cols)+" |")
    lines.extend(["","Joint gains require +0.10 R² percentage points for the interaction, at least three positive folds per partition, and a positive total diet increment on both partitions.",""])
    v1_run.atomic_write(out/"stage_record_v14.md","\n".join(lines).encode())


def main():
    args=parse(__doc__)
    runtime=v1_run.init_runtime(1)
    import numpy as np
    from batch_sources import paths, read_diets, read_outcomes
    from diagnose import adjudicate_history, diary_comparison
    from food_features import read_features
    from interaction_fit import Repartition
    from main_fit import METHODS, collect, diary_scores, score_partition

    out=HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snapshot,study,calibration,sites,cal_files=context()
        v11=v1_run.read_json(REFERENCE/"registration.json")
        inv=v1_run.read_json(REFERENCE/"inventory.json")
        v12=v1_run.read_json(PREVIOUS/"summary.json")
        v13=v1_run.read_json(V13/"summary.json")
        reg13=v1_run.read_json(V13/"registration.json")
        require(v11["snapshot"]==snapshot.manifest_sha and not (PREVIOUS/"RUNNING.lock").exists()
                and not (V13/"RUNNING.lock").exists() and
                v12["reserved_outcomes_read"] is v13["reserved_outcomes_read"] is False,"V14_PREVIOUS_SCOPE")
        require(not {seed for _,seed in FRESH}&{seed for _,seed in (*v11["seeds"],*reg13["seeds"])},
                "V14_REPEAT_ASSESSMENT_SEED")
        old_names=("archive","fresh-1","fresh-2")
        old12=[PREVIOUS/f"{n}.json" for n in old_names]
        old13=[V13/f"{n}.json" for n in old_names[1:]]
        sources={Path(k):v for k,v in v11["sources"].items()}
        require(all(v1_run.file_sha(f)==v for f,v in sources.items()),"V14_SOURCE_CHANGED")
        outcome_paths=paths(root)
        require(set((*outcome_paths.values(),*cal_files))<=sources.keys(),"V14_UNREGISTERED_OUTCOMES")
        outcomes,metadata=read_outcomes(outcome_paths,study)
        diets,_,food_paths=read_diets(root,study)
        require(set(food_paths.values())<=sources.keys(),"V14_UNREGISTERED_DIET")
        endpoints=inv["endpoint_ids"]
        require(list(outcomes)==endpoints and [x[0] for x in diets]==inv["diet_ids"],"V14_CATALOG_CHANGED")
        map_diets(inv["diet_ids"])
        domain={e:end_group(e,metadata["domains"][e]) for e in endpoints}
        require(domain==v12["endpoint_groups"],"V14_OUTCOME_DOMAIN_CHANGED")
        proof={str(f):v1_run.file_sha(f) for f in (*old12,*old13,PREVIOUS/"summary.json",V13/"summary.json",V13/"registration.json")}
        register(out,dict(commit=args.commit,runtime=runtime,seeds=[list(x) for x in FRESH], methods=list(METHODS),
                          prior=proof,code={p.name:v1_run.file_sha(p) for p in sorted(HERE.glob("*.py"))}))
        older=adjudicate_history([v1_run.read_json(f) for f in old12],[v1_run.read_json(f) for f in old13],domain)
        diagnostics,real_paths=read_features(root,study)
        require(set(real_paths.values())<=sources.keys(),"V14_DIAGNOSTIC_SOURCE")
        older["food_reliability"]=diagnostics.reliability
        older["food_day_qc"]=diagnostics.flow
        v1_run.atomic_write(out/"diagnostics.json",v1_run.encode(older))
        emit("V14_DIAGNOSIS",dict(outcomes=older["common_outcomes"],
                                  main_losses=sum(x["main_negative"] for x in older["previous_full_by_domain"].values())))
        values=np.column_stack([outcomes[e] for e in endpoints])
        replay_path=out/"archive_replay.json"
        if replay_path.exists():
            replay=v1_run.read_json(replay_path)
        else:
            m,_,names=collect(study,diets,values,lambda fold,n:emit("V14_ARCHIVE_FOLD",{"fold":fold,"outcomes":n}),
                              recipes=("reference",),strata=False)
            old=[z for z in score_partition(m,endpoints,names) if z["diet"]=="reference:fixed"]
            error=check_replay(old,v1_run.read_json(old12[0]))
            replay=dict(outcomes=len(old),maximum_score_difference=error)
            v1_run.atomic_write(replay_path,v1_run.encode(replay))
        require(replay["maximum_score_difference"]<1e-8,"V14_REPLAY_MISMATCH")
        emit("V14_REPLAY",replay)
        results=[]
        for name,seed in FRESH:
            path=out/f"{name}.json"
            if path.exists():
                p=v1_run.read_json(path)
            else:
                work=Repartition(study,sites,calibration,seed,emit)
                progress=lambda fold,n:emit("V14_OUTER_FOLD",dict(seed=seed,fold=fold,scored=n))
                m,diary,names=collect(work,diets,values,progress)
                require(names==METHODS,"V14_RECIPE_ORDER")
                p=dict(seed=seed,rows=score_partition(m,endpoints,names),diary=diary_scores(diary,endpoints))
                v1_run.atomic_write(path,v1_run.encode(p))
            require(p["seed"]==seed and len(p["rows"])>=1400*len(METHODS),"V14_PARTITION")
            results.append(p)
            emit("V14_PARTITION",dict(seed=seed,rows=len(p["rows"])))
        require(set(snapshot.paths)<=snapshot.verified,"V14_SNAPSHOT_NOT_VERIFIED")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(f)==v for f,v in sources.items()) and
                all(v1_run.file_sha(Path(f))==v for f,v in proof.items()),"V14_INPUT_CHANGED_DURING_FIT")
        older["diary_strata"]=diary_comparison(results,domain)
        spec=summarize_fits(results,domain,METHODS,v13)
        summary=dict(commit=args.commit,archive_replay=replay,diagnostics=older,methods=list(METHODS),
                     **spec,reserved_outcomes_read=False)
        report(out,summary)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(summary))
    emit("V14_COMPLETE",dict(pairs=summary["shared_pairs"],methods=len(METHODS)))


if __name__=="__main__":
    main()
