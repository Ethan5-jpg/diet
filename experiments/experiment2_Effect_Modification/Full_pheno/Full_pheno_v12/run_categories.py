"""Assess whole-diet and grouped interactions on an aligned quantitative panel."""

from pathlib import Path

from review_grid import HERE, REFERENCE, emit, parse, register, require, v1_run
from run_locked import context
from taxonomy import DIET_CLASS, end_group, summarize


RED_OUTCOMES = ("cgm_indices:cgm_sd_roc", "cgm_indices:cgm_sdwsh")


def compare_animals(parts):
    methods=("animal_foods_given_whole_diet","animal_without_red_meat_given_whole_diet")
    return {e:{m:[next(r["interaction_pp"] for r in p["rows"] if r["diet"]==m and r["endpoint"]==e)
                   for p in parts] for m in methods} for e in RED_OUTCOMES}


def report(out, summary):
    lines=["# V12 whole diet and food categories", "",
           "These models cover the registered discovery outcome panel. The original 18 models retain their V10 diet predictions.",
           "All new recipes use matched observations within each outcome and fold. Conditional recipes share the entire diet main prediction.",
           "Repartitioning uses the same discovery cohort and the same archived and two refreshed seeds as V11.",
           "",
           "| Recipe | Shared outcomes | Positive diet main | Negative diet main | Positive interaction | Interaction ≥+0.10 | Interaction ≤−0.10 | Positive diet total | Negative diet total | Joint | Holm |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names=("pairs","main_positive","main_negative","interaction_positive","interaction_reached",
           "interaction_loss","total_positive","total_negative","joint","adjusted")
    for method,cat in summary["categories"].items():
        lines.append("| "+method+" | "+" | ".join(str(cat[k]) for k in names)+" |")
    lines.extend(["","## Whole diet by outcome domain",""])
    for name,x in summary["by_outcome_domain"]["whole_diet"].items():
        lines.append(f"- {name} {x['pairs']} outcomes, {x['main_positive']} positive diet main, "
                     f"{x['joint']} joint gains, {x['interaction_loss']} recurring losses.")
    lines.extend(["","## Animal-food sensitivity on the two V11 CGM outcomes",""])
    for e,scores in summary["animal_sensitivity"].items():
        for r,gains in scores.items():
            lines.append(f"- {e} | {r} | "+" / ".join(f"{g:+.4f}" for g in gains))
    lines.extend(["","## Multiplicity-adjusted recurring interactions",""])
    for r in summary["adjusted"]:
        lines.append(f"- {r['diet']} → {r['endpoint']} | "+" / ".join(f"{v:+.4f}" for v in r["interaction_pp"]))
    if not summary["adjusted"]:
        lines.append("No category–outcome interaction passed the full-grid Holm adjustment in all three partitions.")
    lines.extend(["","Full category classifications, fit audits, and stable pairs are in the JSON outputs.",""])
    v1_run.atomic_write(out/"stage_record_v12.md","\n".join(lines).encode())


def main():
    args=parse(__doc__)
    runtime=v1_run.init_runtime(1)
    import numpy as np
    from batch_fit import stable_combinations
    from batch_sources import paths, read_diets, read_outcomes
    from category_fit import METHODS, accumulate, score_partition
    from interaction_fit import Repartition
    from run_batch import PARTITIONS

    out=HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snapshot,study,calibration,sites,cal_files=context()
        old=v1_run.read_json(REFERENCE/"registration.json")
        inv=v1_run.read_json(REFERENCE/"inventory.json")
        v11_summary=v1_run.read_json(REFERENCE/"summary.json")
        require(v11_summary["reserved_outcomes_read"] is False and
                not (REFERENCE/"RUNNING.lock").exists(),"V12_V11_NOT_COMPLETE")
        require(old["snapshot"]==snapshot.manifest_sha and old["seeds"]==[list(x) for x in PARTITIONS],
                "V12_EXPERIMENT_BASE_MISMATCH")
        require(all(v1_run.file_sha(Path(k))==v for k,v in old["sources"].items()),"V12_SOURCE_CHANGED")
        outcome_files=paths(root)
        require(set(map(str,(*outcome_files.values(),*cal_files)))<=old["sources"].keys(),"V12_SOURCE_NOT_IN_V11")
        outcome,metadata=read_outcomes(outcome_files,study)
        diets,_,inputs=read_diets(root,study)
        require(set(map(str,inputs.values()))<=old["sources"].keys(),"V12_DIET_SOURCE_NOT_IN_V11")
        endpoints=inv["endpoint_ids"]
        require(list(outcome)==endpoints and [d[0] for d in diets]==inv["diet_ids"],"V12_FEATURE_REPLAY")
        require(set(inv["diet_ids"])==DIET_CLASS.keys(),"V12_CLASSIFICATION_COVERAGE")
        require(all(metadata["domains"][e]==inv["source_audit"]["domains"][e] for e in endpoints),
                "V12_OUTCOME_CLASSIFICATION_REPLAY")
        ref_names=("registration","inventory","summary")
        proof={name:v1_run.file_sha(REFERENCE/f"{name}.json") for name in ref_names}
        register(out,dict(commit=args.commit,runtime=runtime,v11=proof,methods=list(METHODS),seeds=old["seeds"],
                          code={f.name:v1_run.file_sha(f) for f in sorted(HERE.glob("*.py"))}))
        values=np.column_stack([outcome[e] for e in endpoints])
        classes={e:end_group(e,metadata["domains"][e]) for e in endpoints}
        emit("V12_GROUP_START",dict(endpoints=len(endpoints),diets=len(diets),recipes=len(METHODS)))
        results=[]
        for name,seed in PARTITIONS:
            path=out/f"{name}.json"
            if path.exists():
                scored=v1_run.read_json(path)
            else:
                work=study if seed is None else Repartition(study,sites,calibration,seed,emit)
                progress=lambda fold,ne:emit("V12_GROUP_FOLD",dict(partition=name,fold=fold,endpoints=ne))
                m,audit=accumulate(work,diets,values,progress)
                rows,eligible=score_partition(m,endpoints)
                scored=dict(seed=seed,rows=rows,eligible=eligible,audit=audit)
                v1_run.atomic_write(path,v1_run.encode(scored))
            require(scored["seed"]==seed and len(scored["rows"])==scored["eligible"]*len(METHODS),
                    "V12_GROUP_CHECKPOINT")
            emit("V12_GROUP_PARTITION",dict(name=name,outcomes=scored["eligible"],rows=len(scored["rows"])))
            results.append(scored)
        require(set(snapshot.paths)<=snapshot.verified,"V12_SOURCE_VERIFICATION")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(k))==v for k,v in old["sources"].items()),"V12_SOURCE_CHANGED_DURING_RUN")
        require(all(v1_run.file_sha(REFERENCE/f"{k}.json")==v for k,v in proof.items()),"V12_REFERENCE_CHANGED")
        summaries=summarize(results,classes,{m:m for m in METHODS})
        stable=stable_combinations(results)
        adjusted=[r for r in stable["stable"] if max(r["p_holm"])<.05]
        summary=dict(commit=args.commit,methods=list(METHODS),**summaries,
                     endpoint_groups=classes,animal_sensitivity=compare_animals(results),
                     adjusted=adjusted,stable=stable["stable"],reserved_outcomes_read=False)
        report(out,summary)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(summary))
    emit("V12_GROUP_COMPLETE",dict(outcomes=min(r["eligible"] for r in results),pairs=summary["shared_pairs"],
                                   corrected=len(summary["adjusted"])))


if __name__=="__main__":
    main()
