"""Inventory and score the expanded discovery-only quantitative grid."""

from collections import Counter
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent/"Full_pheno_v10"))
from run_locked import context, emit, parse, register, require, v1_run


PARTITIONS = (("archive",None),("fresh-1",20261811),("fresh-2",20261847))


def report(out, supply, summary):
    lines=["# V11 quantitative phenotype and dietary batch", "",
           "The original 18 diet–phenotype pairs retain the locked V10 models. This separate grid analyzes the discovery cohort and preserves the confirmation partition.",
           "", f"Discovery contains {supply['people']} participants and {supply['families']} households.",
           f"The inventory contains {len(supply['catalog'])} source directories with {sum(len(v) for v in supply['catalog'].values())} top-level Parquet tables.",
           f"The model grid contains {supply['diets']} diet exposures and {supply['endpoints']} outcomes before pairing. It excludes the original 18 pairs.",
           "", "## Linked outcome sources", "",
           "| Source | Rows in source table | Discovery people linked | Numeric outcomes entered |",
           "|---|---:|---:|---:|"]
    for name,d in supply["endpoints_by_source"].items():
        lines.append(f"| {name} | {d['raw_rows']} | {d['linked_people']} | {d['admitted_endpoints']} |")
    lines.extend(["","## Split results",""])
    for entry in summary["partitions"]:
        lines.append(f"- {entry['name']} scored {entry['tested']} pairs. Exclusion counts {entry['excluded']}.")
    lines.extend(["",f"{summary['shared']} pairs qualified across all three household partitions.",
                 f"{summary['positive_everywhere']} had positive interaction gain in all partitions and at least three outer folds per partition.",
                 f"{summary['threshold_everywhere']} also had interaction gain of at least +0.10 R² percentage points in every partition.",
                 f"{summary['joint_everywhere']} met that threshold and had positive total diet increment in every partition.",
                 f"{summary['holm_everywhere']} qualified after the within-partition Holm adjustment in all three partitions.",
                 "", "## Leading pairs with recurring +0.10 and positive diet total", ""])
    joint=[r for r in summary["stable"] if r["min_gain"]>=.1 and r["min_total"]>0.]
    for row in joint[:40]:
        gain=" / ".join(f"{x:+.4f}" for x in row["interaction_pp"])
        total=" / ".join(f"{x:+.4f}" for x in row["diet_total_pp"])
        lines.append(f"- {row['diet']} → {row['endpoint']} | interaction {gain} | diet total {total} | at least {row['min_families']} households")
    lines.extend(["", "The same discovery households are reassigned across partitions. Clinical and omics predictors are observational.",
                  "Complete coverage receipts and all paired results are retained in TRE.", ""])
    v1_run.atomic_write(out/"stage_record_v11.md", "\n".join(lines).encode())


def main():
    args=parse(__doc__)
    runtime=v1_run.init_runtime(1)
    import numpy as np
    import protocol as p
    from batch_fit import accumulate, partition_scores, stable_combinations
    from batch_sources import inventory, paths, read_diets, read_outcomes
    from food_features import SOURCES
    from interaction_fit import Repartition

    out=HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snap,study,calibration,sites,cal_files=context()
        outcome_paths=paths(root)
        diet_paths={(root/loc).resolve(strict=True) for loc in SOURCES.values()}
        locked=HERE.parent/"Full_pheno_v10/runs/TRE_ONLY/locked-1/summary.json"
        require(v1_run.read_json(locked)["reserved_outcomes_read"] is False,"V11_LOCKED_REFERENCE")
        registered={str(f):v1_run.file_sha(f) for f in {*cal_files,*outcome_paths.values(),*diet_paths,locked}}
        register(out, dict(commit=args.commit,runtime=runtime,snapshot=snap.manifest_sha,
                           seeds=[list(x) for x in PARTITIONS], sources=registered,
                           code={f.name:v1_run.file_sha(f) for f in sorted(HERE.glob("*.py"))}))
        catalog=inventory()
        outcome, y_audit=read_outcomes(outcome_paths,study)
        diets,x_audit,feeds=read_diets(root,study)
        require(set(feeds.values())==diet_paths,"V11_UNREGISTERED_FOOD_SOURCE")
        require(set(p.ENDPOINTS)<=outcome.keys() and len(outcome)>47 and len(diets)>45,"V11_EXPANSION_INCOMPLETE")
        endpoints=list(outcome)
        values=np.column_stack(list(outcome.values()))
        domain_counts=Counter(y_audit["domains"][k] for k in endpoints)
        by_source={k:dict(raw_rows=v["raw_rows"],linked_people=v["linked_people"],admitted_endpoints=domain_counts[k])
                   for k,v in y_audit["sources"].items()}
        n_pairs=len(diets)*len(endpoints)-len(p.MODES["all"])
        supply=dict(people=study.n,families=study.role_counts[0][1],catalog=catalog,
                    diets=len(diets),endpoints=len(endpoints),endpoints_by_source=by_source,
                    candidate_pairs=n_pairs,source_audit=y_audit,diet_audit=x_audit,
                    diet_ids=[d[0] for d in diets],endpoint_ids=endpoints)
        v1_run.atomic_write(out/"inventory.json",v1_run.encode(supply))
        emit("V11_INVENTORY",dict(people=study.n,families=study.role_counts[0][1],
                                 sources=len(catalog),tables=sum(len(v) for v in catalog.values()),
                                 diets=len(diets),endpoints=len(endpoints),candidate_pairs=n_pairs))
        for k,d in by_source.items():
            emit("V11_SOURCE",dict(domain=k,**d))
        parts,coverage=[],[]
        for name,seed in PARTITIONS:
            path=out/f"{name}.json"
            if path.exists():
                data=v1_run.read_json(path)
            else:
                work=study if seed is None else Repartition(study,sites,calibration,seed,emit)
                progress=lambda fold,n:emit("V11_BATCH_FOLD",dict(partition=name,fold=fold,scored=n))
                scores,available=accumulate(work,diets,endpoints,values,progress)
                rows,excluded=partition_scores(diets,endpoints,scores,available)
                data=dict(seed=seed,rows=rows,excluded=excluded)
                v1_run.atomic_write(path,v1_run.encode(data))
            require(data["seed"]==seed and len(data["rows"])+sum(data["excluded"].values())==n_pairs,
                    "V11_CHECKPOINT_COUNT")
            parts.append(data)
            coverage.append(dict(name=name,tested=len(data["rows"]),excluded=data["excluded"]))
            emit("V11_BATCH_PARTITION",coverage[-1])
        require(set(snap.paths)<=snap.verified,"V11_SNAPSHOT_VERIFICATION")
        snap.verify_manifest()
        require(all(v1_run.file_sha(Path(k))==v for k,v in registered.items()),"V11_SOURCE_CHANGE")
        stable=stable_combinations(parts)
        summary=dict(commit=args.commit,candidate_pairs=n_pairs,partitions=coverage,**stable,reserved_outcomes_read=False)
        report(out,supply,summary)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(summary))
    emit("V11_BATCH_COMPLETE",{k:v for k,v in summary.items() if k not in ("stable","partitions")})


if __name__=="__main__":
    main()
