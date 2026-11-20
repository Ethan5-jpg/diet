"""Classify every matched V11 individual diet–outcome result."""

from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/"Full_pheno_v11"))
sys.path.insert(1,str(HERE.parent/"Full_pheno_v10"))
from run_locked import emit, parse, register, require, v1_run
from taxonomy import end_group, map_diets, summarize


REFERENCE=HERE.parent/"Full_pheno_v11/runs/TRE_ONLY/batch-1"


def main():
    args=parse(__doc__)
    out=HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        files=[REFERENCE/f"{n}.json" for n in ("inventory","summary","archive","fresh-1","fresh-2")]
        inventory, old, *parts=(v1_run.read_json(path) for path in files)
        require(not (REFERENCE/"RUNNING.lock").exists() and old["reserved_outcomes_read"] is False,
                "V12_ARCHIVE_INCOMPLETE")
        dietary_classes=map_diets(inventory["diet_ids"])
        endpoints=inventory["endpoint_ids"]
        source=inventory["source_audit"]["domains"]
        classes={e:end_group(e,source[e]) for e in endpoints}
        proof={str(p):v1_run.file_sha(p) for p in files}
        register(out,dict(commit=args.commit,reference=proof,code={p.name:v1_run.file_sha(p)
                            for p in (Path(__file__),HERE/"taxonomy.py")}))
        result=summarize(parts,classes,dietary_classes)
        require(result["shared_pairs"]==old["shared"],"V12_GRID_DENOMINATOR")
        require(sum(r["joint"] for r in result["categories"].values())==old["joint_everywhere"],
                "V12_GRID_JOINT_REPLAY")
        require(sum(r["adjusted"] for r in result["categories"].values())==old["holm_everywhere"],
                "V12_GRID_HOLM_REPLAY")
        v1_run.atomic_write(out/"summary.json",v1_run.encode(dict(commit=args.commit,**result,
                                                               reserved_outcomes_read=False)))
    for name,d in result["categories"].items():
        emit("V12_DIET_DETAIL",dict(category=name,**{k:d[k] for k in ("pairs","main_positive","main_negative",
                   "interaction_loss","joint","adjusted","total_negative","domain_balanced_joint_rate")}))
    emit("V12_DIET_DETAIL_COMPLETE",dict(pairs=result["shared_pairs"],categories=len(result["categories"])))


if __name__=="__main__":
    main()
