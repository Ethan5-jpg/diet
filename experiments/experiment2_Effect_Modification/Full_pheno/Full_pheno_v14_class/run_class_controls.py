"""Test dietary representation sensitivity separately within each of seven diet classes."""

from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
sys.path[1:1]=[str(HERE),str(HERE.parent/"Full_pheno_v14")]
from run_main import FRESH, PREVIOUS, REFERENCE, context, emit, parse, register, require, v1_run
from taxonomy import GROUPS, end_group, map_diets, summarize


RECIPES=("reference","ridge_035","one_pc")


def collect(work,diets,values,progress):
    import numpy as np
    import protocol as p
    from batch_fit import clip_outcomes, context_design, fold_moments, match_rows, outcome_groups
    from category_fit import panels
    from main_fit import INTERACTIONS, predict

    names=tuple(f"{m}:{a}" for m in RECIPES for a in INTERACTIONS)
    moments=np.zeros((len(GROUPS),len(names),values.shape[1],len(p.FOLDS),10))
    family=np.unique(work.family,return_inverse=True)[1]
    groups=outcome_groups(values)
    days=work.numeric["n_energy_days"]
    for j,fold in enumerate(p.FOLDS):
        bg=work.backgrounds[fold]
        y=clip_outcomes(values,bg.role)
        support,plain=context_design(work,fold,"UPF")
        diet,_=panels(diets,work,fold,support)
        for observed,cols in groups:
            match=match_rows(support&observed,bg.role,family)
            if match is None:
                continue
            rows,n,codes,counts=match
            require(np.all(work.fold[rows[n:]]==fold) and np.all(work.fold[rows[:n]]!=fold),
                    "V14_CLASS_HELD_MEMBERSHIP")
            target=y[np.ix_(rows,cols)]
            for g,category in enumerate(GROUPS):
                full=diet[category][rows]
                reduced=full[:,:1]*np.sqrt(full.shape[1])
                preds=predict(plain[rows],bg.pcs[rows,p.N_PC:p.N_PC+2],full,reduced,target,days[rows],n,RECIPES)
                for i,m in enumerate(names):
                    moments[g,i,cols,j]=np.column_stack(fold_moments(target[n:],preds[m],codes,counts))
        require(np.all(moments[:,:,:,j,:5]==moments[:1,:1,:,j,:5]),"V14_CLASS_REFERENCE_CHANGED")
        progress(fold)
    require(np.isfinite(moments).all(),"V14_CLASS_NONFINITE_MOMENTS")
    return moments,names


def main():
    args=parse(__doc__)
    runtime=v1_run.init_runtime(1)
    import numpy as np
    from batch_sources import paths, read_diets, read_outcomes
    from interaction_fit import Repartition
    from main_fit import score_partition
    from run_expanded_grid import holm

    out=HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snap,study,cal,sites,cal_paths=context()
        reg=v1_run.read_json(REFERENCE/"registration.json")
        inventory=v1_run.read_json(REFERENCE/"inventory.json")
        previous=v1_run.read_json(PREVIOUS/"summary.json")
        require(previous["reserved_outcomes_read"] is False and reg["snapshot"]==snap.manifest_sha,
                "V14_CLASS_SOURCE_SCOPE")
        expected={Path(k):v for k,v in reg["sources"].items()}
        require(all(v1_run.file_sha(f)==d for f,d in expected.items()),"V14_CLASS_SOURCE_CHANGED")
        y_paths=paths(root)
        require(set((*y_paths.values(),*cal_paths))<=expected.keys(),"V14_CLASS_UNREGISTERED_OUTCOMES")
        y_meta,meta=read_outcomes(y_paths,study)
        diets,_,diet_paths=read_diets(root,study)
        endpoints=inventory["endpoint_ids"]
        require(set(diet_paths.values())<=expected.keys() and list(y_meta)==endpoints and
                [t[0] for t in diets]==inventory["diet_ids"],"V14_CLASS_CATALOG_CHANGED")
        map_diets(inventory["diet_ids"])
        domain={e:end_group(e,meta["domains"][e]) for e in endpoints}
        require(domain==previous["endpoint_groups"],"V14_CLASS_REPLAY_DOMAINS")
        source=HERE.parent/"Full_pheno_v14/main_fit.py"
        register(out,dict(commit=args.commit,runtime=runtime,sources=reg["sources"],
                          code={x.name:v1_run.file_sha(x) for x in (Path(__file__),source)},
                          seeds=[list(x) for x in FRESH],recipes=list(RECIPES),groups=list(GROUPS)))
        y=np.column_stack([y_meta[e] for e in endpoints])
        parts=[]
        for name,seed in FRESH:
            f=out/f"{name}.json"
            if f.exists():
                part=v1_run.read_json(f)
            else:
                work=Repartition(study,sites,cal,seed,emit)
                matrix,names=collect(work,diets,y,lambda fold:emit("V14_CLASS_FOLD",dict(seed=seed,fold=fold)))
                rows=[]
                for group,m in zip(GROUPS,matrix,strict=True):
                    rows.extend(dict(z,diet=group+"/"+z["diet"]) for z in score_partition(m,endpoints,names))
                holm(rows)
                part=dict(seed=seed,rows=rows)
                v1_run.atomic_write(f,v1_run.encode(part))
            require(part["seed"]==seed and len(part["rows"])>=len(GROUPS)*len(RECIPES)*3*1400,
                    "V14_CLASS_CHECKPOINT")
            parts.append(part)
            emit("V14_CLASS_PARTITION",dict(seed=seed,rows=len(part["rows"])))
        require(set(snap.paths)<=snap.verified,"V14_CLASS_SNAPSHOT_UNVERIFIED")
        snap.verify_manifest()
        require(all(v1_run.file_sha(f)==d for f,d in expected.items()),"V14_CLASS_SOURCE_CHANGED_DURING_FIT")
        labels={r["diet"]:r["diet"] for r in parts[0]["rows"]}
        summary=summarize(parts,domain,labels)
        result=dict(commit=args.commit,**summary,reserved_outcomes_read=False)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(result))
    emit("V14_CLASS_COMPLETE",dict(pairs=result["shared_pairs"],groups=len(GROUPS)))


if __name__=="__main__":
    main()
