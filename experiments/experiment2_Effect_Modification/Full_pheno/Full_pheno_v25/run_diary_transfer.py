"""Check registered assay interactions across household diary coverage."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/"Full_pheno_v24"))
from freeze_assays import (REFERENCE, V18, V22, V23, context, emit, parse, read_json, register, require,
                           v1_run, write_json)


V24 = HERE.parent/"Full_pheno_v24/runs/TRE_ONLY/assay-freeze-1"
TRANSFER = (("shorter","longer",20263011),("longer","shorter",20263029),
            ("random_a","random_b",20263011),("random_b","random_a",20263029))
TOLERANCE = 1e-8


def fold_records(work, fold, diets, y, targets):
    from coverage_transfer import family_moments
    from frozen_coefficients import design, fitted_on_design

    _, fits, test = fitted_on_design(design(work,fold,diets),work,fold,y,targets,held_out=True)
    return [family_moments(work.family[used], measured, p) for used,measured,p in test], fits


def grouped_replay(study, sites, cal, diets, y, targets, groups, previous, seeds):
    import protocol as p
    from coverage_transfer import summarize_panel
    from interaction_fit import Repartition

    results = []
    for i,seed in enumerate(seeds):
        work = Repartition(study,sites,cal,seed,emit)
        rows = [fold_records(work,f,diets,y,targets)[0] for f in p.FOLDS]
        bits = list(zip(*rows,strict=True))
        blocks = {k:summarize_panel(bits,targets,k,selection,seed*11+i) for k,selection in (
            ("pooled",None),("under_10",groups["under_10"]),("at_least_10",groups["at_least_10"])
        )}
        old = [r["baseline"]["reference"]["parts"][i] for r in previous["frozen_probes"]]
        gap = 0.
        for a,b in zip(blocks["pooled"],old,strict=True):
            require(a["scorable"] is True and a["people"]==b["people"] and a["families"]==b["families"],
                    "V25_REPLAY_ASSAY_SUPPORT_CHANGED")
            difference = max(abs(a[k]-b[k]) for k in ("diet_main_pp","interaction_pp","diet_total_pp","p_holm"))
            require(difference < TOLERANCE,"V25_REPLAY_ASSAY_SCORE_CHANGED")
            gap = max(gap,difference)
        results.append(dict(seed=seed, readouts=blocks, replay_max_difference=gap))
        emit("V25_REPLAY",dict(seed=seed,replay_max_difference=gap))
    return results


def transport(study, sites, cal, diets, y, targets, groups):
    import numpy as np
    from coverage_transfer import summarize_panel, transfer_work

    output = []
    for train,held,seed in TRANSFER:
        use = np.isin(study.family,list(groups[train]))
        evaluate = np.isin(study.family,list(groups[held]))
        work = transfer_work(study,sites,cal,use,evaluate,seed)
        scored, fit = fold_records(work,0,diets,y,targets)
        require(all(set(f) <= groups[held] for f,_ in scored),"V25_WRONG_TRANSFER_FAMILIES")
        rows = summarize_panel([[z] for z in scored],targets,f"{train}_to_{held}",None,seed)
        output.append(dict(train=train,holdout=held,food_seed=seed,fit=fit,results=rows))
        emit("V25_COVERAGE_TRANSFER",dict(train=train,held=held,families=[r["families"] for r in rows]))
    return output


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    import pandas as pd
    import protocol as p
    from batch_sources import paths, read_diets
    from coverage_transfer import diary_coverage
    from freeze_reserved import select_values
    import frozen_coefficients
    from reserved_inputs import KEY
    from sample_audit import CANDIDATES

    out = HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snapshot,study,cal,sites,cal_files = context()
        a18,a23,a24 = (read_json(x/"registration.json") for x in (V18,V23,V24))
        candidate = [list(x) for x in CANDIDATES]
        require(a18["targets"]==a23["frozen_targets"]==a24["targets"]==candidate
                and a24["snapshot"] == snapshot.manifest_sha,"V25_REGISTERED_MODEL_DIFF")
        training_receipt = read_json(V24/"FROZEN.json")
        coefficients = V24/"coefficients_TRE_ONLY.npz"
        require(training_receipt["reserved_outcomes_read"] is False
                and v1_run.file_sha(coefficients)==training_receipt["coefficients_sha256"]
                and v1_run.file_sha(V24/"registration.json")==training_receipt["registration_sha256"]
                and not (V24/"RUNNING.lock").exists(),"V25_FROZEN_FIT_RECEIPT_DIFF")
        module = Path(frozen_coefficients.__file__)
        require(v1_run.file_sha(module)==a24["model_code"][module.name],"V25_V24_FORMULA_CHANGED")
        a11 = read_json(REFERENCE/"registration.json")
        expected = {Path(f):h for f,h in a11["sources"].items()}
        sources = paths(root)
        scope = {k:sources[k] for k in ("nightingale","nightingale_tags","serum_lipidomics")}
        require(set((*scope.values(),*cal_files))<=expected.keys()
                and all(v1_run.file_sha(f)==h for f,h in expected.items()),"V25_UNREGISTERED_SOURCE")
        prior = read_json(V23/"summary.json")
        required = (V18/"registration.json",V23/"registration.json",V23/"summary.json",V24/"registration.json",
                    V24/"FROZEN.json",coefficients,V22/"registration.json",REFERENCE/"inventory.json",module)
        proof = {str(f):v1_run.file_sha(f) for f in required}
        groups, cover = diary_coverage(study.family,study.numeric["n_energy_days"])
        require(not groups["shorter"] & groups["longer"],"V25_FAMILY_COVERAGE_SPLIT_DIFF")
        register(out, dict(commit=args.commit,runtime=runtime,snapshot=snapshot.manifest_sha,
                           targets=candidate,prior=proof,coverage=cover,transfer=[list(x) for x in TRANSFER],
                           code={x.name:v1_run.file_sha(x) for x in sorted(HERE.glob("*"))
                                 if x.is_file() and x.suffix in (".py",".md")}))
        diets,_,food = read_diets(root,study)
        expected_diet = read_json(REFERENCE/"inventory.json")["diet_ids"]
        require([x[0] for x in diets]==expected_diet and set(food.values())<=expected.keys(),"V25_DIET_CATALOG_DIFF")
        index = pd.DataFrame(dict(zip(KEY,study.keys,strict=True)))
        index["input_row"] = np.arange(study.n)
        y = select_values(scope,index,[e for _,e in CANDIDATES])
        replay_path = out/"existing_fold_coverage.json"
        if not replay_path.exists():
            seeds = [x[1] for x in a23["seeds"]]
            write_json(replay_path,grouped_replay(study,sites,cal,diets,y,CANDIDATES,groups,prior,seeds))
        replay = read_json(replay_path)
        require(len(replay)==2 and all(x["replay_max_difference"]<TOLERANCE for x in replay),"V25_REPLAY_RECEIPT_DIFF")
        moved = transport(study,sites,cal,diets,y,CANDIDATES,groups)
        snapshot.verify_manifest()
        require(set(snapshot.paths)-{p.outcome_file(e) for e in p.ENDPOINTS} <= snapshot.verified
                and all(v1_run.file_sha(f)==h for f,h in expected.items())
                and all(v1_run.file_sha(Path(f))==h for f,h in proof.items()),"V25_SOURCE_CHANGED_DURING_RUN")
        write_json(out/"summary.json",dict(commit=args.commit,coverage=cover,within_discovery_splits=replay,transfer=moved,
                                          reserved_outcomes_read=False,independent_validation_started=False))
    emit("V25_COMPLETE",dict(targets=len(CANDIDATES),scenarios=len(moved),reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
