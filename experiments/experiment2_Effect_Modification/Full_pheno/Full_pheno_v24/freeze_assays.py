"""Freeze and verify the discovery-trained models for the V18 hypotheses."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path[1:1] = [str(HERE.parent / "Full_pheno_v23"), str(HERE.parent.parent / "v5")]
from run_locked_review import REFERENCE, V18, V22, context, emit, parse, read_json, register, require, v1_run, write_json


V23 = HERE.parent / "Full_pheno_v23/runs/TRE_ONLY/baseline-review-2"
FOOD_SEED = 202621071
REPLAY_TOLERANCE = 1e-8


def fingerprint(arrays):
    import numpy as np

    d = sha256()
    for x in arrays:
        a = np.ascontiguousarray(x)
        if a.dtype.hasobject:
            a = a.astype(str)
        d.update(repr((a.shape,a.dtype.str)).encode())
        d.update(a.tobytes())
    return d.hexdigest()


def replay(study, sites, cal, diets, y, targets, previously):
    import numpy as np
    import protocol as p
    from batch_fit import fold_moments
    from frozen_coefficients import design, fitted_on_design
    from interaction_fit import Repartition
    from planned_validation import readout, sample_plan
    from run_expanded_grid import holm

    past = [r["baseline"]["reference"]["parts"] for r in previously["frozen_probes"]]
    seen = [[] for _ in targets]
    max_gap = 0.
    for k, (_,seed) in enumerate(read_json(V23/"registration.json")["seeds"]):
        work = Repartition(study,sites,cal,seed,emit)
        metrics = []
        for f in p.FOLDS:
            _,_,rows = fitted_on_design(design(work,f,diets),work,f,y,targets,held_out=True)
            bits = []
            for used, measured, predictions in rows:
                _, codes, counts = np.unique(study.family[used],return_counts=True,return_inverse=True)
                trio = tuple(a[:,None] for a in predictions[:3])
                bits.append(np.array(fold_moments(measured[:,None],trio,codes,counts)).ravel())
            metrics.append(np.stack(bits))
        array = np.stack(metrics)
        scored = [readout(m) for m in array.sum(axis=0)]
        holm(scored)
        for j, report in enumerate(scored):
            report["positive_folds"] = int((array[:,j,5] > array[:,j,6]).sum())
            expected = past[j][k]
            require(expected["scorable"] and all(report[x] == expected[x] for x in ("people","families","positive_folds")),
                    "V24_DISCOVERY_REPLAY_SUPPORT_DIFF")
            checks = ("diet_main_pp","interaction_pp","diet_total_pp","p_holm")
            error = max(abs(report[x] - expected[x]) for x in checks)
            require(error < REPLAY_TOLERANCE, "V24_DISCOVERY_REPLAY_SCORE_DIFF")
            max_gap = max(max_gap,error)
            seen[j].append(dict(seed=seed,**report))
        emit("V24_REPLAY",dict(seed=seed,targets=len(targets),largest_gap=max_gap))
    return dict(largest_gap=max_gap,assays=[
        dict(diet=d,endpoint=e,discoveries=seen[i],**sample_plan(seen[i])) for i,(d,e) in enumerate(targets)
    ])


def full_work(study, sites, cal):
    import numpy as np
    from source import Background
    from v3_inputs import fit_site

    training = np.ones(study.n,dtype=bool)
    views = [fit_site(s,sites[s],study.keys[0],training) for s in ("oral","gut")]
    scores = np.column_stack([x[0] for x in views])
    support = views[0][1] & views[1][1]
    upf, params, food_ids = cal.refit(training, FOOD_SEED)
    bg = Background(training,support,scores,{"UPF":upf},params)
    work = SimpleNamespace(family=study.family,numeric=study.numeric,markers=study.markers,
                           ecology=study.ecology,smoking=study.smoking,backgrounds={0:bg})
    return work,food_ids


def freeze(out, study, sites, cal, diets, y, targets):
    import numpy as np
    from frozen_coefficients import design, fitted_on_design, predict

    work, food_ids = full_work(study,sites,cal)
    parts = design(work,0,diets)
    support, host, gut, panel, descriptors = parts
    coefficients, receipts, _ = fitted_on_design(parts,work,0,y,targets,held_out=False)
    payload = {}
    for i, (model,receipt) in enumerate(zip(coefficients,receipts,strict=True)):
        prefix = f"assay_{i}"
        payload.update({f"{prefix}_{k}":v for k,v in model.items()})
        payload[f"{prefix}_clip"] = np.array([receipt["clip_lower"],receipt["clip_upper"]])
    buf = BytesIO()
    np.savez_compressed(buf,**payload)
    target = out/"coefficients_TRE_ONLY.npz"
    if not target.exists():
        v1_run.atomic_write(target,buf.getvalue())
    else:
        with np.load(target,allow_pickle=False) as file:
            require(set(file.files) == payload.keys() and all(np.array_equal(file[k],v) for k,v in payload.items()),
                    "V24_EXISTING_COEFFICIENTS_DIFF")
    for i, (m, (g,e)) in enumerate(zip(coefficients,targets,strict=True)):
        with np.load(target,allow_pickle=False) as file:
            restored = {k:file[f"assay_{i}_{k}"] for k in m}
            require(np.array_equal(file[f"assay_{i}_clip"],
                                   [receipts[i]["clip_lower"],receipts[i]["clip_upper"]]),"V24_CLIP_RELOAD_DIFF")
        args = (host[support],gut[support],panel["whole_diet"][support],panel[g][support])
        require(all(np.array_equal(a,b) for a,b in zip(predict(m,*args),predict(restored,*args),strict=True)),
                "V24_FROZEN_PREDICTIONS_RELOAD_DIFF")
        receipts[i]["training_design_sha256"] = fingerprint(
            [*(x[support] for x in study.keys),study.family[support],*args])
        receipts[i]["category_design"] = descriptors[g]
    return dict(coefficients_sha256=v1_run.file_sha(target), assay_fits=receipts,
                discovery_design_support=int(support.sum()),
                food_seed=FOOD_SEED,fit_reference_foodids=int(food_ids),
                predictions_for_new_evaluation_written=False)


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    import pandas as pd
    import protocol as p
    from batch_sources import paths, read_diets
    from freeze_reserved import select_values
    from reserved_inputs import KEY
    from sample_audit import CANDIDATES

    out = HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root,snapshot,study,cal,sites,cal_files = context()
        a18 = read_json(V18/"registration.json")
        a23 = read_json(V23/"registration.json")
        previous = read_json(V23/"summary.json")
        require(a18["targets"] == a23["frozen_targets"] == [list(x) for x in CANDIDATES]
                and previous["reserved_outcomes_read"] is False and not (V23/"RUNNING.lock").exists(),
                "V24_FROZEN_HYPOTHESES_CHANGED")
        a11 = read_json(REFERENCE/"registration.json")
        require(a11["snapshot"] == a23["snapshot"] == snapshot.manifest_sha,"V24_SOURCE_SCOPE_CHANGED")
        registered = {Path(f):h for f,h in a11["sources"].items()}
        raw = paths(root)
        scope = {s:raw[s] for s in ("nightingale","nightingale_tags","serum_lipidomics")}
        require(set((*scope.values(),*cal_files)) <= registered.keys()
                and all(v1_run.file_sha(f) == h for f,h in registered.items()),"V24_UNREGISTERED_SOURCE")
        protocol_v18 = HERE.parent/"Full_pheno_v18/PROTOCOL.md"
        require(v1_run.file_sha(protocol_v18) == a18["code"][protocol_v18.name],"V24_V18_PROTOCOL_CHANGED")
        census = V18/"candidate_visit_key_overlap.json"
        availability = read_json(census)
        require(availability["outcome_values_accessed"] is False,"V24_AVAILABILITY_SCOPE_CHANGED")
        anchors = {str(f):v1_run.file_sha(f) for f in (
            protocol_v18,census,V18/"registration.json",V23/"registration.json",V23/"summary.json",
            V22/"registration.json",REFERENCE/"inventory.json",
        )}
        register(out,dict(commit=args.commit,runtime=runtime,snapshot=snapshot.manifest_sha,inputs={str(k):v for k,v in registered.items()},
                          anchors=anchors,targets=[list(x) for x in CANDIDATES],food_seed=FOOD_SEED,
                          model_code={f.name:v1_run.file_sha(f) for f in sorted(HERE.glob("*"))
                                      if f.is_file() and f.suffix in (".md",".py")}))
        diets,_,food = read_diets(root,study)
        inv = read_json(REFERENCE/"inventory.json")
        require([d[0] for d in diets] == inv["diet_ids"]
                and set(food.values()) <= registered.keys(),"V24_DIET_CATALOG_CHANGED")
        index = pd.DataFrame(dict(zip(KEY,study.keys,strict=True)))
        index["input_row"] = np.arange(study.n)
        y = select_values(scope,index,[e for _,e in CANDIDATES])
        report_path = out/"assay_replay_and_planning.json"
        if not report_path.exists():
            write_json(report_path,replay(study,sites,cal,diets,y,CANDIDATES,previous))
        report = read_json(report_path)
        require(report["largest_gap"] < REPLAY_TOLERANCE and len(report["assays"])==len(CANDIDATES),
                "V24_INVALID_REPLAY_RECORD")
        model = freeze(out,study,sites,cal,diets,y,CANDIDATES)
        needed = set(snapshot.paths) - {p.outcome_file(e) for e in p.ENDPOINTS}
        snapshot.verify_manifest()
        require(needed <= snapshot.verified and all(v1_run.file_sha(f) == h for f,h in registered.items())
                and all(v1_run.file_sha(Path(f)) == h for f,h in anchors.items()),"V24_SOURCE_CHANGED_DURING_FREEZE")
        receipt_path = out/"FROZEN.json"
        receipt = dict(commit=args.commit,registration_sha256=v1_run.file_sha(out/"registration.json"),
                       planning_sha256=v1_run.file_sha(report_path),current_independent_sample=None,
                       independent_scoring_started=False,reserved_outcomes_read=False,**model)
        if receipt_path.exists():
            require(read_json(receipt_path) == receipt, "V24_RECEIPT_CHANGED")
        else:
            write_json(receipt_path,receipt)
    emit("V24_COMPLETE",dict(assays=len(CANDIDATES),largest_gap=report["largest_gap"],
                             discovery_fits=[r["people"] for r in model["assay_fits"]],
                             independent_scoring_started=False))


if __name__ == "__main__":
    main()
