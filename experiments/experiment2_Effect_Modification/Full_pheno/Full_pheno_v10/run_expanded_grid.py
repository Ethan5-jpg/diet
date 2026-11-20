"""Screen new diet–outcome pairs separately from the immutable original panel."""

from pathlib import Path
from statistics import median

from run_locked import context, emit, parse, register, require, v1_run, HERE


PARTITIONS = (("archive", None), ("fresh-1", 20261703), ("fresh-2", 20261741))
MIN_TRAIN_HOUSEHOLDS = 120
MIN_HELD_HOUSEHOLDS = 40


def diet_catalog(values, audit):
    import numpy as np

    recipes = []
    for group, panel in values.items():
        specs = audit[group]
        for spec, column in zip(specs, panel.T, strict=True):
            name = spec["feature"]
            context = "fiber_density" if group=="fiber_sources" or name in {
                "fiber_excess", "high_fiber_day", "fiber_day_sd"
            } else "UPF"
            recipes.append((name, group, context, column))
    shares = values["food_groups"]
    finite = np.isfinite(shares).all(axis=1)
    support = finite & (np.nansum(shares, axis=1) > 0.)
    freq = shares[support] / shares[support].sum(axis=1, keepdims=True)
    log = np.log(freq, out=np.zeros_like(freq), where=freq>0.)
    diversity = np.full(len(shares), np.nan)
    diversity[support] = -np.sum(freq*log,axis=1)
    recipes.append(("food_group_energy_diversity", "food_groups", "UPF", diversity))
    return recipes


def predict_new(study, fold, context, values, y, backgrounds):
    import numpy as np
    import protocol as p
    from models import create_design, dose_basis, fiber_basis, products
    from v4_models import no_diet_main

    bg = study.backgrounds[fold]
    days, kcal, _ = p.EXPOSURES[context]
    ok = (np.isfinite(values) & np.isfinite(y) & bg.supported & (study.numeric[days]>=3)
          & (study.numeric[kcal]>0.) & np.isfinite(study.numeric["age_calendar_years"])
          & np.isin(study.numeric["sex_code"],(0,1)) & np.isfinite(bg.archived[context]).all(axis=1))
    tr, held = np.flatnonzero(ok & bg.role), np.flatnonzero(ok & ~bg.role)
    train_groups, test_groups = np.unique(study.family[tr]), np.unique(study.family[held])
    if len(train_groups)<MIN_TRAIN_HOUSEHOLDS or len(test_groups)<MIN_HELD_HOUSEHOLDS:
        return None, "household_support"
    require(not np.intersect1d(train_groups,test_groups).size, "V10_GRID_HOUSEHOLD_LEAKAGE")
    rows, n = np.r_[tr, held], len(tr)
    lo, hi = np.quantile(values[tr], [.01,.99])
    x = np.clip(values[rows],lo,hi)
    train_x = x[:n]
    scale = float(train_x.std())
    if scale<=p.RANK_TOL or min(np.sum(train_x > train_x.min()), np.sum(train_x < train_x.max()))<20:
        return None, "diet_variation"
    x = ((x-x[:n].mean())/scale)[:,None]
    # Reuse the same fitted background for this endpoint and exact matched cohort.
    key = (fold,context,np.packbits(ok).tobytes())
    if key not in backgrounds:
        design = create_design(study, context, fold, rows, n)
        raw = (fiber_basis(study.numeric["fiber_density"][rows]) if context=="fiber_density"
               else bg.archived[context][rows])
        dose = dose_basis(raw,n)[0]
        plain = no_diet_main(study, rows, n, design.main, dose)
        inv = np.linalg.pinv(plain[:n],rcond=p.RANK_TOL)
        backgrounds[key] = plain, inv, plain @ (inv @ y[tr])
    plain, inv, a = backgrounds[key]
    z = x - plain @ (inv @ x[:n])
    norm = float((z[:n]**2).sum())
    if norm<=p.RANK_TOL**2*n:
        return None, "diet_collinearity"
    b = a + (z @ (z[:n].T @ (y[tr]-a[:n]))) / (norm+.10*n)
    z_gut = products(x,bg.pcs[rows,p.N_PC:2*p.N_PC])
    gut = z_gut - plain @ (inv @ z_gut[:n])
    gut -= z @ (z[:n].T @ gut[:n]) / norm
    train_gut = gut[:n]
    penalty = train_gut.T @ train_gut + n*np.eye(train_gut.shape[1])
    coef = np.linalg.solve(penalty,train_gut.T @ (y[tr]-b[:n]))
    c = b[n:] + .5*(gut[n:] @ coef)
    require(np.isfinite(np.column_stack((a[n:],b[n:],c))).all(), "V10_GRID_NONFINITE")
    return (held, y[held], a[n:], b[n:], c), None


def scores(study, parts):
    import numpy as np
    from scipy.stats import t

    rows, y, a, b, c = (np.concatenate(x) for x in zip(*parts, strict=True))
    yscale = float(np.sum((y-y.mean())**2))
    if yscale <= 1e-10:
        return None
    require(len(rows)==len(np.unique(rows)), "V10_GRID_DUPLICATE_HELD")
    _, family = np.unique(study.family[rows],return_inverse=True)
    err_a, err_b, err_c = ((y-q)**2 for q in (a,b,c))
    diff = err_b-err_c
    n_families = int(family.max())+1
    adjusted_errors = np.bincount(family,weights=diff-diff.mean())
    se = float(np.linalg.norm(adjusted_errors)*np.sqrt(n_families/(n_families-1)))
    stat = float(diff.sum())/se if se>0. else 0.
    point, diet, total = (100.*float(x.sum())/yscale for x in (diff,err_a-err_b,err_a-err_c))
    require(abs(point+diet-total)<1e-8,"V10_GRID_CONTRAST")
    return dict(people=len(rows), families=n_families, interaction_pp=point, diet_main_pp=diet,
                diet_total_pp=total, p_one_sided=float(t.sf(stat,n_families-1)),
                positive_folds=int(sum(np.sum((q[1]-q[3])**2-(q[1]-q[4])**2)>0. for q in parts)))


def holm(rows):
    high, n = 0., len(rows)
    for j, row in enumerate(sorted(rows,key=lambda r:r["p_one_sided"])):
        high = max(high, (n-j)*row["p_one_sided"])
        row["p_holm"] = min(1.,high)


def run_partition(study, calibration, sites, diets, outcomes, seed):
    import protocol as p
    from interaction_fit import Repartition
    from source import clean_outcome

    work = study if seed is None else Repartition(study, sites, calibration, seed, emit)
    clean = {e:[clean_outcome(y.y,work.backgrounds[f].role) for f in p.FOLDS] for e,y in outcomes.items()}
    records, excluded = [], []
    for j, e in enumerate(outcomes, 1):
        backgrounds = {}
        for name, group, kind, value in diets:
            vals = [work.backgrounds[f].archived["UPF"][:,0] if name=="UPF" else value for f in p.FOLDS]
            if group=="original" and e in p.ENDPOINTS:
                continue
            parts, reason = [], None
            for fold, y, x in zip(p.FOLDS,clean[e],vals,strict=True):
                part, reason = predict_new(work,fold,kind,x,y,backgrounds)
                if reason:
                    break
                parts.append(part)
            identifier = dict(diet=name, group=group, endpoint=e)
            if reason is None:
                row = scores(work,parts)
                if row is not None:
                    records.append(dict(identifier, **row))
                    continue
                reason = "endpoint_variation"
            excluded.append(dict(identifier, reason=reason))
        if j%10==0 or j==len(outcomes):
            emit("V10_GRID_ENDPOINT_BATCH",dict(seed=seed,endpoints=j,tested=len(records)))
    holm(records)
    return dict(seed=seed, rows=records, excluded=excluded)


def shortlist(partitions):
    """Apply the same gain and direction criteria to all registered split repetitions."""
    index = [{(r["diet"],r["endpoint"]):r for r in p["rows"]} for p in partitions]
    shared = set.intersection(*(set(a) for a in index))
    detail = []
    for key in sorted(shared):
        rs = [i[key] for i in index]
        effects = [r["interaction_pp"] for r in rs]
        if min(effects)>0. and min(r["positive_folds"] for r in rs)>=3:
            detail.append(dict(diet=key[0], endpoint=key[1], interaction_pp=effects,
                               diet_total_pp=[r["diet_total_pp"] for r in rs],
                               min_households=min(r["families"] for r in rs),
                               min_gain=min(effects), min_total=min(r["diet_total_pp"] for r in rs),
                               median_gain=median(effects), p_holm=[r["p_holm"] for r in rs]))
    ranking = sorted(detail,key=lambda r:(r["min_gain"],r["median_gain"]),reverse=True)
    return dict(shared_pairs=len(shared), recurring_positive=len(ranking),
                recurring_ge_010=sum(r["min_gain"]>=.1 for r in ranking),
                recurring_joint=sum(r["min_gain"]>=.1 and r["min_total"]>0. for r in ranking),
                recurring_holm=sum(max(r["p_holm"])<.05 for r in ranking), stable=ranking)


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from diet_inventory import read_inventory
    from food_features import SOURCES
    from new_endpoints import read_new_outcomes, source_paths

    out = HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, calibration, sites, cal_files = context()
        outcome_paths = source_paths(root)
        diet_paths = [(root/v).resolve(strict=True) for v in SOURCES.values()]
        sources = {str(path):v1_run.file_sha(path) for path in
                   set((*cal_files, *diet_paths, *outcome_paths.values()))}
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snapshot.manifest_sha, sources=sources,
                           seeds=[list(x) for x in PARTITIONS],
                           code={name:v1_run.file_sha(HERE/name) for name in ("run_expanded_grid.py","new_endpoints.py","run_locked.py")}))
        values, audit, matched_paths = read_inventory(root,study)
        require(set(matched_paths.values())==set(diet_paths), "V10_GRID_DIET_SOURCE")
        diets = diet_catalog(values,audit)
        diets += [("fiber_density","original","fiber_density",study.numeric["fiber_density"]),
                  ("UPF","original","UPF",study.backgrounds[0].archived["UPF"][:,0])]
        extra, endpoint_audit, missing = read_new_outcomes(outcome_paths,study)
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS} | extra
        require(not set(extra)&p.ENDPOINTS.keys() and len(extra)>0, "V10_GRID_DISTINCT_ENDPOINTS")
        count = len(diets)*len(outcomes)-len(p.MODES["all"])
        emit("V10_GRID_INVENTORY",dict(new_diet=len(diets)-2,new_endpoints=len(extra),pairs=count))
        partitions = []
        for name, seed in PARTITIONS:
            path = out/f"{name}.json"
            if path.exists():
                result = v1_run.read_json(path)
                require(result["seed"]==seed, "V10_GRID_CHECKPOINT_SEED")
            else:
                result = run_partition(study, calibration, sites, diets, outcomes, seed)
                v1_run.atomic_write(path,v1_run.encode(result))
            require(len(result["rows"])+len(result["excluded"])==count,"V10_GRID_CHECKPOINT_SIZE")
            partitions.append(result)
            emit("V10_GRID_PARTITION",dict(name=name, tested=len(result["rows"]), skipped=len(result["excluded"])))
        require(set(snapshot.paths)<=snapshot.verified,"V10_GRID_UNVERIFIED_SNAPSHOT")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(k))==v for k,v in sources.items()),"V10_GRID_SOURCE_CHANGED")
        stable = shortlist(partitions)
        summary = dict(commit=args.commit, partitions=[a for a,_ in PARTITIONS], candidate_pairs=count,
                       diet_inventory=audit, candidate_diets=[pair[0] for pair in diets],
                       endpoint_inventory=endpoint_audit, missing_fields=missing, **stable, reserved_outcomes_read=False)
        v1_run.atomic_write(out/"summary.json",v1_run.encode(summary))
    emit("V10_GRID_COMPLETE",{k:v for k,v in stable.items() if k!="stable"})


if __name__=="__main__":
    main()
