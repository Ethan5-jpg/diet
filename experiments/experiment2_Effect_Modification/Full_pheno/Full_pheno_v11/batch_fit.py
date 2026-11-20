"""Fit diet and diet-by-gut models in blocks of exactly matched people."""

from collections import defaultdict
from statistics import median

import numpy as np
from scipy.stats import t

import protocol as p
from models import create_design, dose_basis, fiber_basis
from run_expanded_grid import MIN_HELD_HOUSEHOLDS, MIN_TRAIN_HOUSEHOLDS, holm
from run_v2 import require
from v4_models import no_diet_main


def context_design(study, fold, exposure):
    bg = study.backgrounds[fold]
    days, energy, _ = p.EXPOSURES[exposure]
    ok = (bg.supported & (study.numeric[days] >= 3) & (study.numeric[energy] > 0.)
          & np.isfinite(study.numeric["age_calendar_years"]) & np.isin(study.numeric["sex_code"], (0,1))
          & np.isfinite(bg.archived[exposure]).all(axis=1))
    training = ok & bg.role
    rows = np.r_[np.flatnonzero(training), np.flatnonzero(ok & ~bg.role)]
    n = int(training.sum())
    design = create_design(study, exposure, fold, rows, n)
    raw = (fiber_basis(study.numeric[exposure][rows]) if exposure=="fiber_density"
           else bg.archived[exposure][rows])
    main = no_diet_main(study, rows, n, design.main, dose_basis(raw,n)[0])
    plain = np.zeros((len(study.family), main.shape[1]))
    plain[rows] = main
    return ok, plain


def outcome_groups(values):
    groups = defaultdict(list)
    finite = np.isfinite(values)
    for i, key in enumerate(np.packbits(finite, axis=0).T):
        groups[key.tobytes()].append(i)
    return [(finite[:, col[0]],np.array(col)) for col in groups.values()]


def clip_outcomes(values, role):
    """Winsorize observed training and held measurements at training quantiles."""
    fit = values[role]
    observed = np.isfinite(fit).any(axis=0)
    result = values.copy()
    lo, hi = np.nanquantile(fit[:, observed], [.01,.99], axis=0)
    result[:,observed] = np.clip(values[:,observed],lo,hi)
    return result


def match_rows(mask, role, family):
    tr, held = np.flatnonzero(mask & role), np.flatnonzero(mask & ~role)
    train_families = np.unique(family[tr])
    levels, codes, counts = np.unique(family[held], return_inverse=True, return_counts=True)
    if len(train_families)<MIN_TRAIN_HOUSEHOLDS or len(levels)<MIN_HELD_HOUSEHOLDS:
        return None
    require(not np.intersect1d(train_families,levels).size, "V11_HOUSEHOLD_LEAKAGE")
    return np.r_[tr,held], len(tr), codes, counts


def fit_block(plain, pcs, dietary, outcomes, n):
    """Three paired prediction arms with one background and fixed half-strength gut PC5."""
    fit_x = dietary[:n]
    lo, hi = np.quantile(fit_x,[.01,.99])
    x = np.clip(dietary,lo,hi)
    train_x = x[:n]
    sd = float(train_x.std())
    if sd<=p.RANK_TOL or min(np.sum(train_x>train_x.min()),np.sum(train_x<train_x.max()))<20:
        return None
    x = (x[:,None]-train_x.mean())/sd
    nuisance = plain[:n]
    inverse = np.linalg.pinv(nuisance.T@nuisance, rcond=p.RANK_TOL, hermitian=True)
    left = inverse@nuisance.T
    a = plain@(left@outcomes[:n])
    z = x - plain@(left@x[:n])
    norm = float((z[:n]**2).sum())
    if norm <= p.RANK_TOL**2*n:
        return None
    b = a + z@(z[:n].T@(outcomes[:n]-a[:n]))/(norm+.10*n)
    interaction = x*pcs
    gut = interaction - plain@(left@interaction[:n])
    gut -= z@(z[:n].T@gut[:n])/norm
    design = gut[:n]
    coef = np.linalg.solve(design.T@design+n*np.eye(pcs.shape[1]),design.T@(outcomes[:n]-b[:n]))
    return a[n:], b[n:], b[n:]+.5*(gut[n:]@coef)


def fold_moments(y, predictions, codes, counts):
    err_a,err_b,err_c=((y-x)**2 for x in predictions)
    paired = err_b-err_c
    groups=np.zeros((len(counts),y.shape[1]))
    np.add.at(groups,codes,paired)
    return np.broadcast_arrays(len(y), len(counts), y.sum(0), (y*y).sum(0), err_a.sum(0), err_b.sum(0), err_c.sum(0),
                               (groups*groups).sum(0), counts@groups, counts@counts)


def accumulate(study, diets, endpoints, values, progress):
    family=np.unique(study.family,return_inverse=True)[1]
    groups=outcome_groups(values)
    names=np.array(endpoints)
    old_outcomes=np.isin(names,list(p.ENDPOINTS))
    acceptable=np.zeros((len(diets),len(names),len(p.FOLDS)),dtype=np.int8)
    result=np.zeros((*acceptable.shape,10))
    for index, fold in enumerate(p.FOLDS):
        bg=study.backgrounds[fold]
        y=clip_outcomes(values,bg.role)
        shared={x:context_design(study,fold,x) for x in p.EXPOSURES}
        for j,(diet,kind,data) in enumerate(diets):
            dose=bg.archived["UPF"][:,0] if diet=="UPF" else data
            supported, plain=shared[kind]
            diet_valid=supported & np.isfinite(dose)
            for observed, cols in groups:
                chosen=cols[~old_outcomes[cols]] if diet in p.EXPOSURES else cols
                if not len(chosen):
                    continue
                match=match_rows(diet_valid & observed,bg.role,family)
                if match is None:
                    continue
                rows,n,codes,counts=match
                outcome=y[np.ix_(rows,chosen)]
                z=fit_block(plain[rows],bg.pcs[rows,p.N_PC:2*p.N_PC],dose[rows],outcome,n)
                if z is None:
                    acceptable[j,chosen,index]=1
                    continue
                moments=np.column_stack(fold_moments(outcome[n:],z,codes,counts))
                require(np.isfinite(moments).all(),"V11_NONFINITE_PREDICTION_MOMENTS")
                result[j,chosen,index]=moments
                acceptable[j,chosen,index]=2
        progress(fold,int((acceptable[...,index]==2).sum()))
    return result,acceptable


def partition_scores(diets, endpoints, result, acceptable):
    people, households, sy, sy2, a, b, c, sum2, sum_n, n2 = result.sum(axis=-2).transpose(2,0,1)
    tested = (acceptable==2).all(axis=-1)
    sst = np.zeros_like(sy)
    sst[tested] = sy2[tested]-sy[tested]**2/people[tested]
    valid = tested & (sst > 1e-10*np.maximum(np.finfo(float).tiny,np.abs(sy2)))
    j,k = np.where(valid)
    n,h = people[j,k],households[j,k]
    gain=b[j,k]-c[j,k]
    mean=gain/n
    se2=(sum2[j,k]-2.*mean*sum_n[j,k]+n2[j,k]*mean**2)*h/(h-1.)
    se=np.sqrt(np.maximum(0.,se2))
    stat=np.divide(gain,se,out=np.zeros(len(se)),where=se>0.)
    pvalue=t.sf(stat,h-1.)
    pp=100./sst[j,k]
    records=[dict(diet=diets[u][0],endpoint=endpoints[v],people=int(n[i]),families=int(h[i]),
                  interaction_pp=float(gain[i]*pp[i]),diet_main_pp=float((a[u,v]-b[u,v])*pp[i]),
                  diet_total_pp=float((a[u,v]-c[u,v])*pp[i]),p_one_sided=float(pvalue[i]),
                  positive_folds=int((result[u,v,:,5]>result[u,v,:,6]).sum())) for i,(u,v) in enumerate(zip(j,k))]
    holm(records)
    total=len(diets)*len(endpoints)-len(p.MODES["all"])
    nonstarter=sum((k=="UPF" or k=="fiber_density")*len(p.ENDPOINTS) for k,*_ in diets)
    require(nonstarter==len(p.MODES["all"]),"V11_EXCLUDED_ORIGINAL_COUNT")
    refused=dict(household_support=int((acceptable==0).any(axis=-1).sum())-nonstarter,
                 diet_variation=int((acceptable>0).all(axis=-1).sum()-tested.sum()),
                 endpoint_variation=int(tested.sum()-valid.sum()))
    require(sum(refused.values())+len(records)==total,"V11_PARTITION_COUNTS")
    return records,refused


def stable_combinations(partitions):
    records=[{(r["diet"],r["endpoint"]):r for r in part["rows"]} for part in partitions]
    common=set.intersection(*(set(r) for r in records))
    stable=[]
    for key in common:
        group=[part[key] for part in records]
        gain=[r["interaction_pp"] for r in group]
        if min(gain)<=0. or min(r["positive_folds"] for r in group)<3:
            continue
        total=[r["diet_total_pp"] for r in group]
        stable.append(dict(diet=key[0], endpoint=key[1], interaction_pp=gain,diet_total_pp=total,
                           p_holm=[r["p_holm"] for r in group], min_families=min(r["families"] for r in group),
                           min_gain=min(gain),median_gain=median(gain),min_total=min(total)))
    stable.sort(key=lambda r:(-r["min_gain"], -r["median_gain"],r["diet"],r["endpoint"]))
    return dict(shared=len(common), positive_everywhere=len(stable), threshold_everywhere=sum(r["min_gain"]>=.1 for r in stable),
                joint_everywhere=sum(r["min_gain"]>=.1 and r["min_total"]>0. for r in stable),
                holm_everywhere=sum(max(r["p_holm"])<.05 for r in stable),stable=stable)
