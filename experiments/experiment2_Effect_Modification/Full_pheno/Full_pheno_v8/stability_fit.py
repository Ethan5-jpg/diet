"""Hold the previous precision interaction fixed while testing stronger diet arms."""

from dataclasses import dataclass

import numpy as np

import protocol as p
from breadth import basis, fit_profile
from design import domain
from diet_audit import Fit, ridge_diet
from exposure_models import ANCHOR, inputs
from interaction_fit import Repartition
from interaction_rules import domain_prior, prediction
from models import create_design, dose_basis, predict, products
from run_precision import features
from source import Outcome, check
from stability_rules import FIXED, choose, with_adaptations
from v3_inputs import InnerContexts, read_sites
from v4_models import no_diet_main


@dataclass(frozen=True)
class Endpoint:
    rows: np.ndarray
    y: np.ndarray
    base: np.ndarray
    off: tuple
    mods: dict
    anchor: np.ndarray | None


def projection(x, main, inv, n):
    return x-main@(inv@x[:n])


def bagged_coefficients(x, resid, codes, penalty=1.):
    """Average ridge coefficients over four training household deletions."""
    levels = np.unique(codes)
    check(len(levels)>=4,"V8_BAGGING_HOUSEHOLDS")
    inverse = np.searchsorted(levels,codes)
    block = np.random.default_rng(20261120).permutation(len(levels))[inverse]%4
    coefs = []
    for b in range(4):
        z, r = x[block!=b], resid[block!=b]
        coefs.append(np.linalg.solve(z.T@z/len(z)+penalty*np.eye(x.shape[1]), z.T@r/len(z)))
    return np.mean(coefs,axis=0)


def fit_endpoint(study, exposure, endpoint, outcome, fold, raw, views, reliable):
    y,tr,held = inputs(study,exposure,endpoint,outcome,fold)
    n,rows = len(tr),np.r_[tr,held]
    check(n>=30 and len(held)>=10 and not np.intersect1d(study.family[tr],study.family[held]).size,"V8_FIT_ROLES")
    ytr = y[tr]
    design = create_design(study,exposure,fold,rows,n)
    archived = dose_basis(raw[rows],n)[0]
    base = no_diet_main(study,rows,n,design.main,archived)
    base_inv = np.linalg.pinv(base[:n],rcond=p.RANK_TOL)
    nodiet = base[n:]@(base_inv@ytr)
    legacy_inv = np.linalg.pinv(design.main[:n],rcond=p.RANK_TOL)
    legacy = design.main@(legacy_inv@ytr)
    if (exposure,endpoint)==ANCHOR:
        return Endpoint(held,y[held],nodiet,(legacy[n:],),{},predict(design,ytr)[:,2])

    new = reliable[rows,None]
    med = float(np.median(new[:n]))
    new_dose = dose_basis(np.column_stack((new[:,0],np.maximum(new[:,0]-med,0.))),n)[0]
    old_terms = np.column_stack((archived,products(archived,design.main[:,1:3])))
    new_terms = np.column_stack((new_dose,products(new_dose,design.main[:,1:3])))
    penalty = (.1,.5,1.,1.,1.,1.)
    incumbent, old_projections = ridge_diet(base,base_inv,old_terms,ytr,penalty)
    linear, linear_projection = ridge_diet(base,base_inv,new_dose[:,:1],ytr,(.1,))
    curve, new_projections = ridge_diet(base,base_inv,new_terms,ytr,penalty)
    blend = .5*(incumbent+linear)
    nuisance = (
        (incumbent,np.column_stack((base,old_projections))),
        (linear,np.column_stack((base,linear_projection))),
        (curve,np.column_stack((base,new_projections))),
        (blend,np.column_stack((base,old_projections,linear_projection))),
    )
    pc = views["gut_pc5"][rows]
    pc_legacy = products(raw[rows,:1],pc)
    gut = products(new,pc)
    oral = products(new_dose,views["ecology"][rows,:1])
    legacy_c = fit_profile(pc_legacy,design.main,legacy_inv,ytr-legacy[:n]).fit(1.)
    mods = {"legacy":legacy_c}
    for i,(b,main) in enumerate(nuisance):
        inv = np.linalg.pinv(main[:n],rcond=p.RANK_TOL)
        resid = ytr-b[:n]
        mods[i] = (fit_profile(gut,main,inv,resid),fit_profile(gut[:,:2],main,inv,resid))
        if i in (0,3):
            augmented = np.column_stack((main,gut))
            aug_inv = np.linalg.pinv(augmented[:n],rcond=p.RANK_TOL)
            mods[i,"oral"] = fit_profile(oral,augmented,aug_inv,resid).fit(.25)
        if i==3:
            z = projection(gut,main,inv,n)
            mods["bagged"] = (z[n:]@bagged_coefficients(z[:n],resid,study.family[tr]),
                              z[n:,:2]@bagged_coefficients(z[:n,:2],resid,study.family[tr]))
    return Endpoint(held,y[held],nodiet,(legacy[n:],incumbent[n:],linear[n:],curve[n:],blend[n:]),mods,None)


def weighted(pair):
    return 1.37*(.45*pair[0]+.35*pair[1])


def fit_split(study, exposure, outcomes, fold, days):
    raw, views = basis(study,exposure,fold)
    dose,_ = features(exposure,study.backgrounds[fold],raw[:,0],days)
    work = {e:fit_endpoint(study,exposure,e,y,fold,raw,views,dose["person_clipped"]) for e,y in outcomes.items()}
    result = {}
    for e,f in work.items():
        if f.anchor is not None:
            off = np.repeat(f.off[0][:,None],len(FIXED),axis=1)
            on = np.repeat(f.anchor[:,None],len(FIXED),axis=1)
        else:
            donors = [part.mods[3] for other,part in work.items() if other!=e and domain(e)==domain(other)]
            pool = tuple(prediction(p1,1.,.50,domain_prior([pair[i] for pair in donors]))
                         for i,p1 in enumerate(f.mods[3]))
            c = [weighted(tuple(x.fit(1.) for x in f.mods[i])) for i in range(4)]
            legacy, incumbent, linear, curve, blend = f.off
            off = np.column_stack((incumbent,legacy,linear,curve,blend,blend,blend,blend,incumbent))
            on = off + np.column_stack((c[0],.5*f.mods["legacy"],c[1],c[2],c[3],weighted(f.mods["bagged"]),
                                      weighted(pool),c[3]+.15*f.mods[3,"oral"],c[0]+.15*f.mods[0,"oral"]))
        check(np.isfinite(off).all() and np.isfinite(on).all(),"V8_NONFINITE_PREDICTIONS")
        result[e] = Fit(f.rows,f.y,f.base,off,on)
    return result


def run_partition(study, outcomes, days, calibration, seed, progress, adaptive=True):
    sites = read_sites(study) if adaptive or seed is not None else None
    working = study if seed is None else Repartition(study,sites,calibration,seed,progress)
    inner = InnerContexts(working,sites) if adaptive else None
    fits = {task:[] for task in p.MODES["all"]}
    accepted = [0,0]
    for outer in p.FOLDS:
        choices = {x:{} for x in p.EXPOSURES}
        if adaptive:
            ys = {e:Outcome(np.where(working.fold!=outer,y.y,np.nan),y.unit) for e,y in outcomes.items()}
            candidates = {x:{e:[] for e in outcomes} for x in p.EXPOSURES}
            for val in p.FOLDS:
                if val==outer:
                    continue
                context = inner.split(outer,val)
                for x in p.EXPOSURES:
                    if x=="UPF":
                        context,_ = calibration.inner(context,val,100*(seed or p.BOOTSTRAP_SEED)+10*outer+val)
                    every = fit_split(context,x,ys,val,days)
                    for e,f in every.items():
                        candidates[x][e].append(f)
            for x, per_target in candidates.items():
                for e,parts in per_target.items():
                    choices[x][e] = (choose(parts,strict=False),choose(parts,strict=True))
        for x in p.EXPOSURES:
            for e,f in fit_split(working,x,outcomes,outer,days).items():
                check(np.all(working.fold[f.rows]==outer),"V8_HELD_FOLD_CONFLICT")
                decision = choices[x].get(e,((0,0.),(0,0.)))
                for j, (_, w) in enumerate(decision):
                    accepted[j] += int(w>0.)
                fits[x,e].append(with_adaptations(f,decision))
        progress("V8_FOLD_DONE",dict(fold=outer,seed=seed,accepted_upgrades=accepted))
    return fits, accepted
