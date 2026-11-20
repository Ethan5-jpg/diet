"""Build household diary groups and quantify paired discovery transfer."""

from types import SimpleNamespace

import numpy as np


BOOTSTRAP_DRAWS = 999
RANDOM_SPLIT_SEED = 20263037


def diary_coverage(families, days):
    qualified = np.isfinite(days) & (days >= 3)
    names, codes = np.unique(families,return_inverse=True)
    family_days = np.full(len(names),np.nan)
    for c in np.unique(codes[qualified]):
        family_days[c] = np.median(days[(codes==c)&qualified])
    finite = np.isfinite(family_days)
    observed = family_days[finite]
    levels = np.unique(observed)
    if len(levels)<2:
        raise ValueError("V25_COVERAGE_HAS_ONE_LEVEL")
    boundaries = (levels[:-1]+levels[1:])/2.
    rank = np.searchsorted(np.sort(observed),boundaries)
    boundary = float(boundaries[np.argmin(abs(rank-.5*len(observed)))])
    roles = {k: set(names[finite & v]) for k,v in (
        ("under_10",family_days<10.),("at_least_10",family_days>=10.),
        ("shorter",family_days<boundary),("longer",family_days>=boundary),
    )}
    randomized = np.random.default_rng(RANDOM_SPLIT_SEED).permutation(names[finite])
    boundary_index = len(roles["shorter"])
    roles.update(random_a=set(randomized[:boundary_index]),random_b=set(randomized[boundary_index:]))
    counts = {k:dict(families=len(v), median_days=float(np.median(family_days[np.isin(names,list(v))])))
              for k,v in roles.items() if v}
    return roles, dict(boundary=boundary, household_coverage=counts, unavailable_families=int((~finite).sum()))


def transfer_work(study, sites, cal, training, held, seed):
    from source import Background
    from v3_inputs import fit_site
    from run_v2 import require

    require(np.any(training) and np.any(held) and not np.any(training & held)
            and not np.intersect1d(study.family[training],study.family[held]).size,"V25_TRANSFER_ROLE_LEAKAGE")
    sites_fitted = [fit_site(s,sites[s],study.keys[0],training) for s in ("oral","gut")]
    pc = np.column_stack([x[0] for x in sites_fitted])
    support = sites_fitted[0][1] & sites_fitted[1][1] & (training | held)
    upf, meta, _ = cal.refit(training,seed)
    bg = Background(training,support,pc,{"UPF":upf},meta)
    return SimpleNamespace(family=study.family,numeric=study.numeric,ecology=study.ecology,
                           markers=study.markers,smoking=study.smoking,backgrounds={0:bg})


def family_moments(family, y, predictions):
    u,idx = np.unique(family,return_inverse=True)
    rows = np.column_stack((np.ones(len(y)),y,y*y,*((y-p)**2 for p in predictions)))
    household = np.zeros((len(u),rows.shape[1]))
    np.add.at(household,idx,rows)
    return u, household


def scores(household, seed):
    from planned_validation import readout

    people = int(household[:,0].sum())
    families = len(household)
    if families < 40:
        return dict(scorable=False,people=people,families=families,p_one_sided=1.)
    n, y, y2, a, b, c, w = household.sum(axis=0)
    loss = household[:,4]-household[:,5]
    standard_cp = float((y2-y*y/n))
    values = readout([n,families,y,y2,a,b,c,loss@loss,household[:,0]@loss,
                      household[:,0]@household[:,0]])
    rng = np.random.default_rng(seed)
    delta = []
    for start in range(0,BOOTSTRAP_DRAWS,64):
        count = min(64,BOOTSTRAP_DRAWS-start)
        idx = rng.integers(0,families,size=(count,families))
        tt = household[idx].sum(axis=1)
        sst = tt[:,2]-tt[:,1]**2/tt[:,0]
        delta.extend(np.divide(100.*(tt[:,4]-tt[:,5]),sst,out=np.full(count,np.nan),where=sst>0.))
    values.update(whole_interaction_pp=100.*(b-w)/standard_cp, scorable=True,
                  interaction_percentile_ci_pp=np.nanquantile(delta,[.025,.975]).tolist())
    return values


def summarize_panel(groups, targets, label, select, seed):
    from run_expanded_grid import holm
    from run_v2 import require

    rows = []
    for i,(g,e) in enumerate(targets):
        family = np.concatenate([x[0] for x in groups[i]])
        values = np.vstack([x[1] for x in groups[i]])
        require(len(np.unique(family))==len(family),"V25_HOUSEHOLD_DUPLICATED")
        subset = np.ones(len(family),dtype=bool) if select is None else np.isin(family,list(select))
        row = dict(diet=g,endpoint=e,stratum=label,**scores(values[subset],seed+i))
        rows.append(row)
    holm(rows)
    for r in rows:
        r["practical"] = bool(r["scorable"] and r["interaction_pp"]>=.10 and r["diet_total_pp"]>0.)
    return rows
