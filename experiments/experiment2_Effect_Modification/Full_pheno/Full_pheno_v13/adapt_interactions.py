"""Calibrate category interaction predictions using training households only."""

from collections import Counter
from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy.stats import t

import protocol as p
from batch_fit import clip_outcomes, context_design, fold_moments, match_rows, outcome_groups
from category_fit import left_inverse, panels, residualize
from run_expanded_grid import holm
from run_v2 import require
from taxonomy import GROUPS


DIRECTIONS=("whole_diet", *GROUPS)
METHODS=("fixed_whole", "fixed_half", "domain_whole", "domain_mix", "endpoint_guard", "interaction_off")
WIDTH=len(DIRECTIONS)


def candidate_weights():
    axes=np.eye(WIDTH)
    single=np.vstack([w*axes for w in (.25,.5,.75,1.)])
    blends=[strength*(fraction*axes[a]+(1.-fraction)*axes[b])
            for a,b in combinations(range(WIDTH),2)
            for fraction in (.35,.5,.65) for strength in (.5,1.)]
    return single, np.vstack([single, *blends])


def fit_whole(plain, gut, diets, y, n):
    """Replay the V12 whole-diet main and category-specific residual updates."""
    left=left_inverse(plain[:n])
    a=plain@(left@y[:n])
    z=residualize(diets["whole_diet"],plain,left,n)
    tr=z[:n]
    b=a+z@np.linalg.solve(tr.T@tr+.10*n*np.eye(tr.shape[1]),tr.T@(y[:n]-a[:n]))
    diet_left=left_inverse(tr)
    intercept_error=y[:n]-b[:n]
    on=[]
    for kind in DIRECTIONS:
        g=diets[kind]
        x=(g[:,:,None]*gut[:,None,:]).reshape(len(g),-1)
        design=residualize(residualize(x,plain,left,n),z,diet_left,n)
        h=design[:n]
        coef=np.linalg.solve(h.T@h+n*np.eye(h.shape[1]),h.T@intercept_error)
        on.append(b[n:]+.5*(design[n:]@coef))
    return a[n:], b[n:], np.stack(on,axis=-1)


def matched_predictions(study, fold, diets, values, groups):
    bg=study.backgrounds[fold]
    family=np.unique(study.family,return_inverse=True)[1]
    support,plain=context_design(study,fold,"UPF")
    role=getattr(study,"active",np.ones(len(study.family),dtype=bool))
    diet,_=panels(diets,study,fold,support)
    target=clip_outcomes(values,bg.role)
    for observed,cols in groups:
        match=match_rows(support & observed & role,bg.role,family)
        if match is None:
            continue
        rows,n,codes,counts=match
        require(np.all(role[rows]) and not np.any(bg.role[rows[n:]]),"V13_HOUSEHOLD_ALIGNMENT")
        outcome=target[np.ix_(rows,cols)]
        a,b,c=fit_whole(plain[rows],bg.pcs[rows,p.N_PC:p.N_PC+2],
                        {k:diet[k][rows] for k in DIRECTIONS},outcome,n)
        require(all(np.isfinite(z).all() for z in (a,b,c)),"V13_NONFINITE_PREDICTIONS")
        yield cols, outcome[n:], a,b,c,codes,counts


def inner_profile(study, fold, diets, values, groups):
    result=np.zeros((values.shape[1],WIDTH+1,WIDTH))
    covered=np.zeros(values.shape[1],dtype=bool)
    for cols,y,_,b,on,_,_ in matched_predictions(study,fold,diets,values,groups):
        sst=np.sum((y-y.mean(0))**2,axis=0)
        usable=sst>1e-10*np.maximum(np.finfo(float).tiny,np.sum(y*y,axis=0))
        if not usable.any():
            continue
        j=cols[usable]
        d=on[:,usable,:]-b[:,usable,None]
        residual=y[:,usable]-b[:,usable]
        scale=100./sst[usable]
        result[j,0,:]=np.einsum("nk,nki->ki",residual,d)*scale[:,None]
        result[j,1:,:]=np.einsum("nki,nkj->kij",d,d)*scale[:,None,None]
        covered[j]=True
    return result,covered


def gains(profile, candidates):
    """Evaluate each weight using matched validation SSE differences."""
    return 2.*(profile[:,0,:]@candidates.T)-np.einsum(
        "ci,fij,cj->fc",candidates,profile[:,1:,:],candidates,optimize=True)


def choose(splits, weights, minimum, consistency, rank=None):
    objective=splits.mean(0)
    valid=(splits.min(0)>=consistency)&(objective>=minimum)
    if not np.any(valid):
        return np.zeros(WIDTH)
    score=objective if rank is None else rank
    return weights[np.argmax(np.where(valid,score,-np.inf))]


def make_policy(profiles, eligible, domains, sources):
    single, all_combinations=candidate_weights()
    fixed_full=np.eye(WIDTH)[0]
    whole=np.zeros((len(domains),WIDTH))
    mix=np.zeros_like(whole)
    guarded=np.zeros_like(whole)
    evidence={}
    for domain in sorted(set(domains)):
        belongs=domains==domain
        used=belongs & eligible
        if used.sum()<3:
            evidence[domain]=dict(outcomes=int(used.sum()),source_groups=0,whole={},mix={},guard_on=0)
            continue
        by_source=[profiles[:,used & (sources==name)].mean(axis=1) for name in sorted(set(sources[used]))]
        prior=np.mean(by_source,axis=0)
        w=choose(gains(prior,single[::WIDTH]),single[::WIDTH],.01,.003)
        m=choose(gains(prior,all_combinations),all_combinations,.01,.003)
        whole[belongs]=w
        mix[belongs]=m
        local=np.vstack((single, m))
        prior_gain=gains(prior,local).mean(0)
        for index in np.flatnonzero(used):
            score=gains(profiles[:,index],local)
            rank=.8*score.mean(0)+.2*prior_gain
            guarded[index]=choose(score,local,.025,.005,rank)
        evidence[domain]=dict(outcomes=int(used.sum()),source_groups=len(by_source),
                              whole={k:float(v) for k,v in zip(DIRECTIONS,w,strict=True) if v>0.},
                              mix={k:float(v) for k,v in zip(DIRECTIONS,m,strict=True) if v>0.},
                              guard_on=int(np.any(guarded[belongs]>0.,axis=1).sum()))
    common=np.broadcast_to(fixed_full,(len(domains),WIDTH))
    return (common,.5*common,whole,mix,guarded,np.zeros_like(guarded)), evidence


def collect(study, fold, diets, values, groups, policies=None):
    n_methods=1 if policies is None else len(METHODS)
    receipt=np.zeros((n_methods,values.shape[1],10))
    for cols,y,a,b,on,codes,counts in matched_predictions(study,fold,diets,values,groups):
        if policies is None:
            predicted=[on[:,:,0]]
        else:
            updates=on-b[:,:,None]
            predicted=[on[:,:,0]]+[b+np.einsum("nki,ki->nk",updates,w[cols]) for w in policies[1:]]
        for j,c in enumerate(predicted):
            receipt[j,cols]=np.column_stack(fold_moments(y,(a,b,c),codes,counts))
    require(np.isfinite(receipt).all(),"V13_NONFINITE_FOLD_MOMENTS")
    require(np.array_equal(receipt[:,:,:6],np.repeat(receipt[:1,:,:6],n_methods,axis=0)),
            "V13_MAIN_BASELINE_CHANGED")
    return receipt


def evaluate(work, sites, calibration, diets, values, domains, sources, seed, emit):
    from v3_inputs import InnerContexts

    verified=outcome_groups(values)
    contexts=InnerContexts(work,sites)
    folds,plans=[],[]
    for outer in p.FOLDS:
        checks=[]
        for offset in (1,2):
            inner=(outer+offset)%len(p.FOLDS)
            ctx=contexts.split(outer,inner)
            ctx,_=calibration.inner(ctx,inner,100*seed+10*outer+inner)
            bg=ctx.backgrounds[inner]
            ctx.backgrounds[inner]=replace(bg,supported=bg.supported & ctx.active)
            require(np.array_equal(ctx.active & ~bg.role,work.fold==inner) and
                    not np.any(bg.role & (work.fold==outer)),"V13_INNER_OUTER_OVERLAP")
            checks.append(inner_profile(ctx,inner,diets,values,verified))
            emit("V13_CALIBRATION",dict(seed=seed,outer=outer,inner=inner,covered=int(checks[-1][1].sum())))
        prof=np.stack([z[0] for z in checks])
        eligible=checks[0][1] & checks[1][1]
        policy,evidence=make_policy(prof,eligible,domains,sources)
        scored=collect(work,outer,diets,values,verified,policy)
        folds.append(scored)
        plans.append(dict(fold=outer,common_calibration_outcomes=int(eligible.sum()),selection=evidence))
        emit("V13_OUTER",dict(seed=seed,fold=outer,covered=int((scored[0,:,0]>0).sum())))
    return np.stack(folds,axis=2),plans


def score_partition(metrics, endpoints, labels):
    people, households, sy, sy2, a,b,c,dsq,dn,n2=metrics.sum(axis=2).transpose(2,0,1)
    require(np.all(metrics[:,:,:,:6]==metrics[:1,:,:,:6]),"V13_SCORE_BASELINE_CHANGED")
    eligible=(metrics[0,:,:,0]>0.).all(axis=-1)
    n=people[0]
    sst=sy2[0]-np.divide(sy[0]**2,n,out=np.zeros_like(n),where=n>0.)
    eligible &= sst>1e-10*np.maximum(np.finfo(float).tiny,np.abs(sy2[0]))
    ix=np.flatnonzero(eligible)
    pp=100./sst[ix]
    rows=[]
    for j,name in enumerate(labels):
        h=households[j,ix]
        difference=b[j,ix]-c[j,ix]
        avg=difference/n[ix]
        se=np.sqrt(np.maximum(0.,(dsq[j,ix]-2.*avg*dn[j,ix]+n2[j,ix]*avg**2)*h/(h-1.)))
        t_stat=np.divide(difference,se,out=np.zeros_like(difference),where=se>0.)
        probabilities=t.sf(t_stat,h-1.)
        for l,u in enumerate(ix):
            rows.append(dict(diet=name,endpoint=endpoints[u],people=int(n[u]),families=int(h[l]),
                             diet_main_pp=float((a[j,u]-b[j,u])*pp[l]),interaction_pp=float(difference[l]*pp[l]),
                             diet_total_pp=float((a[j,u]-c[j,u])*pp[l]),
                             vs_fixed_pp=float((c[0,u]-c[j,u])*pp[l]),
                             positive_folds=int((metrics[j,u,:,5]>metrics[j,u,:,6]).sum()),
                             p_one_sided=float(probabilities[l])))
    holm(rows)
    return rows


def selection_counts(plans):
    return {d:dict(Counter(k for f in plans for k in f["selection"][d]["mix"]))
            for d in plans[0]["selection"]}
