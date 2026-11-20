"""Matched whole-diet models and conditional category-by-microbiome comparisons."""

import numpy as np
from scipy.stats import t

import protocol as p
from batch_fit import clip_outcomes, context_design, fold_moments, match_rows, outcome_groups
from run_expanded_grid import holm
from run_v2 import require
from taxonomy import GROUPS, RECORDING, diet_key


BASES = {**GROUPS, "recording": RECORDING,
         "animal_without_red_meat": tuple(x for x in GROUPS["animal_foods"] if x!="red_meat")}
METHODS = {
    **{k:(k,k) for k in GROUPS},
    "recording": ("recording","recording"),
    "whole_diet": ("whole_diet","whole_diet"),
    "whole_diet_with_recording": ("whole_diet_with_recording","whole_diet_with_recording"),
    **{k+"_given_whole_diet":("whole_diet",k) for k in GROUPS},
    "animal_without_red_meat_given_whole_diet": ("whole_diet","animal_without_red_meat"),
}


def block(raw, fit, n):
    supported=[]
    for name in raw:
        x=raw[name]
        use=fit & np.isfinite(x)
        if use.sum()<max(120,.15*n):
            continue
        lo,hi=np.quantile(x[use],[.01,.99])
        y=np.clip(x,lo,hi)
        sd=float(y[use].std())
        if sd>1e-10 and min((y[use]>lo).sum(),(y[use]<hi).sum())>=20:
            supported.append((name,np.where(np.isfinite(y),(y-y[use].mean())/sd,0.),int(use.sum())))
    require(bool(supported),"V12_EMPTY_DIET_CATEGORY")
    labels,values,counts=zip(*supported,strict=True)
    z=np.column_stack(values)
    z-=z[fit].mean(0)
    values,vectors=np.linalg.eigh(z[fit].T@z[fit]/n)
    leading=np.argsort(values)[::-1]
    keep=leading[values[leading]>=.05*values.sum()][:3]
    require(len(keep)>0,"V12_DEGENERATE_CATEGORY_BASIS")
    axes=vectors[:,keep]
    largest=np.argmax(np.abs(axes),axis=0)
    axes*=np.sign(axes[largest,np.arange(len(keep))])
    transformed=(z@axes)/np.sqrt(len(keep)*values[keep])
    audit=dict(features=list(labels),training_observed=list(counts),components=len(keep),
               variance_fraction=float(values[keep].sum()/values.sum()),loadings=axes.tolist())
    return transformed,audit


def panels(diets, study, fold, support):
    bg=study.backgrounds[fold]
    fit=support & bg.role
    n=int(fit.sum())
    raw={diet_key(name):bg.archived["UPF"][:,0] if name=="UPF" else val for name,_,val in diets}
    scores,audit={},{}
    for name,features in BASES.items():
        scores[name],audit[name]=block({x:raw[x] for x in features},fit,n)
    whole=list(GROUPS)
    scores["whole_diet"]=np.column_stack([scores[k] for k in whole])
    scores["whole_diet_with_recording"]=np.column_stack([scores["whole_diet"],scores["recording"]])
    return scores,audit


def residualize(matrix, nuisance, left, n):
    return matrix-nuisance@(left@matrix[:n])


def left_inverse(a):
    return np.linalg.pinv(a.T@a,rcond=p.RANK_TOL,hermitian=True)@a.T


def fit_categories(plain, pcs, designs, y, n):
    """Only the interaction changes for methods sharing a given diet main."""
    left=left_inverse(plain[:n])
    a=plain@(left@y[:n])
    basic=y[:n]-a[:n]
    mains={}
    modifiers={}
    predictions=[]
    for main_name,group_name in METHODS.values():
        if main_name not in mains:
            z=residualize(designs[main_name],plain,left,n)
            tr=z[:n]
            penalty=tr.T@tr+.10*n*np.eye(tr.shape[1])
            b=a+z@np.linalg.solve(penalty,tr.T@basic)
            mains[main_name]=b,z,left_inverse(tr)
        if group_name not in modifiers:
            g=designs[group_name]
            products=(g[:,:,None]*pcs[:,None,:]).reshape(len(g),-1)
            modifiers[group_name]=residualize(products,plain,left,n)
        b,main_terms,main_left=mains[main_name]
        z=residualize(modifiers[group_name],main_terms,main_left,n)
        h=z[:n]
        cov=h.T@h+n*np.eye(h.shape[1])
        c=b[n:]+.5*(z[n:]@np.linalg.solve(cov,h.T@(y[:n]-b[:n])))
        predictions.append((a[n:],b[n:],c))
    return predictions


def accumulate(study, diets, values, progress):
    family=np.unique(study.family,return_inverse=True)[1]
    groups=outcome_groups(values)
    metrics=np.zeros((len(METHODS),values.shape[1],len(p.FOLDS),10))
    audit=[]
    for j,fold in enumerate(p.FOLDS):
        bg=study.backgrounds[fold]
        y=clip_outcomes(values,bg.role)
        support,plain=context_design(study,fold,"UPF")
        diet,sizes=panels(diets,study,fold,support)
        count=0
        for observed,cols in groups:
            matched=match_rows(support & observed,bg.role,family)
            if matched is None:
                continue
            rows,n,codes,counts=matched
            target=y[np.ix_(rows,cols)]
            design={k:v[rows] for k,v in diet.items()}
            gut=bg.pcs[rows,p.N_PC:p.N_PC+2]
            predictions=fit_categories(plain[rows],gut,design,target,n)
            for i,q in enumerate(predictions):
                moment=np.column_stack(fold_moments(target[n:],q,codes,counts))
                require(np.isfinite(moment).all(),"V12_GROUP_NONFINITE_SCORE")
                metrics[i,cols,j]=moment
            count+=len(cols)
        audit.append(dict(fold=fold,eligible_endpoints=count,categories=sizes))
        progress(fold,count)
    return metrics,audit


def score_partition(metrics,endpoints):
    people,households,sy,sy2,a,b,c,dsq,dn,n2=metrics.sum(axis=2).transpose(2,0,1)
    require(np.all(people==people[:1]) and np.all(households==households[:1]),"V12_GROUP_COHORT_DIFFERS")
    require(np.all(a==a[:1]),"V12_NO_DIET_SCORE_DIFFERS")
    labels=list(METHODS)
    fixed=[labels.index(m) for m,(k,_) in METHODS.items() if k=="whole_diet"]
    require(np.all(b[fixed]==b[fixed[0]]),"V12_WHOLE_DIET_MAIN_CHANGED")
    eligible=(metrics[0,:,:,0]>0.).all(axis=-1)
    n=people[0]
    sst=sy2[0]-np.divide(sy[0]**2,n,out=np.zeros_like(n),where=n>0)
    eligible &= sst>1e-10*np.maximum(np.finfo(float).tiny,abs(sy2[0]))
    rows=[]
    for j,label in enumerate(labels):
        k=np.flatnonzero(eligible)
        h=households[j,k]
        diff=b[j,k]-c[j,k]
        avg=diff/people[j,k]
        se=np.sqrt(np.maximum(0., (dsq[j,k]-2.*avg*dn[j,k]+avg**2*n2[j,k])*h/(h-1.)))
        stat=np.divide(diff,se,out=np.zeros_like(diff),where=se>0.)
        significance=t.sf(stat,h-1.)
        for i,u in enumerate(k):
            rows.append(dict(diet=label,endpoint=endpoints[u],people=int(people[j,u]),families=int(h[i]),
                             diet_main_pp=100.*float((a[j,u]-b[j,u])/sst[u]),
                             interaction_pp=100.*float(diff[i]/sst[u]),diet_total_pp=100.*float((a[j,u]-c[j,u])/sst[u]),
                             positive_folds=int((metrics[j,u,:,5]>metrics[j,u,:,6]).sum()),
                             p_one_sided=float(significance[i])))
    holm(rows)
    return rows,int(eligible.sum())
