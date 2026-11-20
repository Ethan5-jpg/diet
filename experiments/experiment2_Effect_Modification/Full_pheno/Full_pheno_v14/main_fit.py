"""Compare expanded whole-diet main effects on exactly matched household folds."""

import numpy as np
from scipy.stats import t

import protocol as p
from batch_fit import clip_outcomes, context_design, fold_moments, match_rows, outcome_groups
from category_fit import left_inverse, panels, residualize
from run_expanded_grid import holm
from run_v2 import require
from taxonomy import GROUPS


# Each entry specifies the design, ridge penalty and post-fit contribution scale.
RECIPES = {
    "reference": ("all",.10,1.),
    "ridge_035": ("all",.35,1.),
    "ridge_100": ("all",1.00,1.),
    "one_pc": ("one_per_class",.10,1.),
    "one_pc_ridge": ("one_per_class",.35,1.),
    "half_diet": ("all",.10,.5),
    "coverage_scaled": ("all",.10,"coverage"),
}
INTERACTIONS = {"off":0.,"fixed":.5,"half":.25}
METHODS = tuple(f"{name}:{arm}" for name in RECIPES for arm in INTERACTIONS)


def predict(plain, pcs, full, compact, y, energy_days, n, recipes):
    """Keep A fixed; use the same interaction design in every main-effect arm."""
    left=left_inverse(plain[:n])
    a=plain@(left@y[:n])
    z=residualize(full,plain,left,n)
    bases={"all":z, "one_per_class":residualize(compact,plain,left,n)}
    error=y[:n]-a[:n]
    components={}
    pairs={}
    for name in recipes:
        design, penalty, amount=RECIPES[name]
        key=design,penalty
        if key not in components:
            x=bases[design]
            train=x[:n]
            components[key]=x@np.linalg.solve(train.T@train+penalty*n*np.eye(train.shape[1]),train.T@error)
        modifier=np.minimum(energy_days/10.,1.)[:,None] if amount=="coverage" else amount
        pairs[name]=a+modifier*components[key]
    raw=(full[:,:,None]*pcs[:,None,:]).reshape(len(full),-1)
    g=residualize(residualize(raw,plain,left,n),z,left_inverse(z[:n]),n)
    train_g=g[:n]
    pooled_errors=np.column_stack([y[:n]-b[:n] for b in pairs.values()])
    c_train=np.linalg.solve(train_g.T@train_g+n*np.eye(train_g.shape[1]),train_g.T@pooled_errors)
    update=np.split(g[n:]@c_train,len(recipes),axis=1)
    forecasts={}
    for name,effect in zip(recipes,update,strict=True):
        b=pairs[name][n:]
        for arm,scale in INTERACTIONS.items():
            c=b+scale*effect
            forecasts[f"{name}:{arm}"]=(a[n:],b,c)
    return forecasts


def subset_main(y,a,b,families):
    """Store aggregate held-out moments for an observational diary-coverage stratum."""
    return np.broadcast_arrays(len(families),len(np.unique(families)),y.sum(0),(y*y).sum(0),
                               ((y-a)**2).sum(0),((y-b)**2).sum(0))


def collect(work,diets,values,progress,recipes=tuple(RECIPES),strata=True):
    family=np.unique(work.family,return_inverse=True)[1]
    groups=outcome_groups(values)
    methods=tuple(f"{m}:{a}" for m in recipes for a in INTERACTIONS)
    scores=np.zeros((len(methods),values.shape[1],len(p.FOLDS),10))
    coverage=np.zeros((2,values.shape[1],len(p.FOLDS),6))
    diary=work.numeric["n_energy_days"]
    for j,fold in enumerate(p.FOLDS):
        bg=work.backgrounds[fold]
        target=clip_outcomes(values,bg.role)
        support,plain=context_design(work,fold,"UPF")
        diet,_=panels(diets,work,fold,support)
        compact=np.column_stack([diet[k][:,0]*np.sqrt(diet[k].shape[1]) for k in GROUPS])
        full=diet["whole_diet"]
        count=0
        for observed,cols in groups:
            match=match_rows(support & observed,bg.role,family)
            if match is None:
                continue
            rows,n,codes,counts=match
            require(np.all(work.fold[rows[n:]]==fold) and np.all(work.fold[rows[:n]]!=fold),
                    "V14_WRONG_HELD_OUT_MEMBERSHIP")
            y=target[np.ix_(rows,cols)]
            gut=bg.pcs[rows,p.N_PC:p.N_PC+2]
            fits=predict(plain[rows],gut,full[rows],compact[rows],y,diary[rows],n,recipes)
            for i,m in enumerate(methods):
                moment=np.column_stack(fold_moments(y[n:],fits[m],codes,counts))
                require(np.isfinite(moment).all(),"V14_NONFINITE_MOMENTS")
                scores[i,cols,j]=moment
            if strata:
                a,b,_=fits["reference:off"]
                people=rows[n:]
                for i,mask in enumerate((diary[people]<10,diary[people]>=10)):
                    coverage[i,cols,j]=np.column_stack(subset_main(y[n:][mask],a[mask],b[mask],family[people][mask]))
            count+=len(cols)
        progress(fold,count)
    return scores,coverage,methods


def score_partition(metrics,endpoints,methods):
    require(np.all(metrics[:,:,:,:5]==metrics[:1,:,:,:5]),"V14_NO_DIET_OR_HELD_SAMPLE_CHANGED")
    for i in range(0,len(methods),3):
        require(np.all(metrics[i:i+3,:,:,5]==metrics[i:i+1,:,:,5]),"V14_MAIN_DIFFERS_WITH_INTERACTION")
        require(np.array_equal(metrics[i,:,:,5],metrics[i,:,:,6]),"V14_OFF_C_EQUALS_B")
    people, households, sy,sy2,a,b,c,dsq,dn,n2=metrics.sum(axis=2).transpose(2,0,1)
    n=people[0]
    sst=sy2[0]-np.divide(sy[0]**2,n,out=np.zeros_like(n),where=n>0.)
    eligible=(metrics[0,:,:,0]>0.).all(axis=-1) & (sst>1e-10*np.maximum(np.finfo(float).tiny,abs(sy2[0])))
    ix=np.flatnonzero(eligible)
    pp=100./sst[ix]
    rows=[]
    for j,m in enumerate(methods):
        h=households[j,ix]
        gain=b[j,ix]-c[j,ix]
        mean=gain/n[ix]
        se=np.sqrt(np.maximum(0.,(dsq[j,ix]-2.*mean*dn[j,ix]+n2[j,ix]*mean**2)*h/(h-1.)))
        stat=np.divide(gain,se,out=np.zeros_like(gain),where=se>0.)
        probs=t.sf(stat,h-1.)
        ref=methods.index("reference:"+m.split(":")[1])
        for k,u in enumerate(ix):
            rows.append(dict(diet=m,endpoint=endpoints[u],people=int(n[u]),families=int(h[k]),
                             diet_main_pp=float((a[j,u]-b[j,u])*pp[k]),interaction_pp=float(gain[k]*pp[k]),
                             diet_total_pp=float((a[j,u]-c[j,u])*pp[k]),
                             vs_reference_pp=float((c[ref,u]-c[j,u])*pp[k]),
                             positive_folds=int((metrics[j,u,:,5]>metrics[j,u,:,6]).sum()),p_one_sided=float(probs[k])))
    holm(rows)
    return rows


def diary_scores(metrics,endpoints):
    n,h,sy,sy2,a,b=metrics.sum(axis=2).transpose(2,0,1)
    sst=sy2-np.divide(sy**2,n,out=np.zeros_like(n),where=n>0.)
    eligible=(metrics[:,:,:,1]>=20).all(axis=-1)&(sst>1e-10*np.maximum(np.finfo(float).tiny,abs(sy2)))
    rows=[]
    for i,name in enumerate(("3_to_9_days","10_plus_days")):
        for j in np.flatnonzero(eligible[i]):
            rows.append(dict(stratum=name,endpoint=endpoints[j],families=int(h[i,j]),
                             diet_main_pp=float(100.*(a[i,j]-b[i,j])/sst[i,j])))
    return rows
