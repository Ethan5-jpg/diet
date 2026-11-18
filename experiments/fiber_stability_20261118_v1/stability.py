"""Household-only partitions, paired model selection and error attribution."""
import numpy as np

ROUTES = ("off","fixed","spread","transfer","both","gate")
TERMS = ("spread_alignment","spread_cost","transfer_alignment","transfer_cost","overlap")
NEW_REPEATS = 8
SEED = 2026111804


def partition(family, seed):
    labels, inv = np.unique(family,return_inverse=True)
    assigned = np.empty(len(labels),int)
    assigned[np.random.default_rng(seed).permutation(len(labels))] = np.arange(len(labels))%5
    return assigned[inv]


def by_family(a,family,n):
    return np.column_stack([np.bincount(family,weights=c,minlength=n) for c in a.T])


def choose(loss):
    """Simplest option within one paired household SE of the minimum inner OOF SSE."""
    if loss.shape[0]<2 or loss.shape[1]!=4 or not np.isfinite(loss).all():
        raise ValueError("Invalid inner household losses")
    total = loss.sum(axis=0)
    best = total.argmin()
    se = (loss-loss[:,best,None]).std(axis=0,ddof=1)*np.sqrt(len(loss))
    valid = np.flatnonzero(total-total[best] <= se+1e-9)
    complexity = (0,1,1,2)
    return int(min(valid,key=lambda j:(complexity[j],total[j],j)))


def checkpoint(y,mean_train,pred,family,n,gate):
    """pred columns: no interaction, fixed panel, spread, transfer, both."""
    full = np.column_stack((pred,pred[:,1+gate]))
    err = y[:,None]-full
    resid = err[:,1]
    s,t = (pred[:,j]-pred[:,1] for j in (2,3))
    terms = np.column_stack((2*resid*s,-s*s,2*resid*t,-t*t,-2*s*t))
    np.testing.assert_allclose(terms.sum(axis=1),resid**2-err[:,4]**2,atol=1e-7,rtol=1e-7)
    data = np.column_stack((np.ones(len(y)),(y-mean_train)**2,err**2,abs(err),terms))
    return by_family(data,family,n)


def summarize(repeats, weights, formal):
    """All repeats reuse the same household bootstrap weights; no independent-repeat assumption."""
    repeat_rows, sampled, decomposition = [], [], []
    for a in repeats:
        total = weights @ a
        n, v = total[:,0], total[:,1]
        if min(n)<=0 or min(v)<=0 or not np.isfinite(total).all():
            raise ValueError("Invalid validation denominators")
        r2 = 1-total[:,2:8]/v[:,None]
        mae = total[:,8:14]/n[:,None]
        repeat_rows.append(dict(r2=r2[0].tolist(),mae=mae[0].tolist(),
                                relative_pct=(100*(r2[0]-r2[0,0])/r2[0,0]).tolist() if r2[0,0]>0 else None))
        sampled.append((r2-r2[:,1,None],mae[:,1,None]-mae))
        decomposition.append(total[0,14:]/v[0])
    delta, mg = (np.mean(m,axis=0) for m in zip(*sampled))
    terms = dict(zip(TERMS,np.mean(decomposition,axis=0).tolist()))
    np.testing.assert_allclose(sum(terms.values()),delta[0,4],atol=1e-9)
    results = {}
    for j in (2,3,4,5):
        gains = [row["r2"][j]-row["r2"][1] for row in repeat_rows]
        positive = sum(v>1e-12 for v in gains)
        ci = np.quantile(delta[1:,j],[.025,.975]).tolist()
        # Two candidate policies (frozen or nested) by two directions.
        simultaneous = np.quantile(delta[1:,j],[.00625,.99375]).tolist()
        passed = bool(formal and j in (4,5) and positive>=7 and mg[0,j]>=0 and simultaneous[0]>0)
        # Remove the 1% of households with largest absolute contributions, refitting no models.
        contribution = np.mean([(a[:,3]-a[:,2+j])/a[:,1].sum() for a in repeats],axis=0)
        active = repeats[0][:,0]>0
        ids = np.flatnonzero(active)
        removed = ids[np.argsort(-abs(contribution[ids]),kind="stable")[:max(1,int(np.ceil(.01*len(ids))))]]
        keep = active.copy()
        keep[removed] = False
        trimmed = np.mean([(a[keep,3]-a[keep,2+j]).sum()/a[keep,1].sum() for a in repeats])
        results[ROUTES[j]] = dict(delta=float(delta[0,j]),mae_gain=float(mg[0,j]),ci95=ci,simultaneous=simultaneous,
                                  wins=int(positive),minimum=float(min(gains)),maximum=float(max(gains)),
                                  household_better=float((contribution[active]>1e-14).mean()),
                                  drop_top_1pct=float(trimmed),stable=passed)
    return dict(repeats=repeat_rows,results=results,components=terms)
