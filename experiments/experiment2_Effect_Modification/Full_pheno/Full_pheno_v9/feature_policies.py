"""Training-only feature selection and arm fallback."""

import numpy as np


BLOCKS = ("nutrients", "fiber_sources", "food_groups", "processing", "day_patterns", "meal_context")
FIXED = ("v8_reference", *BLOCKS, "all_main", "all_main_and_interaction")
METHODS = (*FIXED,"train_select","train_shrink","train_stop","historical_stop")


def pick(parts):
    """All decisions use paired scores on inner-held families."""
    y = np.concatenate([f.outcome for f in parts])
    sst = np.sum((y-y.mean())**2)
    if not np.isfinite(sst) or sst<=0.:
        raise ValueError("V9_DEGENERATE_INNER_OUTCOMES")
    rows = np.asarray([np.sum((f.outcome-f.no_diet)**2) for f in parts])[:,None]/sst
    off = np.stack([np.sum((f.outcome[:,None]-f.off)**2,axis=0) for f in parts])/sst
    on = np.stack([np.sum((f.outcome[:,None]-f.on)**2,axis=0) for f in parts])/sst
    total = rows-on
    interaction = off-on
    better = on[:,[0]]-on
    delta = interaction-interaction[:,[0]]
    qualified = (better.sum(axis=0)>=.0002) & (np.count_nonzero(better>0.,axis=0)>=3)
    qualified &= (delta.sum(axis=0)>=-.0001) & (total.sum(axis=0)>0.)
    j = int(np.argmax(np.where(qualified,better.sum(axis=0),-np.inf))) if qualified.any() else 0
    net, effect = total[:,j],interaction[:,j]
    base = rows[:,0]-off[:,j]

    def recurrent_negative(x):
        return x.sum()<=-.0002 and np.count_nonzero(x<0.)>=3

    if recurrent_negative(net) and recurrent_negative(base):
        drop = "diet"
    elif recurrent_negative(effect):
        drop = "interaction"
    else:
        drop = "keep"
    return j,drop


def adjusted(baseline, off, on, drop, fraction):
    if drop=="diet":
        return baseline+fraction*(off-baseline),baseline+fraction*(on-baseline)
    if drop=="interaction":
        return off,off+fraction*(on-off)
    return off,on


def extend(fit, j, stop, historical):
    off,on = fit.off[:,j],fit.on[:,j]
    new = [(off,on), adjusted(fit.no_diet,off,on,stop,.5), adjusted(fit.no_diet,off,on,stop,0.),
           adjusted(fit.no_diet,fit.off[:,0],fit.on[:,0],historical,0.)]
    a,b = zip(*new,strict=True)
    return type(fit)(fit.rows,fit.outcome,fit.no_diet,np.column_stack((fit.off,*a)),np.column_stack((fit.on,*b)))
