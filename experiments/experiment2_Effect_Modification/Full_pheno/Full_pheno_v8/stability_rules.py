"""Freeze the candidate menu and require paired validation for adaptations."""

import numpy as np


FIXED = ("precision_137", "pc5_half", "precision_linear_main", "precision_curve_main",
         "blended_main", "blended_main_bagged", "blended_main_pooled", "blended_main_oral",
         "precision_137_oral")
METHODS = (*FIXED, "inner_retain", "inner_joint")
MENU = ("precision_linear_main", "precision_curve_main", "blended_main", "blended_main_bagged",
        "blended_main_pooled", "blended_main_oral", "precision_137_oral")


def choose(parts, *, strict):
    """Select a bounded replacement using only predictions for inner-held families."""
    y = np.concatenate([x.outcome for x in parts])
    sst = np.sum((y-y.mean())**2)
    if not np.isfinite(sst) or sst<=0.:
        raise ValueError("V8_INNER_VARIANCE")
    reference_total = sum(np.sum((f.outcome-f.no_diet)**2-(f.outcome-f.on[:, 0])**2) for f in parts)/sst
    best = ((0., 0.), (0, 0.))
    for option in MENU:
        j = FIXED.index(option)
        for w in (.25, .50):
            off, on = [], []
            for f in parts:
                a,b = f.off[:,0], f.on[:,0]
                c,d = w*(f.off[:,j]-a), w*(f.on[:,j]-b)
                off.append(float(c@(2.*(f.outcome-a)-c))/sst)
                on.append(float(d@(2.*(f.outcome-b)-d))/sst)
            paired_delta = np.asarray(on)-np.asarray(off)
            total, interaction = float(sum(on)), float(paired_delta.sum())
            if total<.00015 or np.count_nonzero(np.asarray(on)>0.)<3 or reference_total+total<=0.:
                continue
            if strict and (interaction<.00010 or np.count_nonzero(paired_delta>0.)<3):
                continue
            if not strict and interaction<-.00005:
                continue
            quality = (total,interaction)
            if quality>best[0]:
                best = (quality,(j,w))
    return best[1]


def with_adaptations(fit, actions):
    off = [fit.off]
    on = [fit.on]
    for j,w in actions:
        off.append(fit.off[:,0] + w*(fit.off[:,j]-fit.off[:,0]))
        on.append(fit.on[:,0] + w*(fit.on[:,j]-fit.on[:,0]))
    return type(fit)(fit.rows, fit.outcome, fit.no_diet, np.column_stack(off), np.column_stack(on))
