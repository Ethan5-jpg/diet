"""Fit and apply the three-assay prediction formulas on prepared model designs."""

import numpy as np


def fit(plain, gut, diet, category, y):
    from category_fit import left_inverse

    n = len(y)
    left = left_inverse(plain)
    host = left @ y
    to_diet = left @ diet
    z = diet - plain @ to_diet
    main = np.linalg.solve(z.T @ z + .10*n*np.eye(z.shape[1]), z.T @ (y - plain @ host))
    error = y - plain @ host - z @ main
    z_left = left_inverse(z)
    result = dict(host=host,to_diet=to_diet,main=main)
    for suffix, x in (("category",category),("whole",diet)):
        product = (x[:,:,None] * gut[:,None,:]).reshape(n,-1)
        host_projection = left @ product
        r = product - plain @ host_projection
        diet_projection = z_left @ r
        h = r - z @ diet_projection
        beta = np.linalg.solve(h.T @ h + n*np.eye(h.shape[1]), h.T @ error)
        result.update({f"{suffix}_host":host_projection, f"{suffix}_diet":diet_projection, f"{suffix}_beta":beta})
    return result


def predict(coef, plain, gut, diet, category):
    a = plain @ coef["host"]
    z = diet - plain @ coef["to_diet"]
    b = a + z @ coef["main"]
    forecasts = [a,b]
    for suffix, x in (("category",category),("whole",diet)):
        product = (x[:,:,None] * gut[:,None,:]).reshape(len(plain),-1)
        h = product - plain @ coef[f"{suffix}_host"] - z @ coef[f"{suffix}_diet"]
        forecasts.append(b + .25 * (h @ coef[f"{suffix}_beta"]))
    return tuple(forecasts)


def design(work, fold, diets):
    import protocol as p
    from batch_fit import context_design
    from category_fit import panels

    support, host = context_design(work, fold, "UPF")
    panel, audit = panels(diets, work, fold, support)
    gut = work.backgrounds[fold].pcs[:,p.N_PC:p.N_PC+2]
    return support, host, gut, panel, audit


def fitted_on_design(parts, work, fold, values, candidates, *, held_out):
    from run_v2 import require

    support, host, gut, panel, _ = parts
    training = work.backgrounds[fold].role
    models, reports, held = [], [], []
    for j, (group, endpoint) in enumerate(candidates):
        raw = values[:,j]
        observed = np.isfinite(raw)
        tr = support & training & observed
        family_count = len(np.unique(work.family[tr]))
        require(family_count >= 120, "V24_FIT_ASSAY_FAMILY_SUPPORT")
        low, high = np.quantile(raw[training & observed], [.01,.99])
        y = np.clip(raw,low,high)
        inputs = (host,gut,panel["whole_diet"],panel[group])
        model = fit(*(v[tr] for v in inputs), y[tr])
        require(all(np.isfinite(a).all() for a in model.values()) and np.isfinite([low,high]).all(),
                "V24_NONFINITE_COEFFICIENTS")
        models.append(model)
        reports.append(dict(endpoint=endpoint,diet=group,people=int(tr.sum()),households=family_count,
                            clip_lower=float(low),clip_upper=float(high),
                            design_widths=[m.shape[1] for m in inputs]))
        if held_out:
            q = support & ~training & observed
            require(q.any() and not np.intersect1d(work.family[q],work.family[tr]).size, "V24_REPLAY_HOUSEHOLD_OVERLAP")
            held.append((q,y[q],predict(model,*(v[q] for v in inputs))))
    return models,reports,held
