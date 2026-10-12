"""Participant-level estimating equations for dependent model contrasts."""
import numpy as np
import pandas as pd
from scipy import stats

from .common import require


def inference(beta, se, df):
    require(np.isfinite(beta) and np.isfinite(se) and se >= 0 and df > 0, "Invalid contrast inference")
    if se <= 1e-14:
        require(abs(beta) <= 1e-12, "Nonzero estimate with degenerate standard error")
        return dict(beta=float(beta), se=0.0, low=float(beta), high=float(beta), p=1.0)
    crit = stats.t.ppf(.975, df)
    return dict(beta=float(beta), se=float(se), low=float(beta - crit * se),
                high=float(beta + crit * se), p=float(2 * stats.t.sf(abs(beta / se), df)))


def fit_linear(d, outcome, columns, focal, within, equal_person=True, min_df=30, max_condition=1e8):
    """FE/CR1 for within; equal-person OLS/HC1 for between.

    ``influence`` contains finite-sample-corrected cluster estimating-equation
    contributions, not leave-one-out estimates. It allows covariance-aware
    contrasts between models with overlapping participant samples.
    """
    require(len(set(columns)) == len(columns), "Duplicate design columns")
    group, labels = pd.factorize(d.participant_id, sort=True)
    n, g = len(d), len(labels)
    require(g >= 3, "Too few participant clusters")
    if not within:
        require(n == g, "Between model must have one row per participant")
    values = d[[outcome] + columns].astype(float).reset_index(drop=True)
    require(np.isfinite(values.to_numpy()).all(), "Nonfinite design/outcome")
    counts = np.bincount(group)
    w = 1.0 / counts[group] if within and equal_person else np.ones(n)
    if within:
        sw = pd.Series(w).groupby(group).transform("sum")
        means = values.mul(w, axis=0).groupby(group).transform("sum").div(sw, axis=0)
        values = values - means
    else:
        sw = pd.Series(np.ones(n))
    y = values[outcome].to_numpy() * np.sqrt(w)
    X = values[columns].to_numpy() * np.sqrt(w[:, None])
    scale = np.sqrt(np.mean(X ** 2, axis=0))
    active = scale > 1e-10
    # Between constants are absorbed by the explicit intercept, not rank failure.
    if not within:
        require("intercept" in columns, "Between model requires intercept")
        active &= np.array([c == "intercept" or np.ptp(X[:, i]) > 1e-10 for i, c in enumerate(columns)])
    kept = [c for c, ok in zip(columns, active) if ok]
    require(set(focal).issubset(kept), "No residual variation in a focal regressor")
    X, scale = X[:, active], scale[active]
    Z = X / scale
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    k = len(kept)
    condition = float(s[0] / s[-1]) if len(s) and s[-1] > 0 else np.inf
    require(len(s) == k and np.linalg.matrix_rank(Z) == k and condition < max_condition,
            "Collinear/ill-conditioned design")
    residual_df = n - g - k if within else n - k
    require(residual_df >= min_df, "Insufficient residual degrees of freedom")
    bz = vt.T.dot(u.T.dot(y) / s)
    residual = y - Z.dot(bz)
    bread = (vt.T / s ** 2).dot(vt)
    scores = np.zeros((g, k))
    np.add.at(scores, group, Z * residual[:, None])
    correction = g / (g - 1) * (n - 1) / residual_df
    influence = (scores.dot(bread) / scale) * np.sqrt(correction)
    covariance = influence.T.dot(influence)
    beta = bz / scale
    require(np.isfinite(covariance).all(), "Invalid covariance")
    coefficients = {}
    for col, b, var in zip(kept, beta, np.diag(covariance)):
        require(var > 1e-28, "Degenerate coefficient standard error")
        coefficients[col] = inference(b, np.sqrt(var), g - 1)
    focal_shares = {}
    for name in focal:
        j = kept.index(name)
        others = np.delete(Z, j, axis=1)
        partial = Z[:, j] - others.dot(np.linalg.lstsq(others, Z[:, j], rcond=None)[0])
        info = np.bincount(group, weights=partial ** 2)
        require(info.sum() > 1e-16, "No partial focal support")
        focal_shares[name] = float(info.max() / info.sum())
    leverage = np.sum(u ** 2, axis=1)
    if within:
        leverage += w / sw.to_numpy()
    return dict(coefficients=coefficients, names=kept, covariance=covariance,
                influence=pd.DataFrame(influence, index=pd.Index(labels, name="participant_id"), columns=kept),
                n_rows=n, n_people=g, residual_df=residual_df, cluster_df=g - 1,
                condition=condition, max_leverage=float(leverage.max()),
                focal_information_shares=focal_shares, correction=correction,
                dropped_constants=";".join(c for c, ok in zip(columns, active) if not ok))


def contrast(fit, weights):
    require(set(weights).issubset(fit["names"]), "Unknown contrast term")
    beta = sum(fit["coefficients"][k]["beta"] * v for k, v in weights.items())
    influence = sum((fit["influence"][k] * v for k, v in weights.items()),
                    pd.Series(0.0, index=fit["influence"].index))
    return inference(beta, np.sqrt(influence.dot(influence)), fit["cluster_df"])


def paired_contrast(left, right, left_weights, right_weights):
    """Return left minus right, retaining covariance of shared participants."""
    require(set(left_weights).issubset(left["names"]) and set(right_weights).issubset(right["names"]),
            "Unknown paired contrast term")
    a = sum((left["influence"][k] * v for k, v in left_weights.items()),
            pd.Series(0.0, index=left["influence"].index))
    b = sum((right["influence"][k] * v for k, v in right_weights.items()),
            pd.Series(0.0, index=right["influence"].index))
    delta = a.subtract(b, fill_value=0)
    estimate = (sum(left["coefficients"][k]["beta"] * v for k, v in left_weights.items()) -
                sum(right["coefficients"][k]["beta"] * v for k, v in right_weights.items()))
    result = inference(estimate, np.sqrt(delta.dot(delta)), min(left["cluster_df"], right["cluster_df"]))
    result["overlap_people"] = len(a.index.intersection(b.index))
    result["cross_covariance"] = float(a.mul(b, fill_value=0).sum())
    return result
