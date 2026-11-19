"""Paired held-out prediction scores and descriptive modification contrasts."""

import numpy as np
from scipy.stats import t

import protocol as p
from models import create_design
from source import check


CONTRASTS = {
    "panel_minus_off": ("off", "three_species", False),
    "shannon_increment": ("three_species", "three_species_plus_shannon", True),
    "all_interactions": ("off", "three_species_plus_shannon", True),
}
PREDICTION_SCOPE = "conditional_on_fixed_out_of_fold_predictions"


def interval(values, alpha=p.ALPHA):
    return np.quantile(values, (alpha / 2, 1 - alpha / 2)).tolist()


def loss_table(y, candidates, families):
    y = np.asarray(y, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    families = np.asarray(families)
    check(y.ndim == 1 and len(y) > 0 and families.shape == y.shape and candidates.ndim == 2 and
          candidates.shape[0] == len(y) and np.isfinite(y).all() and np.isfinite(candidates).all(),
          "INVALID_SCORING_ROWS")
    ids, g = np.unique(families, return_inverse=True)
    yc = y - y.mean()
    err = y[:, None] - candidates
    arrays = np.column_stack((np.ones(len(y)), yc, yc**2, err**2, abs(err)))
    return np.column_stack([np.bincount(g, weights=c, minlength=len(ids)) for c in arrays.T])


def resampled_totals(table, draws, seed):
    g = len(table)
    check(g >= 2 and draws >= 1, "INSUFFICIENT_HOUSEHOLD_BOOTSTRAP")
    rng = np.random.default_rng(seed)
    sampled = [table.sum(axis=0)[None, :]]
    probabilities = np.full(g, 1.0 / g)
    for start in range(0, draws, 128):
        w = rng.multinomial(g, probabilities, size=min(128, draws - start))
        sampled.append(w @ table)
    return np.vstack(sampled)


def prediction(y, candidates, families, draws, seed, names=p.PREDICTORS, pairs=CONTRASTS):
    check(np.ndim(candidates) == 2 and np.shape(candidates)[1] == len(names) and
          len(set(names)) == len(names), "SCORING_COLUMNS_DO_NOT_MATCH_MODELS")
    totals = resampled_totals(loss_table(y, candidates, families), draws, seed)
    n, sy, sy2 = totals[:, :3].T
    sst = sy2 - sy**2 / n
    keep = np.isfinite(sst) & (sst > 0)
    if not keep[0]:
        return dict(status="unavailable", reason="OUTCOME_VARIANCE_ZERO", intervals_fixed_predictions=True)
    keep_ci = bool(keep.all())
    r2 = 1 - totals[keep, 3:3 + len(names)] / sst[keep, None]
    mae = totals[keep, 3 + len(names):] / n[keep, None]
    result = dict(status="ok", bootstrap_draws=draws, valid_bootstrap_draws=int(keep[1:].sum()),
                  interval_status="ok" if keep_ci else "bootstrap_contains_degenerate_outcomes",
                  intervals_fixed_predictions=True, inference_scope=PREDICTION_SCOPE, models={}, pairs={})
    for i, name in enumerate(names):
        result["models"][name] = dict(r2=float(r2[0, i]), mae=float(mae[0, i]),
                                       r2_ci95=interval(r2[1:, i]) if keep_ci else None)
    for label, (base, enhanced, adjusted) in pairs.items():
        i, j = names.index(base), names.index(enhanced)
        # Use the same SST and each draw's paired loss difference.
        delta = (totals[keep, 3 + i] - totals[keep, 3 + j]) / sst[keep]
        ratio_ok = bool(keep_ci and np.all(r2[:, [i, j]] > p.R2_EPS))
        ratio = 100 * delta / r2[:, j] if ratio_ok else None
        result["pairs"][label] = dict(
            base=base, enhanced=enhanced, delta_r2=float(delta[0]),
            delta_ci95=interval(delta[1:]) if keep_ci else None,
            delta_simultaneous95=interval(delta[1:], p.ALPHA / p.FAMILY_TESTS) if keep_ci and adjusted else None,
            share_pct=None if ratio is None else float(ratio[0]),
            share_ci95=None if ratio is None else interval(ratio[1:]),
            share_status="stable_positive_scores" if ratio_ok else "withheld_nonpositive_or_unstable_r2",
            improvement_mae=float(mae[0, i] - mae[0, j]),
        )
    return result


def support(amount, marker, q25, q75, doses, tolerance):
    def tail(mask):
        a = amount[mask]
        lo, hi = np.quantile(a, (.05, .95))
        counts = [int(np.count_nonzero(abs(a - target) <= tolerance)) for target in doses]
        return dict(n=len(a), q05_q95=[float(lo), float(hi)], near_doses=counts,
                    covers=bool(lo <= doses[0] < doses[1] <= hi and min(counts) > 0))
    low, high = tail(marker <= q25), tail(marker >= q75)
    return dict(low=low, high=high, supported=bool(q75 > q25 and low["covers"] and high["covers"]))


def cluster_contrasts(x, y, families, contrasts):
    """CR1 sandwich on estimable linear contrasts; t reference uses G - 1 df."""
    _, groups = np.unique(families, return_inverse=True)
    g = int(groups.max()) + 1
    scale = np.linalg.norm(x, axis=0)
    # Do not amplify numerical noise to a unit column. Genuinely estimable
    # signals (the inputs here are already in standardized model units) can
    # still be rescaled for the SVD.
    weak = scale <= p.RANK_TOL * np.sqrt(len(y))
    scale[weak] = 1.
    u, s, vt = np.linalg.svd(x / scale, full_matrices=False)
    k = int(np.count_nonzero(s > s[0] * p.RANK_TOL))
    u, s, vt = u[:, :k], s[:k], vt[:k]
    inv = (vt.T / s) @ u.T
    c = contrasts / scale
    cnorm = np.linalg.norm(c, axis=1)
    estimable = ((cnorm > 0) & ~np.any((c != 0) & weak, axis=1)
                 & (np.linalg.norm(c - (c @ vt.T) @ vt, axis=1) <= (1e-8 * cnorm)))
    residual = y - u @ (u.T @ y)
    estimates = c @ inv @ y
    if g < 2 or len(y) <= k:
        return k, [{"status": "unavailable", "reason": "INSUFFICIENT_DF"} for _ in c]
    psi = (residual[None, :] * (c @ inv))
    cluster = np.stack([np.bincount(groups, weights=row, minlength=g) for row in psi])
    correction = (g / (g - 1)) * ((len(y) - 1) / (len(y) - k))
    ses = np.sqrt(correction * (cluster**2).sum(axis=1))
    ref = (float(t.ppf(1 - p.ALPHA / 2, g - 1)),
           float(t.ppf(1 - p.ALPHA / (2 * p.FAMILY_TESTS), g - 1)))
    result = []
    for est, se, ok in zip(estimates, ses, estimable, strict=True):
        if not ok or not np.isfinite(est) or not np.isfinite(se) or se <= 0:
            result.append(dict(status="unavailable", reason="NONESTIMABLE_OR_ZERO_VARIANCE"))
            continue
        cis = [[float(est - q * se), float(est + q * se)] for q in ref]
        result.append(dict(status="ok", estimate=float(est), se=float(se), ci95=cis[0],
                           simultaneous_ci95=cis[1]))
    return k, result


def association(study, exposure, rows, outcome):
    doses, tol = (p.FIBER_DOSE, 1.) if exposure == "fiber_density" else (p.UPF_DOSE, .02)
    result = dict(unit=outcome.unit, predictors="joint_association_on_out_of_fold_scored_people",
                  exposure_contrast=list(doses), unit_exposure="g/1000 kcal" if exposure == "fiber_density" else "fraction",
                  endpoint_scale="original outcome; participants retained by archived outcome rules",
                  moderation_dose="linear_and_fixed_hinge" if exposure == "fiber_density" else "linear_calibrated_fraction")
    bg = study.backgrounds[None]
    missing = ~bg.supported[rows] | ~np.isfinite(bg.archived[exposure][rows]).all(axis=1)
    if missing.any():
        # Preserve valid paired predictions. A different complete-case association
        # sample would no longer describe the people used in the prediction table.
        return dict(result, rank=None, full_background_unavailable=int(missing.sum()), markers={
            label: dict(status="unavailable", reason="FULL_DISCOVERY_BACKGROUND_NOT_AVAILABLE_ON_PAIRED_ROWS")
            for label in p.MARKERS
        })
    design = create_design(study, exposure, None, rows, len(rows), association=True)
    matrix = np.column_stack((design.main, design.panel, design.shannon))
    nmain, npanel = design.main.shape[1], design.panel.shape[1]
    raw = study.markers[rows]
    quartiles = np.quantile(raw, (.25, .75), axis=0)
    lo, hi = quartiles
    linear = (hi - lo) / design.marker_scales
    bend = (np.maximum(hi - design.marker_medians, 0) -
            np.maximum(lo - design.marker_medians, 0)) / design.marker_scales
    contrasts = np.zeros((2, matrix.shape[1]))
    contrasts[0, nmain:nmain + npanel] = np.outer(design.dose_difference, [linear[0], bend[0]]).ravel()
    contrasts[1, nmain + npanel:] = design.dose_difference * linear[1]
    k, evaluations = cluster_contrasts(matrix, outcome.y[rows], study.family[rows], contrasts)
    values = {}
    for j, marker in enumerate(p.MARKERS):
        evidence = support(design.exposure_values, raw[:, j], lo[j], hi[j], doses, tol)
        estimate = evaluations[j] if evidence["supported"] else dict(status="unavailable", reason="DOSE_OR_MARKER_SUPPORT")
        values[marker] = dict(q25=float(lo[j]), q75=float(hi[j]), **estimate, support=evidence)
    return dict(result, rank=k, markers=values)
