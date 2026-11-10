"""Finite-basis partial regression with household leverage-adjusted influence scores.

These are conditional-association working-model coefficients. Held-out outcomes
fit inference nuisance coefficients, separately from untouched OOF prediction.
"""

import numpy as np
from scipy import linalg, stats

from features import TARGET_NAMES


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1.0)
    return result


def partial_inference(y_residual, target_residual, groups, iqr, draws=1999, seed=20261112, minimum_groups=30, main_blocks=None, minimum_effective_groups=30):
    y, x = np.asarray(y_residual, float), np.asarray(target_residual, float)
    n, p = x.shape
    if y.shape != (n,) or p != 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid ecological partial-regression inputs")
    singular = np.linalg.svd(x, compute_uv=False)
    if singular[-1] < max(singular[0], 1) * 1e-8:
        return dict(status="UNAVAILABLE", reason="Ecological target rank/conditioning failure")
    cluster_count = len(set(np.asarray(groups).astype(str)))
    if cluster_count < minimum_groups or n <= p + 20:
        return dict(status="UNAVAILABLE", reason="Too few independent households")
    inverse = linalg.inv(x.T @ x, check_finite=False)
    theta = inverse @ (x.T @ y)
    residual = y - x @ theta
    group_labels = np.asarray(groups).astype(str)
    _, group_index = np.unique(group_labels, return_inverse=True)
    # A fixed-design concentration diagnostic, invariant to marker units. It is
    # not a substitute df or a power estimate: many nominal households may leave
    # a coordinate supported by only a few informative families.
    weights = x @ inverse
    information = np.zeros((cluster_count, p))
    np.add.at(information, group_index, weights ** 2)
    effective_groups = information.sum(0) ** 2 / (information ** 2).sum(0)
    information_note = dict(
        effective_households_per_coordinate=effective_groups.tolist(),
        minimum_effective_households=float(minimum_effective_groups),
        effective_household_definition="(sum_g sum_i w_ij^2)^2 / sum_g (sum_i w_ij^2)^2; w=X_residual (X_residual'X_residual)^-1; diagnostic, not df/power",
    )
    if np.min(effective_groups) < minimum_effective_groups:
        return dict(status="UNAVAILABLE", reason="Too few effective households for a fixed ecological coordinate",
                    n=n, households=cluster_count, **information_note)
    # H = blockdiag(Q_f Q_f') + residual_T A^-1 residual_T'. Each household
    # must be contained in one fold. CR2-style working-identity adjustment uses
    # (I-H_gg)^(-1/2), accounting for nuisance leverage rather than only q=4.
    main_rows = {}
    nuisance_rank = 0
    if main_blocks:
        coverage = np.concatenate([np.asarray(block["rows"], int) for block in main_blocks])
        if not np.array_equal(np.sort(coverage), np.arange(n)):
            raise ValueError("Inference nuisance blocks must partition participant rows")
    for block in main_blocks or []:
        rows, q = np.asarray(block["rows"], int), np.asarray(block["q"], float)
        if q.shape[0] != len(rows) or not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-7):
            raise ValueError("Invalid inference nuisance projection")
        if not np.allclose(q.T @ x[rows], 0, atol=1e-7 * max(np.linalg.norm(x[rows]), 1)):
            raise ValueError("Target columns were not partialled against the nuisance basis")
        nuisance_rank += q.shape[1]
        for group in np.unique(group_labels[rows]):
            mask = group_labels[rows] == group
            if group in main_rows:
                raise ValueError("A household crossed inference folds")
            main_rows[group] = (rows[mask], q[mask])
    if main_blocks and (len(main_rows) != cluster_count or sum(len(v[0]) for v in main_rows.values()) != n):
        raise ValueError("Inference nuisance blocks do not cover all households exactly once")
    scores, maximum_leverage = [], 0.0
    for group in np.unique(group_labels):
        rows = np.flatnonzero(group_labels == group)
        xg, eg = x[rows], residual[rows]
        h = xg @ inverse @ xg.T
        if main_blocks:
            indices, qg = main_rows[group]
            if not np.array_equal(indices, rows):
                raise ValueError("Nuisance projection household row order mismatch")
            h += qg @ qg.T
        values, vectors = linalg.eigh(np.eye(len(rows)) - (h + h.T) / 2, check_finite=False)
        maximum_leverage = max(maximum_leverage, float(1 - values.min()))
        if values.min() < -1e-7:
            return dict(status="UNAVAILABLE", reason="Invalid household leverage")
        supported = values > 1e-9
        if (~supported).any():
            null = vectors[:, ~supported]
            if np.linalg.norm(null.T @ eg) > 1e-6 or np.linalg.norm(null.T @ xg) > 1e-6:
                return dict(status="UNAVAILABLE", reason="Nonestimable high-leverage household")
        inverse_roots = np.divide(1.0, np.sqrt(np.maximum(values, 1e-300)), out=np.zeros_like(values), where=supported)
        adjusted_error = vectors @ (inverse_roots * (vectors.T @ eg))
        scores.append(xg.T @ adjusted_error)
    scores = np.asarray(scores)
    correction = cluster_count / (cluster_count - 1)
    covariance = correction * inverse @ (scores.T @ scores) @ inverse
    eig = np.linalg.eigvalsh(covariance)
    if eig[0] <= max(eig[-1], 1e-20) * 1e-10:
        return dict(status="UNAVAILABLE", reason="Household covariance rank/conditioning failure")
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], (draws, cluster_count))
    perturbations = np.sqrt(correction) * (signs @ scores) @ inverse
    df = cluster_count - 1
    critical = stats.t.ppf(0.975, df)
    se = np.sqrt(np.diag(covariance))

    def test(columns):
        columns = np.asarray(columns, int)
        v = covariance[np.ix_(columns, columns)]
        vi = linalg.inv(v, check_finite=False)
        estimate = theta[columns]
        statistic = float(estimate @ vi @ estimate)
        simulated = np.einsum("bi,ij,bj->b", perturbations[:, columns], vi, perturbations[:, columns])
        return dict(
            wald=statistic,
            df=len(columns),
            p_wald=float(stats.f.sf(statistic / len(columns), len(columns), df)),
            p_multiplier=float((1 + (simulated >= statistic).sum()) / (draws + 1)),
        )

    coordinates = [test([j]) for j in range(p)]
    # An initial fixed-covariance multivariate multiplier Wald test was mildly
    # liberal in a nonlinear null calibration. Keep it diagnostic only. Before
    # native outcomes are accessed, fix a conservative union test as primary.
    coordinate_p = [max(v["p_wald"], v["p_multiplier"]) for v in coordinates]
    omnibus = test(range(p))
    omnibus.update(p_primary=min(1.0, p * min(coordinate_p)),
                    primary_method="Bonferroni union over four coordinate max(t/F, score-multiplier) p-values")
    sites = {"oral": test([0, 1]), "gut": test([2, 3])}
    for site, columns in (("oral", [0, 1]), ("gut", [2, 3])):
        sites[site]["p_primary"] = min(1.0, 2 * min(coordinate_p[j] for j in columns))
    for site, adjusted in zip(("oral", "gut"), holm([sites[s]["p_primary"] for s in ("oral", "gut")])):
        sites[site]["p_holm_sites"] = float(adjusted)
    adjusted = holm(coordinate_p)
    family_critical = stats.t.ppf(1 - 0.05 / (2 * p), df)
    markers = []
    for j, name in enumerate(TARGET_NAMES):
        markers.append(dict(
            name=name, coefficient=float(theta[j]), se=float(se[j]),
            coefficient_ci=[float(theta[j] - critical * se[j]), float(theta[j] + critical * se[j])],
            training_reference_iqr=float(iqr[j]),
            interaction_iqr_difference_mg_dl=float(theta[j] * iqr[j]),
            interaction_iqr_ci_mg_dl=[float((theta[j] - critical * se[j]) * iqr[j]), float((theta[j] + critical * se[j]) * iqr[j])],
            interaction_iqr_ci_bonferroni95_mg_dl=[float((theta[j] - family_critical * se[j]) * iqr[j]), float((theta[j] + family_critical * se[j]) * iqr[j])],
            p_primary=float(coordinate_p[j]), p_wald=coordinates[j]["p_wald"],
            p_multiplier=coordinates[j]["p_multiplier"], p_holm_markers=float(adjusted[j]),
        ))
    return dict(
        status="OK", n=n, households=cluster_count, omnibus=omnibus, sites=sites, markers=markers,
        covariance=covariance.tolist(), score_multiplier_draws=draws,
        coefficient_units="mg/dL per 5 g/1000 kcal per raw marker unit",
        nuisance_rank=nuisance_rank, residual_df=n - nuisance_rank - p,
        maximum_household_hat_eigenvalue=maximum_leverage,
        **information_note,
        leverage_adjustment="CR2-style working-identity household residual adjustment; asymptotic score multipliers",
        scope="Exploratory fold-stratified finite-basis partial-regression association; approximate household influence inference",
        interval_scope="Conditional on target definitions, fixed feature maps, working-model adequacy and reference IQRs; historical adaptive search not covered",
    )


def prediction_metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.shape != prediction.shape or not np.isfinite(y + prediction).all() or np.var(y) <= 0:
        raise ValueError("Invalid OOF evaluation")
    mse = float(np.mean((y - prediction) ** 2))
    return dict(R2=1 - mse / float(np.var(y)), RMSE=float(np.sqrt(mse)), n=len(y))


def paired_interval(y, base, prediction, groups, draws=999, seed=20261112):
    """Resample households, preserving within-household and matched-model pairs.

    Models and OOF values are fixed. This interval does not refit the pipeline.
    """
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inverse.max()) + 1
    columns = np.column_stack([np.ones(len(y)), y, y ** 2, (y - base) ** 2, (y - prediction) ** 2])
    totals = np.zeros((g, 5))
    np.add.at(totals, inverse, columns)
    rng, increments = np.random.default_rng(seed), []
    for _ in range(draws):
        n, sy, sy2, sse0, sse1 = totals[rng.integers(0, g, g)].sum(0)
        sst = sy2 - sy * sy / n
        if sst > 0:
            increments.append(float((sse0 - sse1) / sst))
    if len(increments) < 0.95 * draws:
        raise ValueError("Degenerate household prediction bootstrap")
    return np.quantile(increments, [0.025, 0.975]).tolist()
