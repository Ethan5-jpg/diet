"""Sufficient statistics and matched predictors for repeated meal slopes."""

import numpy as np


def sums(person, dose, residual, outcome, weights, count):
    """Each row describes one person's observations in one period."""
    values = (
        dose ** 2, dose * residual, weights * dose ** 2,
        weights * dose * residual, weights * residual ** 2, weights * outcome ** 2,
    )
    return np.column_stack([np.bincount(person, weights=x, minlength=count) for x in values])


def variance_moments(early, late, info):
    """Fit a shared slope variance and period-specific inverse-dose noise."""
    covariance = np.cov(early, late, ddof=1)
    variance = np.diag(covariance)
    stable = float(np.clip(covariance[0, 1], 0, .95 * variance.min()))
    noise = np.maximum(variance - stable, 1e-12) / (1 / info).mean(axis=0)
    return stable, noise


def marker_prior(score, response, training, weight, penalty=.03):
    x = score[training]
    weight = weight / weight.sum()
    center = weight @ x
    mean = float(weight @ response[training])
    design = x - center
    gram = design.T @ (weight[:, None] * design)
    beta = np.linalg.solve(gram + penalty * np.eye(design.shape[1]),
                           design.T @ (weight * (response[training] - mean)))
    return mean, mean + (score - center) @ beta


def fold_models(score, source, target, info, training, h):
    paired = (source, target) if h == 0 else (target, source)
    stable, noise = variance_moments(paired[0][training], paired[1][training], info[training])
    noise_person = noise[h] / info[:, h]
    shrink = stable / (stable + noise_person)
    equal_mean, equal = marker_prior(score, source, training, np.ones(training.sum()))
    mean, prior = marker_prior(score, source, training, (1 / (stable + noise_person))[training])
    constant = np.full(len(source), mean)
    observed = shrink * source
    return dict(equal_null=np.full(len(source), equal_mean), equal=equal, precision_null=constant,
                precision=prior, history_null=observed + (1 - shrink) * mean,
                history_marker=observed + (1 - shrink) * prior)


CONTRASTS = {
    "equal_marker": ("equal_null", "equal"),
    "precision_marker": ("precision_null", "precision"),
    "precision_vs_equal": ("equal", "precision"),
    "history": ("precision_null", "history_null"),
    "marker_beyond_history": ("history_null", "history_marker"),
}


def family_prediction(family, data, predicted, valid, families):
    d = data[valid]
    measured = d[:, 1] / d[:, 0]
    stats = {"slope_den": measured ** 2, "meal_den": d[:, 5]}
    for name, values in predicted.items():
        b = values[valid]
        stats["slope_" + name] = (measured - b) ** 2
        stats["meal_" + name] = d[:, 4] - 2 * b * d[:, 3] + b ** 2 * d[:, 2]
    return {key: np.bincount(family[valid], weights=value, minlength=families)
            for key, value in stats.items()}


def family_repeatability(family, early, late, valid, families):
    a, b = early[valid], late[valid]
    values = (np.ones(len(a)), a, b, a * a, b * b, a * b)
    return np.column_stack([np.bincount(family[valid], weights=v, minlength=families) for v in values])


def read_repeatability(totals, draws):
    def moments(m):
        n, a, b, a2, b2, ab = m.T
        cov = (ab - a * b / n) / (n - 1)
        var0 = (a2 - a * a / n) / (n - 1)
        var1 = (b2 - b * b / n) / (n - 1)
        return cov, cov / np.sqrt(var0 * var1)

    fit = moments(totals.sum(axis=0))
    boot = moments(totals[draws].sum(axis=1))
    return {key: dict(value=float(value), ci95=np.quantile(values, [.025, .975]).tolist())
            for key, value, values in zip(("covariance", "pearson_r"), fit, boot)}


def read_prediction(totals, draws):
    result = {}
    for metric in ("slope", "meal"):
        denominator = totals[metric + "_den"]
        boot_den = denominator[draws].sum(axis=1)
        if denominator.sum() <= 0 or not (boot_den > 0).all():
            raise ValueError("Invalid matched score denominator")
        rows = {}
        for contrast, (null, extension) in CONTRASTS.items():
            gain = totals[metric + "_" + null] - totals[metric + "_" + extension]
            rows[contrast] = dict(delta_r2=float(gain.sum() / denominator.sum()),
                                  ci95=np.quantile(gain[draws].sum(axis=1) / boot_den, [.025, .975]).tolist())
        result[metric] = rows
    return result
