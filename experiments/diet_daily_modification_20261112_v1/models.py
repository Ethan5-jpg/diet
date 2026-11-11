"""Equal-person weighted daily fixed effects and household holdout comparisons."""

import numpy as np
from scipy import stats


DIETS = ("carb", "fiber")
CONTRAST = {"carb": (45, 55), "fiber": (10, 15)}
SCALE = {"carb": 10, "fiber": 5}


def center(values, people):
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(people, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), values.shape[1]))
    np.add.at(sums, inverse, values)
    return values - sums[inverse] / counts[inverse, None], 1 / counts[inverse]


def scaled_dose(diet, values):
    return (np.asarray(values) - np.mean(CONTRAST[diet])) / SCALE[diet]


def background(daily):
    dose = {d: scaled_dose(d, daily[d]) for d in DIETS}
    p = (daily.protein.to_numpy(float) - 15) / 10
    energy = daily.energy_kcal.to_numpy(float) / 1000
    a = (daily.age_calendar_years.to_numpy(float) - 50) / 10
    sex = daily.sex_code.to_numpy(float)
    weekday = daily.date.dt.day_of_week.to_numpy()
    day = (daily.date - daily.groupby("input_row").date.transform("min")).dt.days.to_numpy()
    b = np.column_stack([dose["carb"], dose["fiber"], p, energy, day,
                         dose["carb"] ** 2, dose["fiber"] ** 2, p ** 2,
                         dose["carb"] * dose["fiber"], dose["carb"] * p,
                         *(d * c for d in [*dose.values(), p] for c in (a, sex)),
                         *(weekday == i for i in range(1, 7))])
    y = daily.Y_value.to_numpy(float)
    centered, weights = center(np.column_stack([y, b]), daily.input_row.to_numpy())
    return centered[:, 0], centered[:, 1:], weights


def train_standardize(scores, training):
    mean = scores[training].mean(axis=0)
    sd = scores[training].std(axis=0)
    if not (np.isfinite(sd).all() and (sd > 1e-12).all()):
        raise ValueError("Degenerate training microbial representation")
    return (scores - mean) / sd


def site_scores(raw, taxa, index, population, training, clr, width):
    features = {}
    for site in ("oral", "gut"):
        abundance = raw[site][population]
        if not (np.isfinite(abundance).all() and (abundance >= 0).all()
                and (abundance.sum(axis=1) > 0).all()):
            raise ValueError(f"Unsupported raw {site} profiles in daily cohort")
        relative = abundance / abundance.sum(axis=1, keepdims=True)
        logarithm = np.log(relative, out=np.zeros_like(relative), where=relative > 0)
        shannon = -(relative * logarithm).sum(axis=1, keepdims=True)
        features[site + "_shannon"] = train_standardize(shannon, training)
        ids = index.iloc[population[training]].participant_id.tolist()
        fitted = clr().fit(abundance[training], taxa[site], ids, site=site)
        transformed = fitted.transform(abundance, taxa[site], site=site)
        if not transformed["supported"].all():
            raise ValueError(f"Fold-specific CLR has unsupported {site} profiles")
        z = transformed["values"]
        mean = z[training].mean(axis=0)
        _, s, axes = np.linalg.svd(z[training] - mean, full_matrices=False)
        if len(s) < width or s[width - 1] <= 1e-8:
            raise ValueError(f"Insufficient {site} training CLR rank")
        features[site + "_pc"] = train_standardize((z - mean) @ axes[:width].T, training)
    features["both_pc"] = np.column_stack([features["oral_pc"], features["gut_pc"]])
    features["both_shannon"] = np.column_stack([features["oral_shannon"], features["gut_shannon"]])
    return features


def design(daily, diets, daily_scores, curve):
    cols = []
    contrasts = {}
    offset = 0
    width = daily_scores.shape[1]
    for diet in diets:
        t = scaled_dose(diet, daily[diet])
        terms = [t, t ** 2] if curve else [t]
        cols.extend(v[:, None] * daily_scores for v in terms)
        low, high = scaled_dose(diet, CONTRAST[diet])
        contrasts[diet] = (slice(offset, offset + width), high - low)
        offset += width * len(terms)
    centered, _ = center(np.column_stack(cols), daily.input_row.to_numpy())
    return centered, contrasts


def fit_background(x, y, weights):
    sw = np.sqrt(weights)
    wx = x * sw[:, None]
    wy = y * sw
    u, s, vt = np.linalg.svd(wx, full_matrices=False)
    rank = int((s > s[0] * 1e-10).sum())
    u = u[:, :rank]
    inverse = vt[:rank].T / s[:rank]
    return inverse @ (u.T @ wy), u, inverse, sw, wy


def orthogonal_block(z, fitted):
    _, u, inverse, sw, wy = fitted
    wz = z * sw[:, None]
    rotation = u.T @ wz
    return wz - u @ rotation, wy - u @ (u.T @ wy), inverse @ rotation


def fit_interaction(x, z, y, weights, train, valid, fitted, penalty):
    base, _, _, _, _ = fitted
    zr, yr, projection = orthogonal_block(z[train], fitted)
    gram = zr.T @ zr + penalty * weights[train].sum() * np.eye(z.shape[1])
    beta = np.linalg.solve(gram, zr.T @ yr)
    null = x[valid] @ base
    correction = (z[valid] - x[valid] @ projection) @ beta
    error_null = np.sum(weights[valid] * (y[valid] - null) ** 2)
    error_alt = np.sum(weights[valid] * (y[valid] - null - correction) ** 2)
    return beta, float(error_null), float(error_alt)


def household_test(z, fitted, households, n_people):
    zr, yr, _ = orthogonal_block(z, fitted)
    beta, _, rank, _ = np.linalg.lstsq(zr, yr, rcond=1e-10)
    if rank != z.shape[1]:
        raise ValueError("Rank-deficient tested interaction block")
    groups, inverse = np.unique(households, return_inverse=True)
    scores = np.zeros((len(groups), z.shape[1]))
    np.add.at(scores, inverse, zr * (yr - zr @ beta)[:, None])
    bread = np.linalg.inv(zr.T @ zr)
    residual_df = len(yr) - n_people - fitted[1].shape[1] - rank
    if residual_df <= 0 or len(groups) <= rank:
        raise ValueError("Insufficient residual or household degrees of freedom")
    correction = len(groups) / (len(groups) - 1) * (len(yr) - 1) / residual_df
    covariance = correction * bread @ (scores.T @ scores) @ bread
    test = float(beta @ np.linalg.solve(covariance, beta) / rank)
    p = float(stats.f.sf(test, rank, len(groups) - 1))
    return beta, covariance, p


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted
