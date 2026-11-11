"""Exposure overlap and a zero-aware microbial genus representation."""

import numpy as np

GENUS = "g__Lachnoclostridium"
BREAD = ("bread_wholewheat", "bread")


def genus_fraction(species, taxa):
    mask = np.array([GENUS in path.split("|") for path in taxa])
    if (species.ndim != 2 or species.shape[1] != len(taxa) or len(set(taxa)) != len(taxa)
            or not mask.any() or not np.isfinite(species).all() or (species < 0).any()
            or not np.allclose(species.sum(axis=1), 1., atol=.001, rtol=0)):
        raise ValueError("Unexpected archived gut abundance contract")
    leaves = [path for path, member in zip(taxa, mask) if member]
    if any(not path.split("|")[-1].startswith("s__") for path in leaves):
        raise ValueError("Genus path includes an overlapping aggregate")
    return species[:, mask].sum(axis=1), len(leaves)


def zero_aware(fraction, training):
    detected = fraction > 0
    positive = detected & training
    rate = detected[training].mean()
    logs = np.log(fraction[positive])
    spread = logs.std() if len(logs) >= 2 else 0.
    if not (0 < rate < 1 and spread > 1e-12):
        raise ValueError("Insufficient training genus detection or abundance variation")
    detection_sd = np.sqrt(rate * (1 - rate))
    detection = (detected - rate) / detection_sd
    abundance = np.zeros(len(fraction))
    abundance[detected] = (np.log(fraction[detected]) - logs.mean()) / spread
    return np.column_stack([detection, abundance]), 1 / detection_sd


def comparable_pairs(meals):
    """Choose the closest eligible pair per person using logged diet and time."""
    chosen = []
    for _, group in meals.sort_values("start").groupby("input_row", sort=False):
        wheat = group.loc[group.bread_wholewheat.ge(10) & group.bread_wholewheat.gt(group.carbs / 2)]
        bread = group.loc[group.bread.ge(10) & group.bread.gt(group.carbs / 2)]
        same_day = wheat.date.to_numpy()[:, None] == bread.date.to_numpy()
        distance = np.zeros(same_day.shape)
        overlap = ~same_day
        for name, cap in (("carbs", 10.), ("energy", 100.), ("protein", 10.), ("local_hour", 3.)):
            gap = np.abs(wheat[name].to_numpy()[:, None] - bread[name].to_numpy())
            if name == "local_hour":
                gap = np.minimum(gap, 24 - gap)
            overlap &= gap <= cap
            distance += (gap / cap) ** 2
        if overlap.any():
            i, j = np.unravel_index(np.argmin(np.where(overlap, distance, np.inf)), distance.shape)
            chosen.extend((wheat.index[i], bread.index[j]))
    return meals.loc[chosen].copy()


def covariates(meals, score, person, center, model):
    whole, bread = (meals[name].to_numpy(float) for name in BREAD)
    exchange = (whole - bread) / 20
    pair = (whole + bread) / 10
    total = (meals.carbs.to_numpy(float) - 40) / 10
    age = (meals.age_calendar_years.to_numpy(float) - 50) / 10
    sex = meals.sex_code.to_numpy(float)
    z = score[person]
    controls = np.column_stack([exchange, pair, exchange ** 2, pair ** 2, exchange * pair, pair * total,
                                exchange * total, exchange * age, exchange * sex,
                                meals.fruits.to_numpy(float) / 10, meals.unlabelled.to_numpy(float) / 10,
                                *(z * d[:, None] for d in (total, total ** 2, pair, pair ** 2))])
    return model.center(controls, center)[0], model.center(z * exchange[:, None], center)[0]
