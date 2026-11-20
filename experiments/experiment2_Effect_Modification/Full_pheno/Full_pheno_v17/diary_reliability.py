"""Estimate dietary split-half agreement without using clinical outcomes."""

from types import SimpleNamespace
from unittest.mock import patch


def read_halves(root, study, expected_names, day_file):
    import numpy as np
    import pandas as pd
    import batch_sources
    import diet_inventory
    from food_features import DAY, normalise, projected
    from run_v2 import require
    from taxonomy import GROUPS, diet_key

    index, ids = batch_sources.index_for(study)
    day = normalise(projected(
        day_file, DAY + ["energy_kcal", "energy_eligible", "fiber_g", "fiber_eligible"], ids
    )).merge(index, on=DAY[:-1], validate="many_to_one")
    require(not day.duplicated(DAY).any(), "V17_DUPLICATE_DIARY_DAY")
    day = day.loc[day.energy_eligible.eq(True) & day.energy_kcal.gt(0.)].sort_values(["input_row", "date"]).copy()
    by_person = day.groupby("input_row")
    day["half"] = (by_person.cumcount() >= by_person.date.transform("size") // 2).astype(int)
    number = [np.bincount(day.loc[day.half.eq(i), "input_row"], minlength=study.n) for i in (0, 1)]
    expected = study.numeric["n_energy_days"]
    require(np.array_equal((number[0] + number[1])[expected >= 3], expected[expected >= 3]),
            "V17_DIARY_DAY_SUPPORT_CHANGED")
    halves = []
    for i in (0, 1):
        half = day.loc[day.half.eq(i)]
        key = half[DAY]
        fiber = half.loc[half.fiber_eligible.eq(True) & half.fiber_g.ge(0.)].copy()
        fiber["density"] = 1000. * fiber.fiber_g / fiber.energy_kcal
        fiber_days = np.bincount(fiber.input_row, minlength=study.n)
        density = fiber.groupby("input_row").density.mean().reindex(range(study.n)).to_numpy()
        density[fiber_days < 3] = np.nan
        numeric = dict(study.numeric, n_energy_days=number[i],
                       n_fiber_days=fiber_days, fiber_density=density)
        current = SimpleNamespace(keys=study.keys, n=study.n, numeric=numeric, backgrounds=study.backgrounds)
        cache = {}

        def project(path, cols, participants=None):
            id_ = (path, tuple(cols))
            if id_ not in cache:
                f = projected(path, cols, participants)
                if all(c in cols for c in DAY):
                    f = normalise(f).merge(key, on=DAY, validate="many_to_one")
                cache[id_] = f
            return cache[id_].copy()

        with patch.object(batch_sources, "projected", project), patch.object(diet_inventory, "projected", project):
            recipes, _, files = batch_sources.read_diets(root, current)
        require(files["day"] == day_file, "V17_UNEXPECTED_DIARY_SOURCE")
        halves.append({diet_key(k): x for k, _, x in recipes if k in expected_names and k != "UPF"})
    paired = {key: {f: np.column_stack([h[f] for h in halves])
                    for f in names if all(f in h for h in halves)} for key, names in GROUPS.items()}
    require(all(paired.values()), "V17_MISSING_DIET_HALF_CLASS")
    return paired, number


def reliability_weights(paired, days, training):
    import warnings
    import numpy as np
    from scipy.stats import rankdata
    from taxonomy import GROUPS

    coverage = np.clip((np.nan_to_num(days, nan=0.) - 3.) / 7., 0., 1.)
    scores, provenance = [], {}
    for category in GROUPS:
        differences, correlations = [], []
        for feature, values in paired[category].items():
            mask = np.isfinite(values).all(axis=1)
            fit = mask & training
            if fit.sum() < 120:
                continue
            percentiles = np.quantile(values[fit].ravel(), [.05, .95])
            scale = float(percentiles[1] - percentiles[0])
            if scale <= 1e-10:
                continue
            ranks = rankdata(values[fit], axis=0)
            if (ranks.std(axis=0) == 0.).any():
                continue
            r = float(np.corrcoef(ranks.T)[0, 1])
            correlations.append(dict(feature=feature, people=int(fit.sum()), split_half_r=r))
            change = np.abs(np.diff(np.clip(values, *percentiles), axis=1)[:, 0]) / scale
            change[~mask] = np.nan
            differences.append(change)
        if differences:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                disagreement = np.nanmedian(np.column_stack(differences), axis=1)
            repeatability = np.where(np.isfinite(disagreement), 1. - disagreement, 1.)
            available = int(np.isfinite(disagreement).sum())
        else:
            repeatability = np.ones(len(days))
            available = 0
        scores.append(.5 + .5 * coverage * repeatability)
        provenance[category] = dict(paired_people=available, features=correlations)
    individual = np.column_stack(scores)
    average = individual[training].mean(axis=0)
    return individual, average, .5 + .5 * coverage, provenance
