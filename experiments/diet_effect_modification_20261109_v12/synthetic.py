"""Synthetic fixtures only; this module never opens participant input files."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features import rcs


def folds(groups, k, seed):
    unique = np.array(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(unique)
    lookup = {value: j % k for j, value in enumerate(unique)}
    return np.array([lookup[value] for value in groups])


def separated(groups, train, valid):
    if train.dtype != bool or valid.dtype != bool or (train & valid).any() or set(groups[train]) & set(groups[valid]):
        raise ValueError("Household overlap")


class Dataset:
    def __init__(self, n=360, seed=19, taxa_count=18):
        rng = np.random.default_rng(seed)
        groups = np.array(["synthetic_household_%d" % j for j in np.arange(n) // 2])
        diet = np.clip(rng.normal(12, 4, n), 2, 25)
        age = rng.normal(45, 10, n)
        sex = rng.integers(0, 2, n)
        energy = rng.normal(2100, 250, n)
        self.index = pd.DataFrame(dict(
            split_unit_id=groups, fiber_density=diet, age_calendar_years=age,
            sex_code=sex, mean_daily_energy_fiber=energy,
            UPF_lower_equalday_fraction=rng.uniform(0, 0.25, n),
            UPF_upper_equalday_fraction=rng.uniform(0.3, 0.6, n),
        ))
        self.raw = {site: rng.dirichlet(np.linspace(0.3, 1.4, taxa_count), n) for site in ("oral", "gut")}
        self.taxa = {site: ["synthetic_%s_%d" % (site, j) for j in range(taxa_count)] for site in ("oral", "gut")}
        context = rng.normal(size=(n, 7))
        context[rng.random(context.shape) < 0.05] = np.nan
        self.extra = dict(blocks={"full": context}, alcohol=np.array(["never" if j % 3 else None for j in range(n)], object), audit={"synthetic": True})
        household = rng.normal(0, 2, (n + 1) // 2)[np.arange(n) // 2]
        self.y = 95 - 0.5 * diet + 0.08 * age + 12 * self.raw["oral"][:, 0] + household + rng.normal(0, 3, n)
        outer = np.stack([folds(groups, 5, 7), folds(groups, 5, 20261104)])
        self.reference = dict(y=np.stack([self.y.copy(), self.y.copy()]), outer=outer)
        self.api, self.paths = SimpleNamespace(folds=folds, separated=separated), []

    def split(self, train, valid, contrast):
        groups = self.index.split_unit_id.to_numpy()
        separated(groups, train, valid)
        t, v = np.flatnonzero(train), np.flatnonzero(valid)
        diet = self.index.fiber_density.to_numpy()
        center, scale = float(np.median(diet[t])), 5.0
        knots = (np.quantile(diet[t], [0.1, 0.5, 0.9]) - center) / scale
        u = (diet - center) / scale
        h = rcs(u, knots)
        numeric_raw = self.index[["age_calendar_years", "sex_code", "mean_daily_energy_fiber"]].to_numpy()
        means, scales = numeric_raw[t].mean(0), numeric_raw[t].std(0)
        numeric = (numeric_raw - means) / scales
        host = np.column_stack([np.ones(len(diet)), numeric, u, h, u * numeric[:, 0], u * numeric[:, 1]])
        microbes, states = {}, {}
        for site, a in self.raw.items():
            logs = np.log(np.maximum(a, 1e-10))
            clr = logs - logs.mean(1, keepdims=True)
            transform = StandardScaler().fit(clr[t])
            microbes[site] = transform.transform(clr)
            states[site] = dict(output_indices=list(range(a.shape[1])))
        state = dict(
            diet=dict(center=center, scale=scale, knots=knots), CLR=states,
            host=dict(labels=["intercept", "age", "sex", "energy", "diet_linear", "diet_RCS", "diet_age", "diet_sex"],
                      numeric_means=means, numeric_scales=scales),
        )
        return SimpleNamespace(
            ht=host[t], hv=host[v], ot=microbes["oral"][t], ov=microbes["oral"][v],
            gt=microbes["gut"][t], gv=microbes["gut"][v], yt=self.y[t], yv=self.y[v],
            dt=np.column_stack([u[t], h[t]]), dv=np.column_stack([u[v], h[v]]),
            train_rows=t, rows=v, preprocessing=state,
        )
