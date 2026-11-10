"""Validated original inputs plus joint-day bags; never read daily outcomes."""

from pathlib import Path

import numpy as np

from bridge import ContextCore as core

KEY, DAY, FOOD, LIFE, ALCOHOL = core.KEY, core.DAY, core.FOOD, core.LIFE, core.ALCOHOL
NUTRIENTS, ANCHORS = core.NUTRIENTS, core.ANCHORS
projected_read, pq = core.projected_read, core.pq


def aggregate(food, nutrients, lifestyle, index):
    extra = core.aggregate(food, nutrients, lifestyle, index)
    admitted = food.energy_eligible.eq(True).fillna(False).to_numpy(bool, na_value=False)
    source = core.normalize(nutrients)
    source = source.loc[source.date.notna()]
    frame = core.normalize(food.loc[admitted, FOOD]).merge(
        source[DAY + [n + "_complete_sum" for n in NUTRIENTS[:3]]], on=DAY, how="left", validate="one_to_one"
    )
    energy = frame.energy_kcal.to_numpy(float, na_value=np.nan)
    x = frame[[n + "_complete_sum" for n in NUTRIENTS[:3]]].to_numpy(float, na_value=np.nan)
    x = x * 1000 / energy[:, None]
    good = np.isfinite(x).all(1) & (x >= 0).all(1)
    lookup = {k: j for j, k in enumerate(index[KEY].itertuples(index=False, name=None))}
    owners = np.array([lookup[k] for k in frame[KEY].itertuples(index=False, name=None)], int)[good]
    days = np.column_stack([x[good], np.log(energy[good])])
    count = np.bincount(owners, minlength=len(index))
    variance, covariance = np.full((len(index), 3), np.nan), np.full((len(index), 6), np.nan)
    for row in np.flatnonzero(count >= 3):
        values = days[owners == row, :3]
        cov = np.cov(values, rowvar=False, ddof=1)
        variance[row] = np.maximum(np.diag(cov), 0)
        scale = np.sqrt(variance[row])
        correlations = [cov[a, b] / max(scale[a] * scale[b], 1e-12) for a, b in ((0, 1), (0, 2), (1, 2))]
        covariance[row] = [*scale, *np.clip(correlations, -1, 1)]
    extra.update(days=days, owners=owners, joint_count=count, joint_variance=variance, covariance=covariance)
    extra["audit"].update(joint_distribution_days=len(days), distribution_people_ge3=int((count >= 3).sum()))
    return extra


def read_context(package, index, eligible):
    package = Path(package)
    pop = index.loc[eligible].copy()
    paths = [
        package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet",
        package.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet",
        package.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet",
    ]
    columns = [FOOD, DAY + ANCHORS + [n + "_complete_sum" for n in NUTRIENTS], KEY + LIFE + [ALCOHOL]]
    frames = [core.projected_read(p, c, pop) for p, c in zip(paths, columns)]
    extra, positions = aggregate(*frames, pop), np.flatnonzero(eligible)
    extra["owners"] = positions[extra["owners"]]
    for name, value in extra["blocks"].items():
        full = np.full((len(index), value.shape[1]), np.nan)
        full[positions] = value
        extra["blocks"][name] = full
    for name in ("joint_variance", "covariance"):
        value = extra[name]
        full = np.full((len(index), value.shape[1]), np.nan)
        full[positions] = value
        extra[name] = full
    count, alcohol = np.zeros(len(index), int), np.full(len(index), None, object)
    count[positions], alcohol[positions] = extra["joint_count"], extra["alcohol"]
    extra["joint_count"], extra["alcohol"] = count, alcohol
    return extra, paths
