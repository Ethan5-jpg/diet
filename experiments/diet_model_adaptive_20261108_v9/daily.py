"""Food-only variability features; explicit column projection, no daily outcomes."""

import numpy as np
import pandas as pd

KEY = ("participant_id", "cohort", "research_stage")
COLUMNS = KEY + ("date", "energy_kcal", "fiber_g", "energy_eligible", "fiber_eligible")
NAMES = (
    "fiber_sd",
    "fiber_iqr",
    "fiber_p10",
    "fiber_p90",
    "fiber_cv",
    "fiber_mean_square",
    "fiber_mean_log1p",
    "fiber_fraction_gt15",
    "energy_sd",
    "energy_cv",
    "energy_iqr",
    "fiber_energy_correlation",
    "weekend_fiber_difference",
)


def aggregate_food(frame, index):
    """No label-dependent aggregation or change to original participant eligibility."""
    if not set(COLUMNS) <= set(frame.columns):
        raise ValueError("Food daily table lacks documented columns")
    frame = frame[list(COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame[list(KEY) + ["date"]].isna().any().any() or frame.duplicated(list(KEY) + ["date"]).any():
        raise ValueError("Food identity/date keys must be nonmissing and unique")
    lookup = {k: j for j, k in enumerate(index[list(KEY)].itertuples(index=False, name=None))}
    if len(lookup) != len(index):
        raise ValueError("Participant composite identities must be unique")
    values = np.full((len(index), len(NAMES)), np.nan)
    mean_density, admitted = np.full(len(index), np.nan), np.zeros(len(index), int)
    for identity, group in frame.groupby(list(KEY), sort=False, dropna=False):
        if identity not in lookup:
            raise ValueError("Food table contains identities outside discovery")
        row = lookup[identity]
        energy, fiber = group.energy_kcal.to_numpy(float), group.fiber_g.to_numpy(float)
        good_e = group.energy_eligible.fillna(False).to_numpy(bool)
        good_f = good_e & group.fiber_eligible.fillna(False).to_numpy(bool)
        if np.any(good_e & (~np.isfinite(energy) | (energy < 500) | (energy > 4000))) or np.any(
            good_f & (~np.isfinite(fiber) | (fiber < 0))
        ):
            raise ValueError("Admitted food values contradict documented QC")
        admitted[row] = int(good_f.sum())
        if good_f.sum() < 3:
            continue
        d = 1000 * fiber[good_f] / energy[good_f]
        e = energy[good_e]
        q = np.quantile(d, [0.1, 0.25, 0.75, 0.9])
        sd, esd = float(d.std(ddof=1)), float(e.std(ddof=1))
        weekend = group.loc[good_f, "date"].dt.weekday.to_numpy() >= 5
        difference = (
            float(d[weekend].mean() - d[~weekend].mean()) if weekend.any() and (~weekend).any() else np.nan
        )
        corr = (
            float(np.corrcoef(d, energy[good_f])[0, 1])
            if d.std() > 1e-12 and energy[good_f].std() > 1e-12
            else 0.0
        )
        values[row] = [
            sd,
            q[2] - q[1],
            q[0],
            q[3],
            sd / max(float(d.mean()), 1e-8),
            np.mean(d**2),
            np.mean(np.log1p(d)),
            np.mean(d > 15),
            esd,
            esd / float(e.mean()),
            np.quantile(e, 0.75) - np.quantile(e, 0.25),
            corr,
            difference,
        ]
        mean_density[row] = d.mean()
    comparable = np.isfinite(mean_density) & np.isfinite(index.fiber_density.to_numpy(float))
    audit = dict(
        food_people_with_three_days=int((admitted >= 3).sum()),
        discovery_people=len(index),
        food_days=len(frame),
        input_features=len(NAMES),
        daily_outcome_columns_read=[],
        max_mean_density_difference=float(
            np.max(abs(mean_density[comparable] - index.fiber_density.to_numpy(float)[comparable]), initial=0)
        ),
    )
    return dict(values=values, names=NAMES, audit=audit)


def read_daily(path, index):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path, pre_buffer=False)
    if not set(COLUMNS) <= set(parquet.schema_arrow.names):
        raise ValueError("Expected the documented food-only daily schema")
    frame = parquet.read(columns=list(COLUMNS), use_threads=False).to_pandas(ignore_metadata=True)
    return aggregate_food(frame, index)
