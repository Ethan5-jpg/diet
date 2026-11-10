"""Projected discovery-only source reads and outcome-free dietary aggregation."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["participant_id", "cohort", "research_stage"]
DAY = KEY + ["date"]
MACROS = ["carbohydrate_g", "lipid_g", "protein_g"]
NUTRIENTS = MACROS + ["sodium_mg"]
ANCHORS = ["calories_kcal_complete_sum", "dietary_fiber_g_complete_sum"]
FOOD = DAY + ["energy_kcal", "fiber_g", "energy_eligible", "fiber_eligible"]
LIFE = [
    "activity_walking_10min_days_weekly_numeric",
    "activity_moderate_days_weekly_numeric",
    "sleep_hours_daily_numeric",
]
ALCOHOL = "alcohol_current_frequency_label"
BLOCKS = (
    "none",
    "macro",
    "sodium",
    "life",
    "full",
    "macro_life",
    "joint",
    "fiber_days",
    "distribution",
    "ratios",
)


def identities(frame, day=False):
    keys = DAY if day else KEY
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Missing or duplicate composite identity/date")


def normalize(frame):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date.astype("string").str.slice(0, 10), errors="raise")
    return frame


def projected_read(path, columns, population):
    """Apply every composite scope filter before converting source values to pandas."""
    identities(population)
    filters = []
    for (cohort, visit), group in population.groupby(KEY[1:], sort=False, observed=True):
        filters.append(
            [
                ("participant_id", "in", group.participant_id.tolist()),
                ("cohort", "=", cohort),
                ("research_stage", "=", visit),
            ]
        )
    if not filters or not set(columns).issubset(pq.ParquetFile(path).schema_arrow.names):
        raise ValueError("Missing documented context columns or empty discovery scope")
    frame = pq.read_table(
        path, columns=columns, filters=filters, pre_buffer=False, use_threads=False
    ).to_pandas(ignore_metadata=True)
    wanted = set(population[KEY].itertuples(index=False, name=None))
    if not set(frame[KEY].itertuples(index=False, name=None)).issubset(wanted):
        raise ValueError("Source reader admitted identities outside discovery scope")
    return frame


def aggregate(food, nutrients, lifestyle, index):
    """Keep the original population, including people with missing new inputs."""
    identities(index)
    food, nutrients = normalize(food), normalize(nutrients)
    identities(food, True)
    nutrients = nutrients.loc[nutrients.date.notna()].copy()
    identities(nutrients, True)
    identities(lifestyle)
    wanted = set(index[KEY].itertuples(index=False, name=None))
    for frame in (food, nutrients, lifestyle):
        if not set(frame[KEY].itertuples(index=False, name=None)).issubset(wanted):
            raise ValueError("Context identities outside supplied discovery population")
    for name in ("energy_eligible", "fiber_eligible"):
        if not food[name].dropna().isin([True, False]).all():
            raise ValueError("Eligibility flags must be Boolean or 0/1")
    good_e = food.energy_eligible.eq(True).fillna(False).to_numpy(dtype=bool, na_value=False)
    good_f = food.fiber_eligible.eq(True).fillna(False).to_numpy(dtype=bool, na_value=False)
    e, f = food.energy_kcal.to_numpy(float, na_value=np.nan), food.fiber_g.to_numpy(float, na_value=np.nan)
    if np.any(good_e & (~np.isfinite(e) | (e < 500) | (e > 4000))) or np.any(
        good_f & (~good_e | ~np.isfinite(f) | (f < 0))
    ):
        raise ValueError("Current daily values contradict frozen QC")
    data = food.loc[good_e, FOOD].merge(
        nutrients[DAY + ANCHORS + [n + "_complete_sum" for n in NUTRIENTS]],
        on=DAY,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    energy, fiber = (
        data.energy_kcal.to_numpy(float, na_value=np.nan),
        data.fiber_eligible.eq(True).fillna(False).to_numpy(dtype=bool, na_value=False),
    )
    if (
        not data._merge.eq("both").all()
        or not np.isclose(
            data[ANCHORS[0]].to_numpy(float, na_value=np.nan), energy, rtol=1e-8, atol=1e-4
        ).all()
        or not np.isclose(
            data.loc[fiber, ANCHORS[1]].to_numpy(float, na_value=np.nan),
            data.loc[fiber, "fiber_g"].to_numpy(float, na_value=np.nan),
            rtol=1e-8,
            atol=1e-4,
        ).all()
    ):
        raise ValueError("Current-day nutrient energy/fiber anchors do not match")
    density = (
        data[[n + "_complete_sum" for n in NUTRIENTS]].to_numpy(float, na_value=np.nan)
        * 1000
        / energy[:, None]
    )
    density[~np.isfinite(density) | (density < 0)] = np.nan
    lookup = {k: j for j, k in enumerate(index[KEY].itertuples(index=False, name=None))}
    own, joint, aligned = [np.full((len(index), 4), np.nan) for _ in range(3)]
    distribution = np.full((len(index), 9), np.nan)
    counts = np.zeros((len(index), 4), int)
    joint_counts = np.zeros(len(index), int)
    fiber_counts = np.zeros(len(index), int)
    primary_fiber = np.full(len(index), np.nan)
    for key, group in data.groupby(KEY, sort=False, observed=True):
        row, ix = lookup[key], group.index.to_numpy()
        values = density[ix]
        fiber_counts[row] = fiber[ix].sum()
        if fiber_counts[row] >= 3:
            primary_fiber[row] = np.mean(
                1000 * group.loc[fiber[ix], "fiber_g"] / group.loc[fiber[ix], "energy_kcal"]
            )
        mask_joint = np.isfinite(values[:, :3]).all(1)
        joint_counts[row] = mask_joint.sum()
        for j in range(4):
            valid = np.isfinite(values[:, j])
            counts[row, j] = valid.sum()
            for target, mask in ((own, valid), (joint, valid & mask_joint), (aligned, valid & fiber[ix])):
                if mask.sum() >= 3:
                    target[row, j] = values[mask, j].mean()
            if j < 3 and valid.sum() >= 3:
                a = values[valid, j]
                distribution[row, 3 * j : 3 * j + 3] = [a.std(ddof=1), *np.quantile(a, [0.1, 0.9])]
    if (
        "fiber_density" in index
        and not np.isclose(
            primary_fiber, index.fiber_density.to_numpy(float, na_value=np.nan), rtol=1e-8, atol=1e-6
        ).all()
    ):
        raise ValueError("Primary fiber exposure differs from frozen person table")
    if "n_fiber_days" in index and not np.array_equal(
        fiber_counts, index.n_fiber_days.to_numpy(float, na_value=np.nan)
    ):
        raise ValueError("Primary fiber day counts differ from frozen person table")
    life = index[KEY].merge(lifestyle[KEY + LIFE + [ALCOHOL]], on=KEY, how="left", validate="one_to_one")
    numeric = life[LIFE].to_numpy(float, na_value=np.nan)
    for j, upper in enumerate((7, 7, 24)):
        invalid = np.isfinite(numeric[:, j]) & ((numeric[:, j] < 0) | (numeric[:, j] > upper))
        if invalid.any():
            raise ValueError("Lifestyle numeric values outside documented units/range")
    alcohol = life[ALCOHOL].astype(object).to_numpy()
    alcohol = np.array([None if pd.isna(v) else str(v) for v in alcohol], object)
    log = np.log(np.maximum(joint[:, :3], 0.1))
    ratios = np.column_stack(
        [np.sqrt(2 / 3) * (log[:, 0] - (log[:, 1] + log[:, 2]) / 2), (log[:, 1] - log[:, 2]) / np.sqrt(2)]
    )
    blocks = dict(
        none=np.empty((len(index), 0)),
        macro=own[:, :3],
        sodium=own,
        life=numeric,
        full=np.column_stack([own, numeric]),
        macro_life=np.column_stack([own[:, :3], numeric]),
        joint=np.column_stack([joint[:, :3], own[:, 3:4], numeric]),
        fiber_days=np.column_stack([aligned[:, :3], own[:, 3:4], numeric]),
        distribution=np.column_stack([own, numeric, distribution]),
        ratios=np.column_stack([ratios, own[:, 3:4], numeric]),
    )
    return dict(
        blocks=blocks,
        alcohol=alcohol,
        audit=dict(
            discovery_people=len(index),
            energy_days=len(data),
            anchor_mismatches=0,
            nutrient_people_ge3=dict(zip(NUTRIENTS, (counts >= 3).sum(0).tolist())),
            joint_macro_people_ge3=int((joint_counts >= 3).sum()),
            lifestyle_available=dict(zip(LIFE, np.isfinite(numeric).sum(0).tolist())),
            alcohol_available=int(sum(v is not None for v in alcohol)),
            daily_outcome_columns_read=[],
            block_widths={k: v.shape[1] for k, v in blocks.items()},
        ),
    )


def read_context(package, index, eligible):
    pop = index.loc[eligible].copy()
    paths = [
        package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet",
        package.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet",
        package.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet",
    ]
    columns = [FOOD, DAY + ANCHORS + [n + "_complete_sum" for n in NUTRIENTS], KEY + LIFE + [ALCOHOL]]
    frames = [projected_read(p, c, pop) for p, c in zip(paths, columns)]
    extra = aggregate(*frames, pop)
    positions = np.flatnonzero(eligible)
    for name, value in extra["blocks"].items():
        full = np.full((len(index), value.shape[1]), np.nan)
        full[positions] = value
        extra["blocks"][name] = full
    alcohol = np.full(len(index), None, object)
    alcohol[positions] = extra["alcohol"]
    extra["alcohol"] = alcohol
    return extra, paths
