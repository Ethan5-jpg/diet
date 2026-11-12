"""Reconstruct exact FoodID templates and audit their within-template dose support."""

import numpy as np
import pandas as pd

KEYS = ["input_row", "start"]
TOTALS = ["energy", "carbs", "fat", "protein", "fiber"]
DIRECTIONS = ("early_to_late", "late_to_early")


def reconstruct(meals, food, nutrients, span_seconds, eligible_events):
    if meals.duplicated(KEYS).any():
        raise ValueError("Archived meal keys are ambiguous")
    linked = pd.merge_asof(
        food.loc[food.t.notna()].sort_values("t"), meals[KEYS].sort_values("start"),
        left_on="t", right_on="start", by="input_row", direction="backward",
        tolerance=span_seconds,
    ).dropna(subset=["start"])
    amounts = linked[nutrients].to_numpy(float)
    if not (eligible_events(linked).all() and np.isfinite(amounts).all() and (amounts >= 0).all()):
        raise ValueError("Linked events do not satisfy archived meal QC")
    linked["food_id"] = linked.food_id.astype("string").str.strip().fillna("")
    grouped = linked.groupby(KEYS)
    rollup = grouped[nutrients].sum()
    rollup["event_date"] = grouped.date.first()
    rollup["signature"] = grouped.food_id.agg(lambda values: tuple(sorted(values)) if values.ne("").all() else ())
    joined = meals.merge(rollup, on=KEYS, how="left", validate="one_to_one")
    if not (len(rollup) == len(meals) and grouped.date.nunique().eq(1).all()
            and joined.date.eq(joined.event_date).all()
            and np.allclose(joined[TOTALS], joined[nutrients], atol=1e-7, rtol=1e-10)):
        raise ValueError("Reconstructed meal keys, nutrient totals, or dates differ")
    dates = joined.groupby("input_row").date
    joined["late"] = dates.rank(method="dense").gt(dates.transform("nunique") // 2)
    complete = joined.loc[joined.signature.map(bool)].copy()
    complete["template"] = complete.groupby(["input_row", "signature"], sort=False).ngroup()
    return complete.drop(columns=[*nutrients, "event_date", "signature"])


def scopes(complete):
    pooled = complete.loc[complete.groupby("template").date.transform("nunique").ge(2)].copy()
    pooled["group"] = pooled.template
    dates = pooled.groupby(["template", "late"]).date.nunique()
    both = dates.ge(2).groupby(level="template").sum()
    calendar = pooled.loc[pooled.template.isin(both.index[both.eq(2)])].copy()
    calendar["group"] = calendar.template * 2 + calendar.late.astype(int)
    return pooled, calendar


def matched_pair_count(rows):
    count = 0
    for _, group in rows.groupby("group", sort=False):
        day = group.date.to_numpy()
        valid = day[:, None] != day
        for name, limit in (("energy", 50), ("protein", 5), ("local_hour", 2)):
            values = group[name].to_numpy(float)
            distance = np.abs(values[:, None] - values)
            if name == "local_hour":
                distance = np.minimum(distance, 24 - distance)
            valid &= distance <= limit
        carbs = group.carbs.to_numpy(float)
        count += int((valid & (np.abs(carbs[:, None] - carbs) >= 10)).any())
    return count


def summary(rows, calendar=False):
    flow = dict(people=int(rows.input_row.nunique()), households=int(rows.split_unit_id.nunique()),
                meals=len(rows), templates=int(rows.template.nunique()))
    if rows.empty:
        return dict(**flow, carb_sd_g=0., adjusted_carb_sd_g=0., retained_carb_variance=0.,
                    effective_households=0., groups_with_matched_pair=0)
    person_period = ["input_row", "late"] if calendar else "input_row"
    weights = 1 / rows.groupby(person_period).date.transform("size").to_numpy()
    energy = np.log(rows.energy.to_numpy(float) / 100)
    protein = (rows.protein.to_numpy(float) - 15) / 10
    values = pd.DataFrame({"carbs": rows.carbs.to_numpy(float), "energy": energy, "energy_sq": energy ** 2,
                           "protein": protein, "protein_sq": protein ** 2})
    centered = values - values.groupby(rows.group.to_numpy()).transform("mean")
    weighted = centered.to_numpy() * np.sqrt(weights)[:, None]
    dose, adjust = weighted[:, 0], weighted[:, 1:]
    residual = dose - adjust @ np.linalg.lstsq(adjust, dose, rcond=1e-10)[0]
    total, independent = dose @ dose, residual @ residual
    information = pd.Series(residual ** 2).groupby(rows.split_unit_id.to_numpy()).sum().to_numpy()
    info_square = information @ information
    return dict(**flow, carb_sd_g=float(np.sqrt(total / weights.sum())),
                adjusted_carb_sd_g=float(np.sqrt(independent / weights.sum())),
                retained_carb_variance=float(independent / total) if total > 0 else 0.,
                effective_households=float(independent ** 2 / info_square) if independent > 1e-14 else 0.,
                groups_with_matched_pair=matched_pair_count(rows))


def paired_summary(values, seed):
    null, alternative, tss = values
    active = np.flatnonzero(np.any(values != 0, axis=0))
    if (len(active) < 2 or np.count_nonzero(tss > 0) < 2 or not np.isfinite(values).all()):
        raise ValueError("Insufficient household outcome variance")
    gain = null - alternative
    draws = np.random.default_rng(seed).choice(active, (1999, len(active)), replace=True)
    denominator = tss[draws].sum(axis=1)
    if not (denominator > 0).all():
        raise ValueError("Nonpositive bootstrap denominator")
    return dict(delta_r2=float(gain.sum() / tss.sum()),
                interval=np.quantile(gain[draws].sum(axis=1) / denominator, [.025, .975]).tolist())
