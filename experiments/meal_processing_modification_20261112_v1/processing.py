"""Reconstruct meal carbohydrate sources from the audited FoodID crosswalk."""

import numpy as np
import pandas as pd


KEYS = ["input_row", "start"]
TOTALS = ["energy", "carbs", "fat", "protein", "fiber"]
CLASSES = ("upf", "non_upf", "unknown")
CARBS = tuple(f"{group}_carbs" for group in CLASSES)
ENERGY = tuple(f"{group}_kcal" for group in CLASSES)


def reconstruct(meals, food, crosswalk, nutrients, span_seconds):
    if meals.duplicated(KEYS).any() or not crosswalk.food_id.is_unique:
        raise ValueError("Meal keys or crosswalk FoodIDs are ambiguous")
    nova = crosswalk.NOVA.dropna()
    if not nova.isin([1, 2, 3, 4]).all() or crosswalk.food_id.isna().any():
        raise ValueError("Invalid NOVA crosswalk")
    linked = pd.merge_asof(
        food.loc[food.t.notna()].sort_values("t"), meals[KEYS].sort_values("start"),
        left_on="t", right_on="start", by="input_row", direction="backward",
        tolerance=span_seconds,
    ).dropna(subset=["start"])
    qc = linked.clock_agrees & linked.qc_primary_eligible.eq(True)
    qc &= ~linked.qc_same_food_time_n.gt(1).fillna(False)
    qc &= ~linked.qc_exact_signature_n.gt(1).fillna(False)
    amounts = linked[nutrients].to_numpy(float)
    if not (qc.fillna(False).all() and np.isfinite(amounts).all() and (amounts >= 0).all()):
        raise ValueError("Linked events do not satisfy archived meal QC")
    codes = linked.food_id.map(crosswalk.set_index("food_id").NOVA)
    masks = (codes.eq(4).fillna(False), codes.isin([1, 2, 3]), codes.isna())
    for column, mask in enumerate(masks):
        linked[CARBS[column]] = np.where(mask, amounts[:, 1], 0.)
        linked[ENERGY[column]] = np.where(mask, amounts[:, 0], 0.)
    provider = linked.food_category.astype("string").str.strip().str.lower()
    linked["wholewheat_carbs"] = np.where(provider.eq("bread_wholewheat").fillna(False), amounts[:, 1], 0.)
    linked["bread_carbs"] = np.where(provider.eq("bread").fillna(False), amounts[:, 1], 0.)
    linked["unknown_events"] = masks[-1].astype(int)
    grouped = linked.groupby(KEYS)
    sums = grouped[[*nutrients, *CARBS, *ENERGY, "wholewheat_carbs", "bread_carbs", "unknown_events"]].sum()
    sums["linked_events"] = grouped.size()
    sums["event_date"] = grouped.date.first()
    merged = meals.merge(sums, on=KEYS, how="left", validate="one_to_one")
    if not (len(sums) == len(meals) and grouped.date.nunique().eq(1).all()
            and merged.date.eq(merged.event_date).all()
            and np.allclose(merged[TOTALS], merged[nutrients], atol=1e-7, rtol=1e-10)
            and np.allclose(merged[list(CARBS)].sum(axis=1), merged.carbs, atol=1e-7, rtol=1e-10)
            and np.allclose(merged[list(ENERGY)].sum(axis=1), merged.energy, atol=1e-7, rtol=1e-10)):
        raise ValueError("Reconstructed meal keys, nutrient totals, or dates differ")
    return merged.drop(columns=[*nutrients, "event_date"])


def coverage(meal):
    carb_fraction = meal.unknown_carbs / meal.carbs
    energy_fraction = meal.unknown_kcal / meal.energy
    return {
        "classified_carb": meal.unknown_carbs.eq(0) & energy_fraction.le(.10),
        "high_coverage": carb_fraction.le(.20) & energy_fraction.le(.20),
        "all_events_classified": meal.unknown_events.eq(0),
    }


def repeated_support(meal, sources, days=2, unit="input_row"):
    seen = meal[unit].unique()
    for source in sources:
        dates = meal.loc[meal[source].ge(10)].groupby(unit).date.nunique()
        seen = np.intersect1d(seen, dates.index[dates.ge(days)])
    return meal[unit].isin(seen)


def nearby_pair_count(meal, first, second):
    """Count distinct-date meals with comparable macros but different dominant sources."""
    count = 0
    for _, group in meal.groupby("input_row", sort=False):
        a = group.loc[group[first].gt(group.carbs / 2) & group[first].ge(10)]
        b = group.loc[group[second].gt(group.carbs / 2) & group[second].ge(10)]
        match = a.date.to_numpy()[:, None] != b.date.to_numpy()
        for column, limit in (("carbs", 10), ("energy", 100), ("protein", 10)):
            match &= np.abs(a[column].to_numpy()[:, None] - b[column].to_numpy()) <= limit
        hours = np.abs(a.local_hour.to_numpy()[:, None] - b.local_hour.to_numpy())
        count += int((match & (np.minimum(hours, 24 - hours) <= 3)).any())
    return count
