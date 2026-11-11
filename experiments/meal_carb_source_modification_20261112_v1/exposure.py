"""Reconstruct carbohydrate contributions from exact provider categories."""

import numpy as np
import pandas as pd


SOURCE = ("bread", "fruits", "bread_wholewheat", "unlabelled")
TOTALS = ("energy", "carbs", "fat", "protein", "fiber")


def reconstruct(meal, food, nutrient_names, span_seconds):
    keys = ["input_row", "start"]
    if meal.duplicated(keys).any():
        raise ValueError("Archived meal keys are not unique")
    linked = pd.merge_asof(
        food.loc[food.t.notna()].sort_values("t"),
        meal[keys].sort_values("start"),
        left_on="t", right_on="start", by="input_row",
        direction="backward", tolerance=span_seconds,
    ).dropna(subset=["start"])
    qc = linked.clock_agrees & linked.qc_primary_eligible.eq(True)
    qc &= ~linked.qc_same_food_time_n.gt(1).fillna(False) & ~linked.qc_exact_signature_n.gt(1).fillna(False)
    nutrient = linked[list(nutrient_names)].to_numpy(float)
    if not (qc.fillna(False).all() and np.isfinite(nutrient).all() and (nutrient >= 0).all()):
        raise ValueError("Linked meal events do not meet archived QC")
    category = linked.food_category.astype("string").str.strip().str.lower()
    category = category.mask(category.eq("")).fillna("unlabelled")
    for name in SOURCE:
        linked[name] = np.where(category.eq(name), nutrient[:, 1], 0.0)
    summarized = linked.groupby(keys).agg(
        {**{name: "sum" for name in (*nutrient_names, *SOURCE)}, "date": "first"}
    )
    counts = linked.groupby(keys).date.nunique()
    if not counts.eq(1).all() or len(summarized) != len(meal):
        raise ValueError("Reconstructed meal keys or days differ")
    merged = meal.merge(summarized.reset_index(), on=keys, how="left", validate="one_to_one",
                        suffixes=("", "_linked"))
    if not (merged.date.eq(merged.date_linked).all()
            and np.allclose(merged[list(TOTALS)], merged[list(nutrient_names)], atol=1e-7, rtol=1e-10)):
        raise ValueError("Archived meal nutrients or dates do not reconstruct")
    return merged


def supported(meal):
    days = [meal.loc[meal[source].ge(10)].groupby("input_row").date.nunique() for source in SOURCE[:2]]
    return meal.input_row.isin(days[0][days[0].ge(2)].index.intersection(days[1][days[1].ge(2)].index))


def nearby_pair_support(meal):
    """Count people with a different-date, approximately comparable dominant-source pair."""
    count = 0
    for _, group in meal.groupby("input_row", sort=False):
        bread = group.loc[group.bread.gt(group.carbs / 2) & group.bread.ge(10)]
        fruit = group.loc[group.fruits.gt(group.carbs / 2) & group.fruits.ge(10)]
        comparable = bread.date.to_numpy()[:, None] != fruit.date.to_numpy()
        for column, limit in (("carbs", 10), ("energy", 100), ("protein", 10)):
            distance = np.abs(bread[column].to_numpy()[:, None] - fruit[column].to_numpy())
            comparable &= distance <= limit
        time = np.abs(bread.local_hour.to_numpy()[:, None] - fruit.local_hour.to_numpy())
        count += int((comparable & (np.minimum(time, 24 - time) <= 3)).any())
    return count
