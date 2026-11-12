"""Interpret recorded milk product labels and measure common meal-dose support."""

import re
import unicodedata

import numpy as np
import pandas as pd


MILK = {"milk", "low fat milk"}
FREE = re.compile(r"\blactose[- ]free\b|\bfree (?:of|from) lactose\b|ללא לקטוז|נטול(?:ת)? לקטוז")
REDUCED = re.compile(r"\b(?:low|reduced)[- ]lactose\b|דל(?:ת)? לקטוז|מופחת(?:ת)? לקטוז")
OTHER = re.compile(r"\bmilk\b|\b(?:soy|oat|almond|coconut)milk\b|\blactose\b|לקטוז")
ALTERNATIVE = re.compile(r"\b(?:soy|soya|oat|almond|coconut|goat|sheep)s?[- ]?milk\b|\bplant[- ]based\b")
LABELS = ("free", "reduced", "unspecified", "other")


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def classify(short_name, product_name):
    short, product = normalized(short_name), normalized(product_name)
    if not short or ALTERNATIVE.search(product):
        return "other"
    if short not in MILK:
        return "other" if OTHER.search(short) or OTHER.search(product) else "no_listed_milk"
    free, reduced = FREE.search(product), REDUCED.search(product)
    if free and reduced:
        return "other"
    if free:
        return "free"
    if reduced:
        return "reduced"
    return "unspecified"


def reconstruct(meals, food, nutrients, span, eligible_events):
    keys = ["input_row", "start"]
    totals = ["energy", "carbs", "fat", "protein", "fiber"]
    if meals.duplicated(keys).any():
        raise ValueError("Archived meal keys are ambiguous")
    linked = pd.merge_asof(
        food.loc[food.t.notna()].sort_values("t"), meals[keys].sort_values("start"),
        left_on="t", right_on="start", by="input_row", direction="backward", tolerance=span,
    ).dropna(subset=["start"])
    amounts = linked[nutrients].to_numpy(float)
    if not (eligible_events(linked).all() and np.isfinite(amounts).all() and (amounts >= 0).all()):
        raise ValueError("Linked events do not satisfy archived meal QC")
    labels = pd.Series(
        map(classify, linked.short_food_name.fillna(""), linked.product_name.fillna("")),
        index=linked.index,
    )
    for name in LABELS:
        linked[name] = labels.eq(name)
    grouped = linked.groupby(keys)
    rollup = grouped[[*nutrients, *LABELS]].sum()
    rollup["event_date"] = grouped.date.first()
    joined = meals.merge(rollup, on=keys, how="left", validate="one_to_one")
    if not (len(rollup) == len(meals) and grouped.date.nunique().eq(1).all()
            and joined.date.eq(joined.event_date).all()
            and np.allclose(joined[totals], joined[nutrients], atol=1e-7, rtol=1e-10)):
        raise ValueError("Reconstructed meal keys, nutrient totals, or dates differ")
    flags = joined[list(LABELS)].gt(0)
    joined["context"] = flags.idxmax(axis=1).where(flags.sum(axis=1).eq(1), "ambiguous")
    joined.loc[~flags.any(axis=1), "context"] = "no_listed_milk"
    return joined.drop(columns=[*nutrients, "event_date"])


def overlapping_rows(meals, first, second):
    """Require two dates in each context and a common recorded carb interval of 10 g."""
    rows = meals.loc[meals.context.isin((first, second)) & meals.carbs.between(15, 60)]
    stats = rows.groupby(["input_row", "context"]).agg(
        days=("date", "nunique"), low=("carbs", "min"), high=("carbs", "max"),
    ).reset_index()
    left = stats.loc[stats.context.eq(first)].set_index("input_row")
    right = stats.loc[stats.context.eq(second)].set_index("input_row")
    paired = left.join(right, how="inner", lsuffix="_a", rsuffix="_b")
    low = paired[["low_a", "low_b"]].max(axis=1)
    high = paired[["high_a", "high_b"]].min(axis=1)
    keep = paired.days_a.ge(2) & paired.days_b.ge(2) & (high - low).ge(10)
    support = pd.DataFrame({"low": low[keep], "high": high[keep]})
    common = rows.join(support, on="input_row", how="inner")
    common = common.loc[common.carbs.ge(common.low) & common.carbs.le(common.high)]
    days = common.groupby(["input_row", "context"]).date.nunique()
    people = days.ge(2).groupby(level="input_row").sum()
    return common.loc[common.input_row.isin(people.index[people.eq(2)])].copy()


def genus_detection(raw, taxa, expected_rows):
    leaves = [path.split("|") for path in taxa]
    mask = np.array(["g__Bifidobacterium" in path[:-1] and path[-1].startswith("s__") for path in leaves])
    if (raw.ndim != 2 or raw.shape != (expected_rows, len(taxa)) or len(set(taxa)) != len(taxa)
            or not mask.any() or not np.isfinite(raw).all() or (raw < 0).any()
            or not np.allclose(raw.sum(axis=1), 1., atol=.001, rtol=0)):
        raise ValueError("Unexpected archived gut abundance contract")
    return (raw[:, mask] > 0).any(axis=1), int(mask.sum())
