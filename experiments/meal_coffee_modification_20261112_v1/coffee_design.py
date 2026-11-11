"""Reconstruct documented coffee co-intake and define the matched contrasts."""

import numpy as np
import pandas as pd

KEYS = ["input_row", "start"]
TOTALS = ["energy", "carbs", "fat", "protein", "fiber"]
LABELS = {"coffee", "cappuccino"}
SPECIES = "s__Clostridium_phoceensis"
STRATA = ("documented", "low_calorie")


def eligible_events(food):
    return (food.clock_agrees & food.qc_primary_eligible.eq(True)
            & ~food.qc_same_food_time_n.gt(1).fillna(False)
            & ~food.qc_exact_signature_n.gt(1).fillna(False)).fillna(False)


def reconstruct(meals, food, nutrients, span_seconds):
    if meals.duplicated(KEYS).any():
        raise ValueError("Archived meal keys are ambiguous")
    names = food.short_food_name.astype("string").str.strip().str.casefold().fillna("")
    coffee = names.isin(LABELS)
    food = food.assign(coffee=coffee, cappuccino=names.eq("cappuccino"),
                       questionable=names.eq("") | (names.str.contains("coffee|espresso|cappuccino|latte|decaf")
                                                     & ~coffee))
    daily = (food.loc[eligible_events(food)].groupby(["input_row", "date"]).coffee.any()
             .sort_index().rename("day_coffee").reset_index())
    by_person = daily.groupby("input_row")
    seen = by_person.cumcount()
    before = by_person.day_coffee.cumsum() - daily.day_coffee
    daily["prior_coffee_share"] = (before + 1) / (seen + 2)
    daily["prior_log_days"] = seen
    linked = pd.merge_asof(
        food.loc[food.t.notna()].sort_values("t"), meals[KEYS].sort_values("start"),
        left_on="t", right_on="start", by="input_row", direction="backward",
        tolerance=span_seconds,
    ).dropna(subset=["start"])
    amounts = linked[nutrients].to_numpy(float)
    if not (eligible_events(linked).all() and np.isfinite(amounts).all() and (amounts >= 0).all()):
        raise ValueError("Coffee reconstruction disagrees with archived meal QC")
    linked["coffee_kcal"] = np.where(linked.coffee, amounts[:, 0], 0.)
    linked["coffee_carbs"] = np.where(linked.coffee, amounts[:, 1], 0.)
    groups = linked.groupby(KEYS)
    sums = groups.agg({**{n: "sum" for n in [*nutrients, "coffee_kcal", "coffee_carbs"]},
                       **{n: "any" for n in ("coffee", "cappuccino", "questionable")},
                       "date": "first"})
    joined = meals.merge(sums, on=KEYS, validate="one_to_one", suffixes=("", "_events"))
    if not (len(joined) == len(meals) == len(sums) and joined.date.eq(joined.date_events).all()
            and groups.date.nunique().eq(1).all()
            and np.allclose(joined[TOTALS], joined[nutrients], atol=1e-7, rtol=1e-10)):
        raise ValueError("Archived meal nutrients or dates do not reconstruct")
    joined = joined.merge(daily.drop(columns="day_coffee"), on=["input_row", "date"], validate="many_to_one")
    if len(joined) != len(meals):
        raise ValueError("An archived meal lacks a valid log day")
    joined["pure"] = joined.coffee & ~joined.cappuccino & ~joined.questionable & joined.coffee_kcal.le(20)
    return joined.drop(columns=[*nutrients, "date_events"])


def supported(meal, stratum):
    present = meal.coffee if stratum == "documented" else meal.pure
    take = (present | (~meal.coffee & ~meal.questionable)) & meal.carbs.between(15, 60)
    rows = meal.loc[take].assign(context=present[take])
    stats = rows.groupby(["input_row", "context"]).agg(
        meals=("date", "size"), days=("date", "nunique"), low=("carbs", "min"), high=("carbs", "max"),
    )
    keep = stats.meals.ge(4) & stats.days.ge(2) & (stats.high - stats.low).ge(10)
    both = stats.loc[keep].groupby(level="input_row").size().eq(2)
    return rows.loc[rows.input_row.isin(both.index[both])].copy()


def nearby_people(meal):
    count = 0
    for _, group in meal.groupby("input_row", sort=False):
        a, b = group.loc[group.context], group.loc[~group.context]
        comparable = a.date.to_numpy()[:, None] != b.date.to_numpy()
        for column, cap in (("carbs", 10), ("energy", 100), ("protein", 10)):
            comparable &= np.abs(a[column].to_numpy()[:, None] - b[column].to_numpy()) <= cap
        hour = np.abs(a.local_hour.to_numpy()[:, None] - b.local_hour.to_numpy())
        count += int((comparable & (np.minimum(hour, 24 - hour) <= 3)).any())
    return count


def taxon_fraction(raw, taxa):
    labels = [name.rsplit("|", 1)[-1] for name in taxa]
    if (raw.ndim != 2 or raw.shape[1] != len(taxa) or labels.count(SPECIES) != 1
            or not np.isfinite(raw).all() or (raw < 0).any()
            or not np.allclose(raw.sum(axis=1), 1., atol=.001, rtol=0)):
        raise ValueError("Expected a unique exact HPP species in closed gut profiles")
    return raw[:, labels.index(SPECIES)]


def interaction_design(meal, score, person, model):
    dose = (meal.carbs.to_numpy(float) - 40) / 10
    coffee = meal.context.to_numpy(float)
    hour = 2 * np.pi * meal.local_hour.to_numpy(float) / 24
    sin, cos = np.sin(hour), np.cos(hour)
    gap = np.log1p(meal.prior_gap.to_numpy(float) / 60)
    share = meal.prior_coffee_share.to_numpy(float) - .5
    history = np.log1p(meal.prior_log_days.to_numpy(float))
    energy = np.log(meal.energy.to_numpy(float) / 100)
    protein = (meal.protein.to_numpy(float) - 15) / 10
    age = (meal.age_calendar_years.to_numpy(float) - 50) / 10
    sex = meal.sex_code.to_numpy(float)
    kcals = meal.coffee_kcal.to_numpy(float) / 50
    coffee_carb = meal.coffee_carbs.to_numpy(float) / 10
    z = score[person]
    background = np.column_stack([
        coffee, coffee * dose, coffee * dose ** 2, kcals, coffee_carb, share, history,
        *(coffee * value for value in (energy, protein, age, sex, sin, cos, gap, share)),
        *(dose * value for value in (sin, cos, gap, share, coffee * share)),
        *(z * value[:, None] for value in (dose, dose ** 2, coffee, kcals, coffee_carb,
                                          dose * sin, dose * cos, dose * gap, dose * share)),
    ])
    return model.center(background, meal.person_index.to_numpy())[0], model.center(
        z * (dose * coffee)[:, None], meal.person_index.to_numpy(),
    )[0]
