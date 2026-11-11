"""Date alignment and carbohydrate interaction terms for sample handoff timing."""

import numpy as np
import pandas as pd

KEYS = ["input_row", "start"]
TZ = "Asia/Jerusalem"
STRATA = ("all_doses", "common_dose")


def align_dates(meals, visits, registry):
    if visits.first_observed.dt.tz is None or not visits.timezone.eq(TZ).all():
        raise ValueError("Expected local time zone for verified CGM monitoring")
    first = visits.first_observed.dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    days = visits[["input_row"]].assign(first_day=first)
    usable = registry.loc[
        ~registry.date_conflict_between_linked_sources.fillna(True)
        & registry.descriptive_sample_calendar_date.notna()
        & registry.descriptive_sample_calendar_date.eq(registry.linked_index_calendar_date),
        ["input_row", "descriptive_sample_calendar_date"],
    ]
    aligned = meals.merge(days, on="input_row", validate="many_to_one")
    if len(aligned) != len(meals):
        raise ValueError("An archived meal lacks its selected monitoring connection")
    aligned = aligned.merge(usable, on="input_row", how="left", validate="many_to_one")
    meal_day = pd.to_datetime(aligned.date, format="%Y-%m-%d")
    handoff = pd.to_datetime(aligned.descriptive_sample_calendar_date, format="%Y-%m-%d")
    aligned["monitor_day"] = (meal_day - aligned.first_day).dt.days
    aligned["handoff_lag"] = (handoff - meal_day).dt.days
    if not aligned.monitor_day.ge(0).all():
        raise ValueError("An archived meal predates its monitoring connection")
    aligned["near"] = aligned.handoff_lag.abs().le(7)
    return aligned


def supported(meals, stratum):
    selected = meals.loc[meals.handoff_lag.notna()]
    if stratum == "common_dose":
        selected = selected.loc[selected.carbs.between(15, 60)]
    support = selected.groupby(["input_row", "near"]).agg(
        meals=("date", "size"), dates=("date", "nunique"), low=("carbs", "min"), high=("carbs", "max"),
    )
    ok = support.meals.ge(4) & support.dates.ge(2) & (support.high - support.low).ge(10)
    people = support.loc[ok].groupby(level="input_row").size()
    return selected.loc[selected.input_row.isin(people.index[people.eq(2)])].copy()


def overlap(meals):
    carb = meals.groupby(["input_row", "near"]).carbs.agg(["min", "max"])
    comparable = carb.groupby("input_row")["max"].min() - carb.groupby("input_row")["min"].max()
    by_day = meals.groupby(["monitor_day", "near"]).size().unstack(fill_value=0)
    common_days = by_day.index[(by_day > 0).all(axis=1)]
    return dict(people=int(meals.input_row.nunique()), households=int(meals.split_unit_id.nunique()),
                meals=len(meals), near_meals=int(meals.near.sum()),
                far_meals=int((~meals.near).sum()), dose_overlap_10g_people=int(comparable.ge(10).sum()),
                common_monitor_days=len(common_days),
                meals_on_common_monitor_days=int(meals.monitor_day.isin(common_days).sum()))


def design(meals, score, person, model):
    dose = (meals.carbs.to_numpy(float) - 40) / 10
    dose2 = dose ** 2
    day = (meals.monitor_day.to_numpy(float) - 7) / 7
    lag = meals.handoff_lag.to_numpy(float)
    delay = (meals.monitor_day.to_numpy(float) + lag - 14) / 7
    near = meals.near.to_numpy(float)
    energy = np.log(meals.energy.to_numpy(float) / 100)
    protein = (meals.protein.to_numpy(float) - 15) / 10
    z = score[person]
    controls = np.column_stack([
        day, day ** 2, near, near * dose, near * dose2, near * energy, near * protein,
        *(dose * a for a in (day, day ** 2, delay, delay ** 2)),
        *(z * a[:, None] for a in (dose, dose2, dose * day, dose * day ** 2, dose * delay)),
    ])
    return model.center(controls, person)[0], model.center(z * (dose * near)[:, None], person)[0]
