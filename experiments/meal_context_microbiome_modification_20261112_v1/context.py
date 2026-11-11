"""Define meal contexts and carbohydrate support without glucose outcomes."""

import numpy as np


def select_contexts(meal):
    morning = meal.local_hour.ge(5) & meal.local_hour.lt(10) & meal.prior_gap.between(480, 960)
    daytime = meal.local_hour.ge(11) & meal.local_hour.lt(21) & meal.prior_gap.between(135, 360)
    use = meal.loc[(morning | daytime) & meal.carbs.between(15, 60)].copy()
    use["morning"] = morning.loc[use.index]
    return require_support(use, ("morning",), 2, 10, 2)


def require_support(meal, keys, days, spread, groups):
    counts = meal.groupby(["input_row", *keys]).agg(
        days=("date", "nunique"), low=("carbs", "min"), high=("carbs", "max")
    )
    supported = counts.days.ge(days) & (counts.high - counts.low).ge(spread)
    ok = supported.groupby(level=0).sum().eq(groups)
    return meal.loc[meal.input_row.isin(ok.index[ok])].copy()


def split_background(x, morning):
    morning = morning[:, None]
    return np.column_stack((x * ~morning, x * morning))


def interaction_design(dose, marker, morning, host=None):
    common = dose[:, None] * marker
    differential = common * (morning[:, None] - .5)
    clinical = None if host is None else split_background(dose[:, None] * host, morning)
    return common, differential, clinical
