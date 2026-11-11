"""Recover the previous logged episode and define current-by-previous dose terms."""

import numpy as np
import pandas as pd

KEYS = ["input_row", "start"]
PREVIOUS = ["prior_energy", "prior_carbs", "prior_fat", "prior_protein", "prior_fiber"]


def preceding(meals, food, nutrients, span):
    food = food.loc[food.t.notna()].sort_values("t", kind="stable")
    times = food.t.to_numpy(float)
    begins = []
    anchor = -np.inf
    for row, value in enumerate(times):
        if value - anchor > span:
            begins.append(row)
            anchor = value
    starts = np.asarray(begins, int)
    if not len(starts):
        raise ValueError("Missing archived meal episodes")
    ends = np.r_[starts[1:] - 1, len(times) - 1]
    current = meals.start.to_numpy(float)
    where = np.searchsorted(times[starts], current)
    if (where <= 0).any() or (where >= len(starts)).any() or not np.array_equal(times[starts[where]], current):
        raise ValueError("Archived meal anchors do not match reconstructed episodes")
    previous = where - 1
    gap = (current - times[ends[previous]]) / 60
    if not np.allclose(gap, meals.prior_gap, atol=1e-6, rtol=0):
        raise ValueError("Reconstructed preceding intervals differ")
    quality = food.clock_agrees & food.qc_primary_eligible.eq(True)
    quality &= ~food.qc_same_food_time_n.gt(1).fillna(False) & ~food.qc_exact_signature_n.gt(1).fillna(False)
    dates = food.date.to_numpy()
    changed = np.r_[False, dates[1:] != dates[:-1]]
    same_day = np.add.reduceat(changed.astype(int), starts) == changed[starts]
    episode_qc = np.logical_and.reduceat(quality.fillna(False).to_numpy(bool), starts) & same_day
    amounts = food[nutrients].to_numpy(float)
    valid = np.isfinite(amounts) & (amounts >= 0)
    seen = np.logical_and.reduceat(valid, starts, axis=0)
    episode_id = np.repeat(np.arange(len(starts)), ends - starts + 1)
    totals = pd.DataFrame(np.where(valid, amounts, 0.)).groupby(episode_id, sort=False).sum().to_numpy()[previous]
    linked = meals.copy()
    linked[PREVIOUS] = totals
    linked["pair_verified"] = episode_qc[previous] & seen[previous, :2].all(axis=1)
    linked["complete_previous"] = episode_qc[previous] & seen[previous].all(axis=1)
    return linked


def rows_with_context(meals):
    return meals.loc[meals.prior_gap.between(135, 360) & meals.carbs.between(15, 60)].copy()


def repeated(meal, unit="input_row", minimum_dates=2):
    low = meal.loc[meal.prior_carbs.le(20)].groupby(unit).date.nunique()
    high = meal.loc[meal.prior_carbs.ge(40)].groupby(unit).date.nunique()
    return meal[unit].isin(low[low.ge(minimum_dates)].index.intersection(high[high.ge(minimum_dates)].index))


def flow(meal):
    return dict(people=int(meal.input_row.nunique()), households=int(meal.split_unit_id.nunique()), meals=len(meal))


def design(meal, scores, model):
    now = (meal.carbs.to_numpy(float) - 40) / 10
    prior = (meal.prior_carbs.to_numpy(float) - 30) / 10
    energy = np.log(meal.prior_energy.to_numpy(float) / 100)
    protein = meal.prior_protein.to_numpy(float) / 10
    fiber = meal.prior_fiber.to_numpy(float) / 5
    lag = np.log(meal.prior_gap.to_numpy(float) / 180)
    hour = meal.local_hour.to_numpy(float) * (2 * np.pi / 24)
    time = (lag, np.sin(hour), np.cos(hour))
    measured = (energy, protein, fiber)
    z = scores[meal.input_row.to_numpy(int)]
    nuisance = np.column_stack([
        prior, prior ** 2, energy, energy ** 2, protein, fiber,
        *(prior * a for a in (*measured, *time)),
        *(now * a for a in (prior, prior ** 2, *measured, *time)),
        *(z * a[:, None] for a in (now ** 2, prior, prior ** 2,
                                  *(now * value for value in (*measured, *time)))),
    ])
    centered, _ = model.center(np.column_stack((nuisance, z * now[:, None], z * (now * prior)[:, None])),
                               meal.person_index.to_numpy())
    width = z.shape[1]
    return centered[:, :-2 * width], centered[:, -2 * width:-width], centered[:, -width:]
