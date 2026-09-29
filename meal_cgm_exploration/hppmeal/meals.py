"""Food items to bounded eating episodes. Missing nutrients never become zero."""
import hashlib
import json

import numpy as np
import pandas as pd

from .common import KEYS, NUTRIENTS


def complete_sum(s):
    return float(s.sum()) if s.notna().all() else np.nan


def build_meals(food, config):
    if food.empty:
        return pd.DataFrame()
    d = food.sort_values("t", kind="mergesort").reset_index(drop=True)
    labels, current, start = [], -1, -np.inf
    # Relative to episode start, not pairwise chaining of successive snack times.
    for t in d.t:
        if t - start > config["meal"]["max_span_minutes"] * 60:
            current += 1
            start = t
        labels.append(current)
    d["episode"] = labels
    rows = []
    for _, g in d.groupby("episode", sort=True):
        row = {k: str(g[k].iloc[0]) for k in KEYS}
        row.update(start=float(g.t.min()), end=float(g.t.max()), items=len(g),
                   duplicate_item=bool(g.duplicate_item.any()),
                   timezone_conflict=g.tz.nunique() != 1)
        row["duration_minutes"] = (row["end"] - row["start"]) / 60
        for col in NUTRIENTS:
            row[col] = complete_sum(g[col])
            row[col + "_missing_items"] = int(g[col].isna().sum())
        energy = row["calories_kcal"]
        valid_energy = np.isfinite(energy) and energy > 0
        classified = g.nova.isin([1, 2, 3, 4])
        row["nova_classified_kcal"] = float(g.loc[classified, "calories_kcal"].sum())
        row["nova4_observed_kcal"] = float(g.loc[g.nova.eq(4), "calories_kcal"].sum())
        row["nova_coverage"] = row["nova_classified_kcal"] / energy if valid_energy else np.nan
        row["nova4_pct"] = 100 * row["nova4_observed_kcal"] / energy if valid_energy else np.nan
        row["nova4_classified_pct"] = (100 * row["nova4_observed_kcal"] / row["nova_classified_kcal"]
                                       if valid_energy and row["nova_classified_kcal"] > 0 else np.nan)
        row["signature"] = ""
        if g.food_id.notna().all() and g.weight_g.notna().all() and not row["duplicate_item"]:
            items = g.groupby("food_id", sort=True).weight_g.sum()
            payload = [(str(k), round(float(v), config["meal"]["signature_weight_decimals"]))
                       for k, v in items.items()]
            row["signature"] = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        try:
            local = pd.to_datetime(row["start"], unit="s", utc=True).tz_convert(g.tz.iloc[0])
            row["local_date"] = local.strftime("%Y-%m-%d")
            row["local_hour"] = local.hour + local.minute / 60.0 + local.second / 3600.0
        except (ValueError, TypeError, KeyError):
            row["timezone_conflict"] = True
            row["local_date"], row["local_hour"] = "", np.nan
        identity = [row[k] for k in KEYS] + [row["start"]]
        row["meal_id"] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
        row["time_semantics_verified"] = config["time"]["consumption_time_verified"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["previous_gap_minutes"] = (out.start - out.end.shift(1)) / 60
    # Next intake measured from meal START, because outcomes run 0..120 from START.
    out["next_gap_minutes"] = (out.start.shift(-1) - out.start) / 60
    out["isolated_window"] = (
        out.previous_gap_minutes.ge(config["meal"]["previous_event_gap_minutes"])
        & out.next_gap_minutes.gt(config["meal"]["next_event_gap_minutes"]))
    out["meal_eligible"] = (
        out.calories_kcal.ge(config["meal"]["minimum_kcal"])
        & out.carbohydrate_g.ge(config["meal"]["minimum_carbohydrate_g"])
        & ~out.duplicate_item & ~out.timezone_conflict)
    out["nova_eligible"] = out.nova_coverage.ge(config["meal"]["nova_min_coverage"])
    return out
