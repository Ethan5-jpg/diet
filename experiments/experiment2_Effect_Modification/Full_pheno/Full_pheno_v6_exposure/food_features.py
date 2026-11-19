"""Construct matched discovery-only dietary representations from existing TRE sources."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from source import check


KEY = ["participant_id", "cohort", "research_stage"]
DAY = [*KEY, "date"]
FOOD_GROUPS = {
    "grain": {"bread wholewheat", "pasta, grains and side dishes wholewheat"},
    "legume": {"pulses and products"},
    "produce": {"fruits", "vegetables"},
}
SNACK_DRINK = {"fruit juices and soft drinks", "low calories and diet drinks", "sweets", "snacks",
               "baked goods", "sweet milk products", "cereals"}
SOURCES = {
    "day": "inputs/daily/food_days_discovery_TRE_ONLY.parquet",
    "qc": "../analysis/shared/food_day_qc_TRE_ONLY.parquet",
    "nova_day": "../analysis/shared/food_daily_TRE_ONLY.parquet",
    "events": "../analysis/shared/food_events_clean_TRE_ONLY.parquet",
    "catalog": "../config/food_id_crosswalk_versioned.parquet",
}


@dataclass(frozen=True)
class Features:
    values: dict
    flow: dict
    reliability: dict


def projected(path, columns, ids=None):
    dataset = ds.dataset(path, format="parquet")
    rule = None if ids is None else ds.field(KEY[0]).isin(ids)
    return dataset.to_table(columns=columns, filter=rule, use_threads=False).to_pandas(ignore_metadata=True)


def normalise(frame):
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.normalize()
    return frame.dropna(subset=DAY)


def aggregate_average(frame, columns, baseline_count, n, *, reliable=False):
    valid = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
    frame = frame.loc[valid].sort_values(["input_row", "date"])
    groups = frame.groupby("input_row", sort=False)
    counts = groups.size().reindex(range(n), fill_value=0).to_numpy()
    good = (counts >= 3) & (counts >= .8 * baseline_count)
    average = groups[columns].mean().reindex(range(n)).to_numpy(float)
    average[~good] = np.nan
    reliability = {}
    if reliable:
        both = [(half[:len(half)//2].mean(axis=0), half[len(half)//2:].mean(axis=0))
                for row, group in groups if counts[row] >= 4 and good[row]
                for half in [group[columns].to_numpy(float)]]
        first, second = np.asarray(both).transpose(1, 0, 2) if both else (np.empty((0, len(columns))),)*2
        for j, name in enumerate(columns):
            a, b = first[:, j], second[:, j]
            corr = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 20 and a.std() > 0 and b.std() > 0 else None
            reliability[name] = {"people_ge4": len(a), "split_half_r": corr}
    return average, reliability


def read_features(root, study):
    root = Path(root)
    paths = {name: (root / loc).resolve(strict=True) for name, loc in SOURCES.items()}
    index = pd.DataFrame(dict(zip(KEY, study.keys, strict=True)))
    index["input_row"] = np.arange(study.n)
    ids = index.participant_id.astype("int64").unique().tolist()
    day = normalise(projected(paths["day"], DAY + ["energy_eligible", "fiber_eligible", "energy_kcal", "fiber_g"], ids))
    day = day.merge(index, on=KEY, how="inner", validate="many_to_one")
    check(not day.duplicated(DAY).any(), "V6_DUPLICATE_DAY_INPUT")
    energy = day.loc[day.energy_eligible.eq(True) & day.energy_kcal.gt(0)].copy()
    fiber = energy.loc[energy.fiber_eligible.eq(True) & energy.fiber_g.ge(0)].copy()
    energy_counts = study.numeric["n_energy_days"]
    fiber_counts = study.numeric["n_fiber_days"]

    def count_match(frame, values):
        actual = np.bincount(frame.input_row, minlength=study.n)
        scored = values >= 3
        return float(np.mean(actual[scored] == values[scored]))

    counts = dict(fiber_day_count_match=count_match(fiber, fiber_counts),
                  energy_day_count_match=count_match(energy, energy_counts))
    check(min(counts.values()) >= .99, "V6_ARCHIVED_FOOD_WINDOW_MISMATCH")
    fiber["total_fiber"] = 1000. * fiber.fiber_g / fiber.energy_kcal
    sums = fiber.groupby("input_row")[["fiber_g", "energy_kcal"]].sum().reindex(range(study.n))
    ratio = (1000. * sums.fiber_g / sums.energy_kcal).to_numpy()[:, None]
    ratio[~(fiber_counts >= 3)] = np.nan

    event_cols = ["food_id", "food_category", "qc_primary_eligible", "calories_kcal_clean", "dietary_fiber_g_clean"]
    events = normalise(projected(paths["events"], DAY + event_cols, ids))
    events = events.loc[events.qc_primary_eligible.eq(True)].merge(energy[DAY], on=DAY, validate="many_to_one")
    events["category"] = events.food_category.astype("string").fillna("").str.lower().str.strip()
    grams = events.dietary_fiber_g_clean
    events["fiber_sum"] = grams
    for name, accepted in FOOD_GROUPS.items():
        events[name] = grams.where(events.category.isin(accepted), 0.)
    verified = events.groupby(DAY)[["fiber_sum", *FOOD_GROUPS]].sum(min_count=1).reset_index()
    source = fiber.merge(verified, on=DAY, how="inner", validate="one_to_one")
    source = source.loc[np.isclose(source.fiber_g, source.fiber_sum, rtol=.01, atol=.05)].copy()
    for name in FOOD_GROUPS:
        source[name] = 1000. * source[name] / source.energy_kcal
    origins, rel_fiber = aggregate_average(source, ["total_fiber", *FOOD_GROUPS], fiber_counts, study.n, reliable=True)

    nutrients = ["carbohydrate_g", "protein_g", "lipid_g", "sodium_mg"]
    labels = [n + "_complete_sum" for n in nutrients]
    qc = normalise(projected(paths["qc"], DAY + labels, ids))
    nutrient = energy.merge(qc, on=DAY, how="inner", validate="one_to_one")
    for label, name in zip(labels, nutrients, strict=True):
        nutrient[name] = (1000. * nutrient[label] / nutrient.energy_kcal).where(nutrient[label].ge(0))
    panel, rel_nutrient = aggregate_average(nutrient, nutrients, energy_counts, study.n, reliable=True)

    catalogue = projected(paths["catalog"], ["food_id", "NOVA"]).dropna(subset=["food_id"])
    catalogue["id"] = catalogue.food_id.astype("string")
    catalogue["nova"] = pd.to_numeric(catalogue.NOVA, errors="coerce")
    check(not catalogue.id.duplicated().any(), "V6_NONUNIQUE_FOOD_CATALOGUE")
    events["nova"] = events.food_id.astype("string").map(catalogue.set_index("id").nova)
    kcal = events.calories_kcal_clean
    known = kcal.where(events.nova.eq(4), 0.)
    events["known_snacks"] = known.where(events.category.isin(SNACK_DRINK), 0.)
    events["known_other"] = known.where(~events.category.isin(SNACK_DRINK), 0.)
    events["unknown"] = kcal.where(~events.nova.isin([1, 2, 3, 4]), 0.)
    upf_labels = ["known_snacks", "known_other", "unknown"]
    classified = events.groupby(DAY)[upf_labels].sum(min_count=1).reset_index()
    old_nova = normalise(projected(paths["nova_day"], DAY + ["known_UPF_kcal", "unknown_NOVA_kcal"], ids))
    upf = energy.merge(classified, on=DAY, validate="one_to_one").merge(old_nova, on=DAY, validate="one_to_one")
    good = np.isclose(upf.known_snacks + upf.known_other, upf.known_UPF_kcal, rtol=.005, atol=.1)
    good &= np.isclose(upf.unknown, upf.unknown_NOVA_kcal, rtol=.005, atol=.1)
    upf = upf.loc[good].copy()
    upf[upf_labels] = upf[upf_labels].div(upf.energy_kcal, axis=0)
    subtypes, rel_upf = aggregate_average(upf, upf_labels, energy_counts, study.n, reliable=True)
    values = {"fiber_cumulative": ratio, "fiber_sources": origins[:, 1:], "fiber_total_and_sources": origins,
              "nutrient_density": panel, "sodium_density": panel[:, 3:],
              "upf_known_and_unknown": np.column_stack((subtypes[:, :2].sum(axis=1), subtypes[:, 2])),
              "upf_known_subtypes": subtypes}
    counts.update(fiber_source_days=len(source), nova_matched_days=len(upf), energy_days=len(energy),
                  fiber_days=len(fiber), support_people={k: int(np.isfinite(v).all(axis=1).sum())
                                                        for k, v in values.items()})
    return Features(values, counts, {**rel_fiber, **rel_nutrient, **rel_upf}), paths
