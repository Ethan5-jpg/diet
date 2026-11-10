"""Audit existing daily nutrient sums on the frozen discovery cohort; no modeling."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["participant_id", "cohort", "research_stage"]
DAY = KEY + ["date"]
NUTRIENTS = ["carbohydrate_g", "lipid_g", "protein_g", "sodium_mg", "alcohol_g", "dietary_fiber_g"]


def numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def normalize_days(frame):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date.astype("string").str.slice(0, 10), errors="coerce")
    return frame


def odd_even(frame, density):
    data = frame[DAY].assign(value=density).dropna(subset=["value"]).sort_values(DAY)
    if data.empty:
        return "NA", 0
    data["half"] = data.groupby(KEY).cumcount() % 2
    cells = data.groupby(KEY + ["half"]).value.agg(["mean", "count"]).unstack("half")
    if not {0, 1}.issubset(cells["mean"].columns):
        return "NA", 0
    count = cells["count"].fillna(0)
    means = cells["mean"].loc[(count[0] >= 3) & (count[1] >= 3)]
    if len(means) < 10 or means[0].nunique() < 2 or means[1].nunique() < 2:
        return "NA", len(means)
    rho = means[0].rank().corr(means[1].rank())
    return f"{rho:.3f}", len(means)


def main(root):
    root = Path(root)
    path = root / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    pop = pq.ParquetFile(path, pre_buffer=False).read(
        columns=KEY + ["input_row", "final_role"]).to_pandas(ignore_metadata=True)
    if not pop.final_role.eq("discovery").fillna(False).all() or not np.array_equal(pop.input_row, np.arange(len(pop))):
        raise ValueError("Unexpected discovery role or input-row alignment")
    path = root / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        masks = np.isfinite(archive["y"])
    if masks.shape != (2, len(pop)):
        raise ValueError("Frozen cohort shape differs")
    pop = pop.loc[masks[0]].copy()
    if pop.empty or pop[KEY].isna().any().any() or pop.duplicated(KEY).any():
        raise ValueError("Invalid discovery identities")
    if pop.cohort.nunique() != 1 or pop.research_stage.nunique() != 1:
        raise ValueError("Expected one explicitly scoped cohort and visit")
    filters = [("participant_id", "in", pop.participant_id.tolist()),
               ("cohort", "=", pop.cohort.iloc[0]), ("research_stage", "=", pop.research_stage.iloc[0])]

    def read(path, columns):
        data = pq.read_table(path, columns=columns, filters=filters, pre_buffer=False,
                             use_threads=False).to_pandas(ignore_metadata=True)
        return normalize_days(data)

    print(f"STEP3 FOOD ONLY: frozen repeat0_N={len(pop)}; no new outcome association or fit.")
    path = root.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet"
    if not path.is_file():
        print("DAILY_NUTRIENT_QC_MISSING; event table was not scanned.")
        return
    names = pq.ParquetFile(path, pre_buffer=False).schema_arrow.names
    anchors = ["calories_kcal_complete_sum", "dietary_fiber_g_complete_sum"]
    if not set(DAY + anchors).issubset(names):
        print("DAILY_NUTRIENT_QC_MISSING_ANCHORS; inspect the schema before continuing.")
        return
    columns = DAY + list(dict.fromkeys(anchors + [n + "_complete_sum" for n in NUTRIENTS
                                                 if n + "_complete_sum" in names]))
    old = read(path, columns)
    old = old.loc[old.date.notna()].copy()
    if old.duplicated(DAY).any():
        raise ValueError("Duplicate nutrient day keys")
    base = read(root / "inputs/daily/food_days_discovery_TRE_ONLY.parquet",
                DAY + ["energy_kcal", "fiber_g", "energy_eligible", "fiber_eligible"])
    if base[DAY].isna().any().any() or base.duplicated(DAY).any():
        raise ValueError("Invalid current food day keys")
    for name in ("energy_eligible", "fiber_eligible"):
        if not base[name].dropna().isin([True, False]).all():
            raise ValueError("Eligibility flags must be Boolean or 0/1")
    admitted = base.energy_eligible.fillna(False).eq(True)
    energy = numeric(base.energy_kcal)
    if (admitted & (~np.isfinite(energy) | ~energy.between(500, 4000))).any():
        raise ValueError("Current food energy contradicts documented QC")
    fiber_claim = base.fiber_eligible.fillna(False).eq(True)
    fiber_amount = numeric(base.fiber_g)
    if (fiber_claim & (~admitted | ~np.isfinite(fiber_amount) | fiber_amount.lt(0))).any():
        raise ValueError("Current fiber eligibility contradicts documented QC")
    data = base.loc[admitted].merge(old, on=DAY, how="left", validate="one_to_one", indicator=True)
    print(f"current_energy_days={len(data)} people={data[KEY].drop_duplicates().shape[0]}/{len(pop)}")
    matched = data._merge.eq("both")
    energy = numeric(data.energy_kcal)
    bad_energy = matched & ~np.isclose(numeric(data[anchors[0]]), energy, rtol=1e-8, atol=1e-4)
    fiber = data.fiber_eligible.fillna(False).eq(True)
    bad_fiber = matched & fiber & ~np.isclose(numeric(data[anchors[1]]), numeric(data.fiber_g),
                                            rtol=1e-8, atol=1e-4)
    print(f"day_key_matches={matched.sum()}/{len(data)} energy_mismatch={bad_energy.sum()} fiber_mismatch={bad_fiber.sum()}")
    if (~matched).any() or bad_energy.any() or bad_fiber.any():
        print("ALIGNMENT_NOT_VERIFIED: stop nutrient interpretation; do not combine these versions yet.")
        return
    valid_masks = {}
    for nutrient in NUTRIENTS:
        name = nutrient + "_complete_sum"
        if name not in data:
            print(nutrient + " MISSING_COMPLETE_SUM")
            continue
        value = numeric(data[name])
        good = np.isfinite(value) & value.ge(0)
        if nutrient == "dietary_fiber_g":
            good &= fiber
        valid_masks[nutrient] = good
        density = (1000 * value / energy).where(good)
        grouped = data[KEY].assign(value=density).groupby(KEY).value.agg(["count", "mean"])
        means = grouped.loc[grouped["count"] >= 3, "mean"]
        q = "/".join(f"{v:.2f}" for v in means.quantile([0.1, 0.5, 0.9])) if len(means) else "NA"
        rho, n = odd_even(data, density)
        unit = "mg" if nutrient == "sodium_mg" else "g"
        print(f"{nutrient}: days={good.sum()}/{len(data)} people_ge3={(grouped['count'] >= 3).sum()}/{len(pop)} ge7={(grouped['count'] >= 7).sum()}")
        print(f"  density_{unit}/1000kcal p10/50/90={q}; odd_even_rho={rho} paired_people={n}")
    macros = ["carbohydrate_g", "lipid_g", "protein_g"]
    if set(macros).issubset(valid_masks):
        good = valid_masks[macros[0]] & valid_masks[macros[1]] & valid_masks[macros[2]]
        counts = data.loc[good].groupby(KEY).size()
        print(f"joint_CARB_FAT_PROTEIN: days={good.sum()} people_ge3={(counts >= 3).sum()}/{len(pop)} ge7={(counts >= 7).sum()}")
    else:
        print("joint_CARB_FAT_PROTEIN MISSING_COMPLETE_SUM")
    print("Complete sums cover recorded events, not proven complete daily intake; missing is not zero.")
    print("Odd/even: >=3 days/half, >=10 paired people, variable half-means; same-window consistency only.")


if __name__ == "__main__":
    main(Path("/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1"))
