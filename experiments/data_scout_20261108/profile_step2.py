"""Describe the frozen discovery cohort; filter upstream values before materialization."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["participant_id", "cohort", "research_stage"]


def describe(label, series):
    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    if len(finite):
        low, median, high = np.quantile(finite, [0.1, 0.5, 0.9])
        print(f"{label}: finite={len(finite)}/{len(values)} p10/50/90={low:.2f}/{median:.2f}/{high:.2f}")
    else:
        print(f"{label}: finite=0/{len(values)}")


def main(root):
    root = Path(root)
    fields = KEY + ["input_row", "final_role", "n_fiber_days", "n_energy_days",
                    "fiber_density", "mean_daily_energy_fiber"]
    path = root / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    x = pq.ParquetFile(path, pre_buffer=False).read(columns=fields).to_pandas(ignore_metadata=True)
    if not x.final_role.eq("discovery").all() or not np.array_equal(x.input_row, np.arange(len(x))):
        raise ValueError("Unexpected discovery role or row alignment")
    path = root / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        masks = np.isfinite(archive["y"])
    if masks.shape != (2, len(x)):
        raise ValueError("Frozen evaluation mask shape differs")
    pop = x.loc[masks[0]].copy()
    if pop.empty or pop[KEY].isna().any().any() or pop.duplicated(KEY).any():
        raise ValueError("Invalid evaluation identities")
    if pop.cohort.nunique() != 1 or pop.research_stage.nunique() != 1:
        raise ValueError("Expected one explicitly scoped cohort and visit")
    filters = [("participant_id", "in", pop.participant_id.tolist()),
               ("cohort", "=", pop.cohort.iloc[0]),
               ("research_stage", "=", pop.research_stage.iloc[0])]

    def read(path, wanted, person=True):
        names = pq.ParquetFile(path, pre_buffer=False).schema_arrow.names
        columns = KEY + [c for c in wanted if c in names]
        frame = pq.read_table(path, columns=columns, filters=filters, pre_buffer=False,
                              use_threads=False).to_pandas(ignore_metadata=True)
        return pop[KEY].merge(frame, on=KEY, how="left", validate="one_to_one") if person else frame

    print(f"STEP2 repeat0_N={len(pop)} repeat1_N={masks[1].sum()} shared={(masks[0] & masks[1]).sum()}")
    print("Distributions: finite/N and p10/50/90; original population retained.")
    for name in ("n_fiber_days", "n_energy_days", "fiber_density", "mean_daily_energy_fiber"):
        describe(name, pop[name])
    raw = read(root / "inputs/discovery_v2/D_Glu_raw_Y_TRE_ONLY.parquet", ["Y_value", "Y_unit"])
    if "Y_unit" not in raw or set(raw.Y_unit.dropna()) != {"mg/dL"}:
        raise ValueError("Unexpected glucose unit")
    describe("raw_Glu_mg_dL", raw.Y_value)
    extra = ["activity_walking_10min_days_weekly_numeric", "activity_moderate_days_weekly_numeric",
             "activity_vigorous_days_weekly_numeric", "sleep_hours_daily_numeric"]
    cov = read(root.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet",
               extra + ["alcohol_current_frequency_label"])
    for name in extra:
        if name in cov:
            describe(name, cov[name])
        else:
            print(name + " MISSING_COLUMN")
    if "alcohol_current_frequency_label" in cov:
        alcohol = cov.alcohol_current_frequency_label.astype("string").str.strip()
        known = alcohol.notna() & ~alcohol.str.lower().isin(["", "unknown", "__unknown__", "missing"])
        print(f"alcohol_label: available={known.sum()}/{len(pop)} categories={alcohol[known].nunique()}")
    else:
        print("alcohol_label MISSING_COLUMN")
    common = read(root / "inputs/daily/common_days_no_scores_TRE_ONLY.parquet", ["date"], False)
    if common[KEY + ["date"]].isna().any().any() or common.duplicated(KEY + ["date"]).any():
        raise ValueError("Invalid prepared common-day identities")
    counts = common.groupby(KEY).size().reindex(pd.MultiIndex.from_frame(pop[KEY]), fill_value=0)
    print(f"prepared_common_people={(counts >= 3).sum()}/{len(pop)} days={len(common)}")
    describe("prepared_common_days_all", counts)
    date_columns = ["food_first_date", "food_last_date", "Y_first_date", "Y_last_date",
                    "O_sample_date", "G_sample_date"]
    dates = read(root.parent / "analysis/D_Glu/fiber_density/base_table_TRE_ONLY.parquet", date_columns)
    missing = [c for c in date_columns if c not in dates]
    if missing:
        print("TIMING_MISSING_COLUMNS=" + ",".join(missing))
    for name in date_columns:
        value = dates[name] if name in dates else pd.Series(pd.NaT, index=dates.index)
        dates[name] = pd.to_datetime(value.astype("string").str.slice(0, 10), errors="coerce")
    valid_y = dates[["Y_first_date", "Y_last_date"]].notna().all(axis=1) & (dates.Y_first_date <= dates.Y_last_date)
    valid_food = dates[["food_first_date", "food_last_date"]].notna().all(axis=1) & (dates.food_first_date <= dates.food_last_date)
    known = valid_y & valid_food
    overlap = (dates.food_first_date <= dates.Y_last_date) & (dates.Y_first_date <= dates.food_last_date)
    print(f"window_envelope_overlap={(known & overlap).sum()}/{known.sum()} known; not exact-day overlap")
    for name in ("O_sample_date", "G_sample_date"):
        seen = valid_y & dates[name].notna()
        before = seen & (dates[name] < dates.Y_first_date)
        during = seen & dates[name].between(dates.Y_first_date, dates.Y_last_date)
        after = seen & (dates[name] > dates.Y_last_date)
        print(f"{name}: known={seen.sum()}/{len(pop)} before/within/after={before.sum()}/{during.sum()}/{after.sum()}")
        describe(name + "_minus_CGM_end_days", (dates[name] - dates.Y_last_date).dt.days.where(seen))
    print("Prepared common-day absence does not mean no shared day; source calendar dates are descriptive.")


if __name__ == "__main__":
    main(Path("/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1"))
