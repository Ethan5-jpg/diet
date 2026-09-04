import pandas as pd
import numpy as np
from pathlib import Path

PATH = Path("csv/lifestyle_and_environment.csv")

df = pd.read_csv(PATH, low_memory=False)

# ============================================================
# BASELINE
# ============================================================

base = df[
    df["research_stage"].astype(str).eq("00_00_visit")
].copy()

# 一人一行：
# 优先 array_index 最小的记录，通常就是 array_index=0
if "array_index" in base.columns:
    base["_array_sort"] = pd.to_numeric(
        base["array_index"], errors="coerce"
    ).fillna(999999)

    base = (
        base.sort_values(
            ["participant_id", "_array_sort"]
        )
        .drop_duplicates(
            "participant_id",
            keep="first"
        )
    )
else:
    base = base.drop_duplicates(
        "participant_id",
        keep="first"
    )

print("=" * 90)
print("BASELINE")
print("=" * 90)
print("participants =", base["participant_id"].nunique())


# ============================================================
# FIELDS
# ============================================================

wd = "activity_walking_10min_days_weekly"
wm = "activity_walking_minutes_daily"

md = "activity_moderate_days_weekly"
mm = "activity_moderate_minutes_daily"

vd = "activity_vigorous_days_weekly"
vm = "activity_vigorous_minutes_daily"

cols = [wd, wm, md, mm, vd, vm]

x = base[cols].apply(
    pd.to_numeric,
    errors="coerce"
).copy()


# ============================================================
# BASIC INVALID VALUE CLEANING
# ============================================================

day_cols = [wd, md, vd]
min_cols = [wm, mm, vm]

# days/week 合理范围 0–7
for c in day_cols:
    x.loc[(x[c] < 0) | (x[c] > 7), c] = np.nan

# minutes/day 负数视为 invalid
for c in min_cols:
    x.loc[x[c] < 0, c] = np.nan


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(name, met_min_week, eligible):

    s = met_min_week[eligible].dropna()

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    print("N =", len(s))
    print(
        "Missing within baseline =",
        len(base) - len(s),
        f"({100 * (len(base)-len(s))/len(base):.2f}%)"
    )

    if len(s) == 0:
        return

    def out(label, z):
        q = z.quantile([0.25, 0.5, 0.75])

        print(
            f"{label:18s}",
            f"median={q.loc[0.50]:.4f}",
            f"IQR=({q.loc[0.25]:.4f}, {q.loc[0.75]:.4f})"
        )

    out(
        "MET-min/week",
        s
    )

    out(
        "MET-h/week",
        s / 60.0
    )

    out(
        "MET-h/day",
        s / 60.0 / 7.0
    )


# ============================================================
# MODEL A:
# STRICT / UPDATED IPAQ-LIKE RULE
# ============================================================

strict = x.copy()

strict_complete = strict.notna().all(axis=1)

# IPAQ extreme case:
# walking + moderate + vigorous reported minutes/day > 960
time_sum = (
    strict[wm]
    + strict[mm]
    + strict[vm]
)

strict_extreme = time_sum > 960

strict_eligible = (
    strict_complete
    & ~strict_extreme
)

# truncate activity duration to 180 min/day
strict_mins = strict[[wm, mm, vm]].clip(
    upper=180
)

strict_met = (
    3.3 * strict[wd] * strict_mins[wm]
    + 4.0 * strict[md] * strict_mins[mm]
    + 8.0 * strict[vd] * strict_mins[vm]
)

summarize(
    "A. STRICT / UPDATED IPAQ",
    strict_met,
    strict_eligible
)


# ============================================================
# MODEL B:
# LEGACY UK BIOBANK DERIVATION
#
# If one activity category is incomplete,
# while both other activity categories are complete:
# set BOTH fields of the incomplete category to 0.
# ============================================================

legacy = x.copy()

pairs = {
    "walking": (wd, wm),
    "moderate": (md, mm),
    "vigorous": (vd, vm),
}

pair_complete = {}

for name, (d, m) in pairs.items():
    pair_complete[name] = (
        x[d].notna()
        & x[m].notna()
    )

for name, (d, m) in pairs.items():

    other = [
        z for z in pairs
        if z != name
    ]

    salvage = (
        ~pair_complete[name]
        & pair_complete[other[0]]
        & pair_complete[other[1]]
    )

    legacy.loc[salvage, d] = 0
    legacy.loc[salvage, m] = 0

    print(
        f"Legacy salvaged {name:10s}:",
        int(salvage.sum())
    )


legacy_complete = legacy.notna().all(axis=1)

# Original UKB derivation truncated >180 min/day to 180
legacy_mins = legacy[[wm, mm, vm]].clip(
    upper=180
)

legacy_met = (
    3.3 * legacy[wd] * legacy_mins[wm]
    + 4.0 * legacy[md] * legacy_mins[mm]
    + 8.0 * legacy[vd] * legacy_mins[vm]
)

summarize(
    "B. LEGACY UK BIOBANK",
    legacy_met,
    legacy_complete
)


# ============================================================
# PAPER TARGET
# ============================================================

print("\n" + "=" * 90)
print("PAPER REFERENCE")
print("=" * 90)

print("Reported physical activity:")
print("median = 11.6")
print("IQR    = 4.4 - 23.1")
print("unit   = MET-h/day")
print("missing in paper diet cohort = 22.0%")

print("\nIMPORTANT:")
print(
    "Do NOT choose a method merely because it numerically "
    "matches the paper."
)
print(
    "This script is diagnostic only. "
    "The legacy rule is tested because it is a documented "
    "historical UK Biobank derivation."
)