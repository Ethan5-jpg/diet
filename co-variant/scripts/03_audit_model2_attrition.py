import pandas as pd
from pathlib import Path

MASTER = Path(
    "outputs/data/02_covariate_master.csv"
)

df = pd.read_csv(MASTER, low_memory=False)

COMMON = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "cgm_device_type",
]

AMED = COMMON

HPDI = COMMON + [
    "alcohol_intake_g_day"
]


def audit(name, columns):

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print("Master participants:", len(df))

    # --------------------------------------------------------
    # 1. Individual coverage
    # --------------------------------------------------------

    print("\nINDIVIDUAL COVERAGE")
    print("-" * 100)

    rows = []

    for c in columns:

        nonmissing = df[c].notna().sum()
        missing = df[c].isna().sum()

        rows.append({
            "variable": c,
            "nonmissing": nonmissing,
            "missing": missing,
            "missing_pct": missing / len(df) * 100,
        })

    coverage = pd.DataFrame(rows).sort_values(
        "missing",
        ascending=False,
    )

    print(
        coverage.to_string(
            index=False,
            formatters={
                "missing_pct":
                    lambda x: f"{x:.2f}%"
            }
        )
    )

    # --------------------------------------------------------
    # 2. Full complete case
    # --------------------------------------------------------

    full_complete = (
        df[columns]
        .notna()
        .all(axis=1)
    )

    full_n = int(full_complete.sum())

    print("\nFULL MODEL")
    print("-" * 100)

    print("Complete cases =", full_n)

    # --------------------------------------------------------
    # 3. Leave-one-variable-out rescue
    #
    # If removing one variable adds many people, that variable
    # is an important bottleneck.
    # --------------------------------------------------------

    print("\nLEAVE-ONE-OUT RESCUE")
    print("-" * 100)

    rescue_rows = []

    for removed in columns:

        remaining = [
            c for c in columns
            if c != removed
        ]

        n_without = int(
            df[remaining]
            .notna()
            .all(axis=1)
            .sum()
        )

        rescue_rows.append({
            "removed_variable": removed,
            "complete_without": n_without,
            "additional_people": n_without - full_n,
        })

    rescue = pd.DataFrame(
        rescue_rows
    ).sort_values(
        "additional_people",
        ascending=False,
    )

    print(
        rescue.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 4. Incremental attrition
    #
    # Follow the paper-like model order and report how many
    # participants disappear when each variable is introduced.
    # --------------------------------------------------------

    print("\nSEQUENTIAL ATTRITION")
    print("-" * 100)

    running = pd.Series(
        True,
        index=df.index,
    )

    previous_n = len(df)

    seq_rows = []

    for c in columns:

        running &= df[c].notna()

        current_n = int(
            running.sum()
        )

        seq_rows.append({
            "added_variable": c,
            "remaining": current_n,
            "lost_at_step": previous_n - current_n,
        })

        previous_n = current_n

    print(
        pd.DataFrame(seq_rows)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. Missing-count pattern
    # --------------------------------------------------------

    print("\nNUMBER OF MISSING MODEL VARIABLES PER PERSON")
    print("-" * 100)

    n_missing = (
        df[columns]
        .isna()
        .sum(axis=1)
    )

    print(
        n_missing
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 6. Exact single-variable blockers
    #
    # People who fail complete-case ONLY because of one field.
    # --------------------------------------------------------

    print("\nEXACT SINGLE-VARIABLE BLOCKERS")
    print("-" * 100)

    single = n_missing.eq(1)

    blockers = []

    for c in columns:

        n = int(
            (
                single
                & df[c].isna()
            ).sum()
        )

        blockers.append({
            "variable": c,
            "participants_blocked_only_by_this": n,
        })

    blockers = pd.DataFrame(
        blockers
    ).sort_values(
        "participants_blocked_only_by_this",
        ascending=False,
    )

    print(
        blockers.to_string(
            index=False
        )
    )


audit(
    "AMED MODEL 2",
    AMED,
)

audit(
    "hPDI MODEL 2",
    HPDI,
)