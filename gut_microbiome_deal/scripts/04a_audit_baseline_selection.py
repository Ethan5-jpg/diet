from pathlib import Path
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

SOURCE = (
    ROOT
    / "Transfer"
    / "gut_microbiome"
    / "abundance"
    / "metaphlan4"
    / "aggregated"
    / "metaphlan_abundance_species.csv"
)

REPORT_DIR = (
    ROOT
    / "gut_microbiome_deal"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

KEY = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]


# ============================================================
# Load original 11295 rows
# ============================================================

df = pd.read_csv(SOURCE)

species_cols = [
    c for c in df.columns
    if c not in KEY
]

baseline = df[
    df["research_stage"].eq("00_00_visit")
].copy()

print("=" * 80)
print("BASELINE SOURCE")
print("=" * 80)

print("Rows:", len(baseline))
print(
    "Unique participants:",
    baseline["participant_id"].nunique()
)
print("Species:", len(species_cols))


# Convert once
baseline[species_cols] = (
    baseline[species_cols]
    .apply(pd.to_numeric, errors="coerce")
)


# ============================================================
# Different ways to resolve the 18 duplicated participants
# ============================================================

def select_array0(data):
    x = data.copy()

    x["_priority"] = (
        x["array_index"].ne(0).astype(int)
    )

    return (
        x.sort_values(
            [
                "participant_id",
                "_priority",
                "array_index",
            ]
        )
        .drop_duplicates(
            "participant_id",
            keep="first",
        )
        .drop(columns="_priority")
    )


def select_array1(data):
    """
    For duplicated participants, prefer array_index=1.
    Participants that only have index 0 remain unchanged.
    """
    x = data.copy()

    x["_priority"] = (
        x["array_index"].ne(1).astype(int)
    )

    return (
        x.sort_values(
            [
                "participant_id",
                "_priority",
                "array_index",
            ]
        )
        .drop_duplicates(
            "participant_id",
            keep="first",
        )
        .drop(columns="_priority")
    )


def select_best_completeness(data):
    """
    For duplicated participants, select the sample having
    the largest number of observed species among all 2088.
    """
    x = data.copy()

    x["_nonmissing"] = (
        x[species_cols]
        .notna()
        .sum(axis=1)
    )

    return (
        x.sort_values(
            [
                "participant_id",
                "_nonmissing",
                "array_index",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            "participant_id",
            keep="first",
        )
        .drop(columns="_nonmissing")
    )


def select_first(data):
    """
    Simply use the first baseline record appearing in source.
    """
    return data.drop_duplicates(
        "participant_id",
        keep="first",
    ).copy()


# ============================================================
# Full paper pipeline
# ============================================================

def run_pipeline(name, selected):
    X = selected[species_cols]

    if len(X) != 9087:
        raise RuntimeError(
            f"{name}: expected 9087 rows, got {len(X)}"
        )

    # ----------------------------
    # Step 1: prevalence >= 10%
    # ----------------------------

    prevalence = (
        X.notna().sum(axis=0)
        / len(X)
    )

    keep388 = prevalence >= 0.10

    species388 = set(
        prevalence.index[keep388]
    )

    X388 = X.loc[
        :,
        list(species388),
    ]

    # ----------------------------
    # Step 2: observed-only
    # mean abundance >= 0.01%
    # ----------------------------

    mean_abundance = X388.mean(
        axis=0,
        skipna=True,
    )

    keep379 = (
        mean_abundance >= 0.01
    )

    species379 = set(
        mean_abundance.index[keep379]
    )

    X379 = X.loc[
        :,
        list(species379),
    ]

    # ----------------------------
    # Step 3: sample missing >90%
    # ----------------------------

    missing_fraction = (
        X379.isna().sum(axis=1)
        / len(species379)
    )

    remove = (
        missing_fraction > 0.90
    )

    removed_ids = set(
        selected.loc[
            remove,
            "participant_id",
        ]
    )

    result = {
        "strategy": name,
        "samples": len(X),
        "species_start": len(species_cols),
        "species_after_prevalence": len(species388),
        "species_after_mean": len(species379),
        "removed_gt90_missing": int(remove.sum()),
        "retained_samples": int((~remove).sum()),
    }

    return (
        result,
        species388,
        species379,
        removed_ids,
    )


# ============================================================
# Run strategies
# ============================================================

strategies = {
    "array_index_0": select_array0(baseline),
    "array_index_1": select_array1(baseline),
    "best_completeness": select_best_completeness(baseline),
    "first_in_file": select_first(baseline),
}

results = {}
species388_sets = {}
species379_sets = {}
removed_sets = {}

for name, selected in strategies.items():

    (
        result,
        species388,
        species379,
        removed,
    ) = run_pipeline(
        name,
        selected,
    )

    results[name] = result
    species388_sets[name] = species388
    species379_sets[name] = species379
    removed_sets[name] = removed


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame(
    results.values()
)

print("\n" + "=" * 100)
print("FULL PIPELINE UNDER DIFFERENT BASELINE SELECTION RULES")
print("=" * 100)

print(
    summary.to_string(index=False)
)


# ============================================================
# Compare species identities
# ============================================================

reference = "array_index_0"

print("\n" + "=" * 100)
print("SPECIES SET DIFFERENCES VS CURRENT array_index_0")
print("=" * 100)

for name in strategies:

    if name == reference:
        continue

    diff388 = (
        species388_sets[name]
        ^ species388_sets[reference]
    )

    diff379 = (
        species379_sets[name]
        ^ species379_sets[reference]
    )

    print(
        f"\n{name}"
    )

    print(
        "388-species symmetric difference:",
        len(diff388),
    )

    print(
        "379-species symmetric difference:",
        len(diff379),
    )

    print(
        "Removed participant difference:",
        len(
            removed_sets[name]
            ^ removed_sets[reference]
        ),
    )


# ============================================================
# Show any strategy reproducing full paper checkpoints
# ============================================================

print("\n" + "=" * 100)
print("STRATEGIES MATCHING PAPER")
print("=" * 100)

for name, result in results.items():

    full_match = (
        result["samples"] == 9087
        and
        result["species_start"] == 2088
        and
        result["species_after_prevalence"] == 388
        and
        result["species_after_mean"] == 379
        and
        result["retained_samples"] == 9075
    )

    print(
        f"{name:20s}: "
        f"{'FULL MATCH' if full_match else 'no'}"
    )


# ============================================================
# Save
# ============================================================

summary.to_csv(
    REPORT_DIR
    / "04a_baseline_selection_pipeline_summary.csv",
    index=False,
)

print("\nSaved:")
print(
    REPORT_DIR
    / "04a_baseline_selection_pipeline_summary.csv"
)