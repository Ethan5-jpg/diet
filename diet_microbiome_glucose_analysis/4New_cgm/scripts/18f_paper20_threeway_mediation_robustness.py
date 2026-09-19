#!/usr/bin/env python3
"""
Step 18f — Paper-20 three-way mediation robustness consolidation

Integrates the already-completed:
  1) primary mediation run
  2) strict-cohort mediation sensitivity
  3) protocol-window antibiotic/PPI mediation sensitivity

No models are refit. No bootstrap or FDR is recomputed.

Primary discovery is frozen to the primary run. "Highest robustness" means:
  primary significant + direction-consistent
  AND retained significant + direction-consistent in strict
  AND retained significant + direction-consistent in protocol ABX/PPI sensitivity.

Sensitivity-only gains are reported descriptively and are NOT promoted into
the primary discovery set.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BRANCH = ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm"
OUTPUTS = BRANCH / "outputs"

KEYS = ["exposure", "outcome_field", "species"]


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes", "y", "t"}
    )


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(x[ok].corr(y[ok], method="spearman"))


def find_model_file(run_dir: Path) -> Path:
    preferred = run_dir / "models" / "18b_primary_mediation_paths.csv"
    if preferred.is_file():
        return preferred
    candidates = sorted((run_dir / "models").glob("*mediation_paths*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No mediation path model table found under {run_dir}")
    return candidates[0]


def latest_run(kind: str) -> Path:
    found = []
    for p in OUTPUTS.glob("paper20_mediation_*"):
        if not p.is_dir():
            continue

        name = p.name

        if kind == "primary":
            if "protocol_abxppi" in name or not name.endswith("_primary"):
                continue
        elif kind == "strict":
            if "protocol_abxppi" in name or not name.endswith("_strict"):
                continue
        elif kind == "protocol":
            if "protocol_abxppi" not in name or not name.endswith("_primary"):
                continue
        else:
            raise ValueError(kind)

        try:
            f = find_model_file(p)
        except FileNotFoundError:
            continue
        found.append((f.stat().st_mtime, p))

    if not found:
        raise FileNotFoundError(f"No completed {kind} mediation run found.")
    return sorted(found, reverse=True)[0][1]


def pick_sig_col(df: pd.DataFrame) -> str:
    for c in [
        "mediation_FDR05_primary",
        "mediation_FDR05_primary_primary",
        "significant_mediation_FDR05",
    ]:
        if c in df.columns:
            return c
    # tolerant fallback
    hits = [c for c in df.columns if "mediation" in c.lower() and "fdr05" in c.lower()]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(f"Could not identify mediation significance column. candidates={hits}")


def pick_consistent_col(df: pd.DataFrame) -> str:
    for c in [
        "bridge_direction_consistent_mediator",
        "candidate_mediator_consistent",
        "canonical_direction_consistent",
    ]:
        if c in df.columns:
            return c
    hits = [
        c for c in df.columns
        if "consistent" in c.lower()
        and ("mediator" in c.lower() or "candidate" in c.lower())
    ]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(f"Could not identify direction-consistency column. candidates={hits}")


def pick_indirect_col(df: pd.DataFrame) -> str:
    for c in ["indirect_effect", "indirect"]:
        if c in df.columns:
            return c
    hits = [c for c in df.columns if "indirect" in c.lower() and "effect" in c.lower()]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(f"Could not identify indirect-effect column. candidates={hits}")


def load_run(run_dir: Path, tag: str) -> pd.DataFrame:
    f = find_model_file(run_dir)
    df = pd.read_csv(f, low_memory=False)

    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing path identity columns {missing}")
    if df.duplicated(KEYS).any():
        raise RuntimeError(f"{tag}: duplicate path identities found")

    sig_col = pick_sig_col(df)
    con_col = pick_consistent_col(df)
    ind_col = pick_indirect_col(df)

    keep = KEYS.copy()
    extra = [
        c for c in [
            "outcome_label", "role", "N",
            sig_col, con_col, ind_col,
            "FDR_BH_within_score_outcome",
            "bootstrap_p_indirect",
        ]
        if c in df.columns
    ]
    keep += [c for c in extra if c not in keep]

    out = df[keep].copy()
    rename = {
        sig_col: f"significant_{tag}",
        con_col: f"consistent_{tag}",
        ind_col: f"indirect_effect_{tag}",
    }
    if "N" in out.columns:
        rename["N"] = f"N_{tag}"
    if "FDR_BH_within_score_outcome" in out.columns:
        rename["FDR_BH_within_score_outcome"] = f"FDR_{tag}"
    if "bootstrap_p_indirect" in out.columns:
        rename["bootstrap_p_indirect"] = f"bootstrap_p_{tag}"

    # Only keep descriptive columns once, from primary.
    if tag != "primary":
        for c in ["outcome_label", "role"]:
            if c in out.columns:
                out = out.drop(columns=c)

    out = out.rename(columns=rename)
    out[f"significant_{tag}"] = as_bool(out[f"significant_{tag}"])
    out[f"consistent_{tag}"] = as_bool(out[f"consistent_{tag}"])
    out[f"robust_{tag}"] = (
        out[f"significant_{tag}"] & out[f"consistent_{tag}"]
    )
    return out


def main():
    print("=== STEP 18f PAPER-20 THREE-WAY MEDIATION ROBUSTNESS ===")
    print("MODELS_REFIT=False")
    print("BOOTSTRAP_RERUN=False")
    print("FDR_RECALCULATED=False")
    print("PRIMARY_DISCOVERY_REDEFINED=False")

    primary_dir = latest_run("primary")
    strict_dir = latest_run("strict")
    protocol_dir = latest_run("protocol")

    print(f"PRIMARY_DIR={primary_dir}")
    print(f"STRICT_DIR={strict_dir}")
    print(f"PROTOCOL_DIR={protocol_dir}")

    primary = load_run(primary_dir, "primary")
    strict = load_run(strict_dir, "strict")
    protocol = load_run(protocol_dir, "protocol")

    pkeys = set(map(tuple, primary[KEYS].to_numpy()))
    skeys = set(map(tuple, strict[KEYS].to_numpy()))
    qkeys = set(map(tuple, protocol[KEYS].to_numpy()))

    print(f"PRIMARY_PATHS={len(pkeys)}")
    print(f"STRICT_PATHS={len(skeys)}")
    print(f"PROTOCOL_PATHS={len(qkeys)}")
    print(f"EXACT_1490_PATH_SET_MATCH_ALL_THREE={pkeys == skeys == qkeys}")

    if not (pkeys == skeys == qkeys) or len(pkeys) != 1490:
        raise RuntimeError("Three-way frozen path identity guard failed.")

    x = primary.merge(strict, on=KEYS, how="inner", validate="one_to_one")
    x = x.merge(protocol, on=KEYS, how="inner", validate="one_to_one")

    # Primary discovery family is frozen.
    p = x["robust_primary"]
    s = x["robust_strict"]
    q = x["robust_protocol"]

    x["primary_retained_strict"] = p & s
    x["primary_retained_protocol"] = p & q
    x["highest_robustness"] = p & s & q

    x["robustness_class"] = np.select(
        [
            p & s & q,
            p & s & ~q,
            p & ~s & q,
            p & ~s & ~q,
            ~p & s & q,
            ~p & s & ~q,
            ~p & ~s & q,
        ],
        [
            "primary_retained_both",
            "primary_strict_only",
            "primary_protocol_only",
            "primary_lost_both",
            "sensitivity_only_both",
            "strict_only_gain",
            "protocol_only_gain",
        ],
        default="nonsignificant_all",
    )

    # Effect stability.
    ip = pd.to_numeric(x["indirect_effect_primary"], errors="coerce")
    is_ = pd.to_numeric(x["indirect_effect_strict"], errors="coerce")
    iq = pd.to_numeric(x["indirect_effect_protocol"], errors="coerce")

    valid_ps = ip.notna() & is_.notna()
    valid_pq = ip.notna() & iq.notna()

    x["sign_flip_primary_strict"] = (
        valid_ps & (np.sign(ip) != np.sign(is_))
    )
    x["sign_flip_primary_protocol"] = (
        valid_pq & (np.sign(ip) != np.sign(iq))
    )

    # Primary-core counts.
    primary_n = int(p.sum())
    retained_strict = int((p & s).sum())
    retained_protocol = int((p & q).sum())
    retained_both = int((p & s & q).sum())
    strict_only = int((p & s & ~q).sum())
    protocol_only = int((p & ~s & q).sum())
    lost_both = int((p & ~s & ~q).sum())

    print()
    print("--- THREE-WAY PRIMARY CORE ---")
    print(f"PRIMARY_ROBUST_PATHS={primary_n}")
    print(f"PRIMARY_RETAINED_STRICT={retained_strict}")
    print(f"PRIMARY_RETAINED_PROTOCOL={retained_protocol}")
    print(f"PRIMARY_RETAINED_BOTH={retained_both}")
    print(f"PRIMARY_RETAINED_BOTH_FRACTION={retained_both / primary_n:.6f}")
    print(f"PRIMARY_STRICT_ONLY={strict_only}")
    print(f"PRIMARY_PROTOCOL_ONLY={protocol_only}")
    print(f"PRIMARY_LOST_BOTH={lost_both}")

    print()
    print("--- SENSITIVITY-ONLY GAINS (DESCRIPTIVE ONLY) ---")
    print(f"SENSITIVITY_ONLY_BOTH={int((~p & s & q).sum())}")
    print(f"STRICT_ONLY_GAIN={int((~p & s & ~q).sum())}")
    print(f"PROTOCOL_ONLY_GAIN={int((~p & ~s & q).sum())}")

    print()
    print("--- EFFECT STABILITY ---")
    print(f"INDIRECT_RHO_PRIMARY_VS_STRICT={safe_spearman(ip, is_):.9f}")
    print(f"INDIRECT_RHO_PRIMARY_VS_PROTOCOL={safe_spearman(ip, iq):.9f}")
    print(f"SIGN_FLIPS_PRIMARY_VS_STRICT={int(x['sign_flip_primary_strict'].sum())}")
    print(f"SIGN_FLIPS_PRIMARY_VS_PROTOCOL={int(x['sign_flip_primary_protocol'].sum())}")

    # Summaries.
    def summarize_group(cols):
        g = (
            x.groupby(cols, dropna=False)
            .agg(
                paths_tested=("species", "size"),
                unique_species=("species", "nunique"),
                primary_robust=("robust_primary", "sum"),
                retained_strict=("primary_retained_strict", "sum"),
                retained_protocol=("primary_retained_protocol", "sum"),
                retained_both=("highest_robustness", "sum"),
                primary_strict_only=("robustness_class", lambda s: int((s == "primary_strict_only").sum())),
                primary_protocol_only=("robustness_class", lambda s: int((s == "primary_protocol_only").sum())),
                primary_lost_both=("robustness_class", lambda s: int((s == "primary_lost_both").sum())),
            )
            .reset_index()
        )
        g["retained_both_fraction_of_primary"] = np.where(
            g["primary_robust"] > 0,
            g["retained_both"] / g["primary_robust"],
            np.nan,
        )
        return g

    exposure_summary = summarize_group(["exposure"])
    outcome_summary = summarize_group(["outcome_field"])
    context_summary = summarize_group(["exposure", "outcome_field"])

    retained = x.loc[x["highest_robustness"]].copy()

    species_summary = (
        retained.groupby("species", dropna=False)
        .agg(
            highest_robust_paths=("species", "size"),
            diet_exposures=("exposure", "nunique"),
            cgm_outcomes=("outcome_field", "nunique"),
            exposure_list=("exposure", lambda s: ";".join(sorted(set(map(str, s))))),
            outcome_list=("outcome_field", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(
            ["highest_robust_paths", "diet_exposures", "cgm_outcomes"],
            ascending=False,
        )
    )

    # N stability among the three runs.
    for c in ["N_primary", "N_strict", "N_protocol"]:
        if c not in x.columns:
            x[c] = np.nan

    # Output directory.
    out_dir = OUTPUTS / "paper20_mediation_threeway_robustness"
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    path_table = reports / "18f_threeway_path_robustness.csv"
    highest_table = reports / "18f_highest_robustness_paths.csv"
    exposure_table = reports / "18f_exposure_robustness_summary.csv"
    outcome_table = reports / "18f_outcome_robustness_summary.csv"
    context_table = reports / "18f_context_robustness_summary.csv"
    species_table = reports / "18f_highest_robustness_species_recurrence.csv"

    x.to_csv(path_table, index=False)
    retained.to_csv(highest_table, index=False)
    exposure_summary.to_csv(exposure_table, index=False)
    outcome_summary.to_csv(outcome_table, index=False)
    context_summary.to_csv(context_table, index=False)
    species_summary.to_csv(species_table, index=False)

    print()
    print("--- EXPOSURE SUMMARY ---")
    print(exposure_summary.to_string(index=False))

    print()
    print("--- TOP HIGHEST-ROBUSTNESS SPECIES ---")
    print(species_summary.head(30).to_string(index=False))

    print()
    print("INTERPRETATION RULES:")
    print("- Primary discovery remains frozen to the primary Step18b result.")
    print("- Highest robustness requires retention in BOTH strict and protocol ABX/PPI sensitivity.")
    print("- Sensitivity-only gains are descriptive and are not promoted into discovery.")
    print("- These remain cross-sectional mediation-style associations, not causal mediation.")
    print("- Mean glucose and GMI remain highly dependent phenotypes and should not be described as independent replication.")
    print()
    print("STEP18F_THREEWAY_CONSOLIDATION=PASS")
    print(f"PATH_TABLE={path_table}")
    print(f"HIGHEST_ROBUSTNESS={highest_table}")
    print(f"EXPOSURE_SUMMARY={exposure_table}")
    print(f"OUTCOME_SUMMARY={outcome_table}")
    print(f"CONTEXT_SUMMARY={context_table}")
    print(f"SPECIES_SUMMARY={species_table}")
    print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
