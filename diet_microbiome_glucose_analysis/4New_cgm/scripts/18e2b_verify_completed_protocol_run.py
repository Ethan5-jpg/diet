#!/usr/bin/env python3
"""
Step 18e2b — Post-hoc verification/finalization of the completed Paper-20
protocol-window antibiotic/PPI mediation sensitivity run.

Purpose
-------
The full 1490-path Step18e2 run already completed, but the launcher stopped at
an overly strict post-run guard requiring at least one complete-case N drop.
That condition is not logically necessary: the 234 protocol-medication-missing
participants may all be outside a given frozen mediation complete-case sample.

This script DOES NOT refit any model and DOES NOT recompute bootstrap/FDR.
It verifies the completed protocol run against the frozen primary run using:
  1) exact 1490 path identity match;
  2) protocol N never larger than primary N;
  3) validated protocol source counts;
  4) actual change in fitted indirect-effect point estimates;
  5) primary/protocol significance and direction-consistency comparison.

If the fitted indirect effects changed while the frozen path identities are
identical and N never increased, the completed sensitivity run is usable even
when PATHS_WITH_N_DROP=0.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"
BRANCH = DG / "4New_cgm"
OUTPUTS = BRANCH / "outputs"

PROTOCOL_SOURCE = (
    DG / "outputs" / "new_diet_extension" / "mediation" / "protocol_abxppi"
    / "15f2_protocol_abxppi_cohort.csv"
)

PID = "participant_id"
ABX_SOURCE = "antibiotic_use_protocol_window"
PPI_SOURCE = "ppi_use_protocol_window"


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


def latest_primary_dir() -> Path:
    found = []
    for p in OUTPUTS.glob("paper20_mediation_*_primary"):
        if "protocol_abxppi" in p.name:
            continue
        f = p / "models" / "18b_primary_mediation_paths.csv"
        if f.is_file():
            found.append((f.stat().st_mtime, p))
    if not found:
        raise FileNotFoundError("No completed primary Step18b run found.")
    return sorted(found, reverse=True)[0][1]


def latest_protocol_dir() -> Path:
    found = []
    for p in OUTPUTS.glob("paper20_mediation_protocol_abxppi_*_primary"):
        f = p / "models" / "18b_primary_mediation_paths.csv"
        if f.is_file():
            found.append((f.stat().st_mtime, p))
    if not found:
        raise FileNotFoundError("No completed protocol ABX/PPI Step18e2 run found.")
    return sorted(found, reverse=True)[0][1]


def pick_consistent_col(df: pd.DataFrame, suffix: str) -> str:
    candidates = [
        f"bridge_direction_consistent_mediator_{suffix}",
        f"candidate_mediator_consistent_{suffix}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find direction-consistency column for suffix={suffix}; "
        f"tried {candidates}"
    )


def main():
    print("=== STEP 18e2b VERIFY COMPLETED PROTOCOL ABX/PPI RUN ===")
    print("MODELS_REFIT=False")
    print("BOOTSTRAP_RERUN=False")
    print("FDR_RECALCULATED=False")

    # Revalidate the exact protocol source.
    src = pd.read_csv(PROTOCOL_SOURCE, low_memory=False)
    required = [PID, ABX_SOURCE, PPI_SOURCE]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise RuntimeError(f"Protocol source missing columns: {missing}")

    if len(src) != 6706 or src[PID].astype(str).nunique() != 6706:
        raise RuntimeError(
            f"Unexpected protocol source size: rows={len(src)}, "
            f"unique_pid={src[PID].astype(str).nunique()}"
        )

    abx = pd.to_numeric(src[ABX_SOURCE], errors="coerce")
    ppi = pd.to_numeric(src[PPI_SOURCE], errors="coerce")
    print(
        "PROTOCOL_SOURCE_COUNTS="
        f"rows:{len(src)} "
        f"abx1:{int(abx.eq(1).sum())} "
        f"ppi1:{int(ppi.eq(1).sum())} "
        f"abx_missing:{int(abx.isna().sum())} "
        f"ppi_missing:{int(ppi.isna().sum())}"
    )
    if (
        int(abx.eq(1).sum()) != 90
        or int(ppi.eq(1).sum()) != 254
        or int(abx.isna().sum()) != 234
        or int(ppi.isna().sum()) != 234
    ):
        raise RuntimeError("Protocol medication counts no longer match audited values.")

    primary_dir = latest_primary_dir()
    protocol_dir = latest_protocol_dir()
    primary_file = primary_dir / "models" / "18b_primary_mediation_paths.csv"
    protocol_file = protocol_dir / "models" / "18b_primary_mediation_paths.csv"

    print(f"PRIMARY_REFERENCE={primary_dir}")
    print(f"PROTOCOL_RUN={protocol_dir}")

    primary = pd.read_csv(primary_file, low_memory=False)
    protocol = pd.read_csv(protocol_file, low_memory=False)

    keys = ["exposure", "outcome_field", "species"]
    for label, df in [("primary", primary), ("protocol", protocol)]:
        miss = [c for c in keys if c not in df.columns]
        if miss:
            raise RuntimeError(f"{label} result missing keys: {miss}")
        if df.duplicated(keys).any():
            raise RuntimeError(f"{label} result contains duplicate path identities.")

    pkeys = set(map(tuple, primary[keys].to_numpy()))
    qkeys = set(map(tuple, protocol[keys].to_numpy()))
    exact = pkeys == qkeys
    print(f"EXACT_PATH_SET_MATCH={exact}")
    print(f"PRIMARY_PATHS={len(pkeys)}")
    print(f"PROTOCOL_PATHS={len(qkeys)}")
    if not exact or len(pkeys) != 1490:
        raise RuntimeError("Frozen 1490-path identity guard failed.")

    cmp = primary.merge(
        protocol,
        on=keys,
        suffixes=("_primary", "_protocol"),
        validate="one_to_one",
    )

    # N guard: N may stay the same. It simply must never increase.
    if "N_primary" not in cmp.columns or "N_protocol" not in cmp.columns:
        raise RuntimeError("N columns missing after primary/protocol merge.")
    n1 = pd.to_numeric(cmp["N_primary"], errors="coerce")
    n2 = pd.to_numeric(cmp["N_protocol"], errors="coerce")
    n_increase = int((n2 > n1).fillna(False).sum())
    n_drop = int((n2 < n1).fillna(False).sum())
    max_drop = int((n1 - n2).fillna(0).max())

    print(f"PATHS_WITH_N_INCREASE={n_increase}")
    print(f"PATHS_WITH_N_DROP={n_drop}")
    print(f"MAX_N_DROP={max_drop}")
    if n_increase:
        raise RuntimeError(
            f"Protocol-adjusted N exceeded primary N on {n_increase} paths."
        )

    # Crucial verification: did the fitted point estimates actually change?
    for c in ["indirect_effect_primary", "indirect_effect_protocol"]:
        if c not in cmp.columns:
            raise RuntimeError(f"Required point-estimate column missing: {c}")

    pind = pd.to_numeric(cmp["indirect_effect_primary"], errors="coerce")
    qind = pd.to_numeric(cmp["indirect_effect_protocol"], errors="coerce")
    valid = pind.notna() & qind.notna() & np.isfinite(pind) & np.isfinite(qind)
    abs_diff = (qind - pind).abs()
    changed = valid & ~np.isclose(
        pind, qind, rtol=1e-10, atol=1e-12, equal_nan=False
    )

    changed_n = int(changed.sum())
    valid_n = int(valid.sum())
    median_abs_diff = float(abs_diff[valid].median()) if valid_n else np.nan
    max_abs_diff = float(abs_diff[valid].max()) if valid_n else np.nan
    rho = safe_spearman(pind, qind)

    print(f"INDIRECT_POINT_ESTIMATES_VALID={valid_n}")
    print(f"INDIRECT_POINT_ESTIMATES_CHANGED={changed_n}")
    print(f"INDIRECT_CHANGED_FRACTION={changed_n / valid_n:.6f}" if valid_n else "INDIRECT_CHANGED_FRACTION=nan")
    print(f"INDIRECT_MEDIAN_ABS_DIFF={median_abs_diff:.12g}")
    print(f"INDIRECT_MAX_ABS_DIFF={max_abs_diff:.12g}")
    print(f"INDIRECT_RHO_PRIMARY_VS_PROTOCOL={rho:.9f}")

    if changed_n == 0:
        raise RuntimeError(
            "All fitted indirect-effect point estimates are unchanged. "
            "This does NOT prove the protocol covariates entered the models; "
            "do not accept the completed run."
        )

    # Compare inference membership.
    sig_p = "mediation_FDR05_primary_primary"
    sig_q = "mediation_FDR05_primary_protocol"
    if sig_p not in cmp.columns or sig_q not in cmp.columns:
        raise RuntimeError("Primary/protocol mediation-FDR columns not found.")

    ps = as_bool(cmp[sig_p])
    qs = as_bool(cmp[sig_q])

    pcon_col = pick_consistent_col(cmp, "primary")
    qcon_col = pick_consistent_col(cmp, "protocol")
    pc = as_bool(cmp[pcon_col])
    qc = as_bool(cmp[qcon_col])

    flip = valid & (np.sign(pind) != np.sign(qind))

    print()
    print("--- PRIMARY VS PROTOCOL SUMMARY ---")
    print(f"PRIMARY_SIGNIFICANT={int(ps.sum())}")
    print(f"PROTOCOL_SIGNIFICANT={int(qs.sum())}")
    print(f"PRIMARY_SIG_RETAINED={int((ps & qs).sum())}")
    print(f"PRIMARY_SIG_LOST={int((ps & ~qs).sum())}")
    print(f"PROTOCOL_SIG_GAINED={int((~ps & qs).sum())}")
    print(f"PRIMARY_CONSISTENT={int(pc.sum())}")
    print(f"PROTOCOL_CONSISTENT={int(qc.sum())}")
    print(f"PRIMARY_CONSISTENT_RETAINED={int((pc & qc).sum())}")
    print(f"PRIMARY_CONSISTENT_LOST={int((pc & ~qc).sum())}")
    print(f"PROTOCOL_CONSISTENT_GAINED={int((~pc & qc).sum())}")
    print(f"INDIRECT_SIGN_FLIPS={int(flip.sum())}")

    # Save a compact comparison report into the already-completed protocol run.
    report_dir = protocol_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    out = cmp[keys].copy()
    out["N_primary"] = n1
    out["N_protocol"] = n2
    out["N_lost_protocol"] = n1 - n2
    out["indirect_effect_primary"] = pind
    out["indirect_effect_protocol"] = qind
    out["indirect_abs_diff"] = abs_diff
    out["indirect_changed"] = changed
    out["indirect_sign_flip"] = flip
    out["primary_significant"] = ps
    out["protocol_significant"] = qs
    out["primary_consistent"] = pc
    out["protocol_consistent"] = qc

    report_file = report_dir / "18e2b_primary_vs_protocol_verified_comparison.csv"
    out.to_csv(report_file, index=False)

    print()
    print("N_DROP_REQUIRED_FOR_VALIDITY=False")
    print("POINT_ESTIMATE_CHANGE_GUARD=PASS")
    print("FROZEN_PATH_GUARD=PASS")
    print("N_NONINCREASE_GUARD=PASS")
    print("STEP18E2_COMPLETED_RUN_VERIFICATION=PASS")
    print(f"REPORT={report_file}")


if __name__ == "__main__":
    main()
