#!/usr/bin/env python3
"""
Step 18e2 v3 — Paper-20 protocol-window antibiotic/PPI mediation sensitivity.

Strategy
--------
Reuse the completed Step18b Paper-20 PRIMARY mediation implementation unchanged
except for ONE prespecified sensitivity change:

    + antibiotic_use
    + ppi_use

where these two generic modeling columns are mapped from the already validated:
    antibiotic_use_protocol_window
    ppi_use_protocol_window

source:
    outputs/new_diet_extension/mediation/protocol_abxppi/
    15f2_protocol_abxppi_cohort.csv

The Step18b frozen path set, primary cohort rule, score-specific extra covariates,
bootstrap logic, within-family BH-FDR, bridge-direction rule, and statistical core
are preserved.

This launcher does NOT edit 18b_paper20_primary_mediation.py on disk.
It creates a temporary in-memory/runtime copy with a distinct output prefix.

Safety guards:
- validated protocol source shape/counts
- no duplicate participant IDs
- generic antibiotic/PPI covariates are injected with NaN preserved
- canonical get_covariate_list is forced to append them when present
- output must contain exactly the same 1490 frozen path identities
- protocol complete-case N must never exceed primary N
- at least one path must lose N, otherwise medication adjustment was probably not active
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"
BRANCH = DG / "4New_cgm"
SCRIPTS = BRANCH / "scripts"
OUTPUTS = BRANCH / "outputs"

PRIMARY_SCRIPT = SCRIPTS / "18b_paper20_primary_mediation.py"
PROTOCOL_SOURCE = (
    DG / "outputs" / "new_diet_extension" / "mediation" / "protocol_abxppi"
    / "15f2_protocol_abxppi_cohort.csv"
)

PID = "participant_id"
ABX_SOURCE = "antibiotic_use_protocol_window"
PPI_SOURCE = "ppi_use_protocol_window"
ABX_MODEL = "antibiotic_use"
PPI_MODEL = "ppi_use"

CORE_COV_NAMES = {
    "age_years", "age", "sex", "gender", "education_level", "education",
    "smoking_status", "smoking", "sleep_duration_hours_day", "sleep_duration",
    "physical_activity_met_h_week", "physical_activity", "vitamin_use",
    "hormone_use", "bmi", "BMI", "cgm_device_type", "device_type",
}


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes", "y", "t"}
    )


def latest_primary_dir() -> Path:
    found = []
    for p in OUTPUTS.glob("paper20_mediation_*_primary"):
        if "protocol_abxppi" in p.name:
            continue
        f = p / "models" / "18b_primary_mediation_paths.csv"
        if f.is_file():
            found.append((f.stat().st_mtime, p))
    if not found:
        raise FileNotFoundError(
            "Could not find completed Step18b primary mediation output."
        )
    found.sort(reverse=True)
    return found[0][1]


def safe_spearman(a, b):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(x[ok].corr(y[ok], method="spearman"))


def load_and_validate_protocol():
    if not PROTOCOL_SOURCE.is_file():
        raise FileNotFoundError(PROTOCOL_SOURCE)

    d = pd.read_csv(PROTOCOL_SOURCE, low_memory=False)
    req = [PID, ABX_SOURCE, PPI_SOURCE]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise RuntimeError(f"Protocol source missing columns: {missing}")

    d = d[req].copy()
    d["_pid_key"] = norm_id(d[PID])

    if d["_pid_key"].duplicated().any():
        raise RuntimeError("Protocol source contains duplicate participant IDs.")

    # Reconfirm the exact source audited in Step18e0.
    if len(d) != 6706 or d["_pid_key"].nunique() != 6706:
        raise RuntimeError(
            f"Unexpected protocol source size: rows={len(d)}, "
            f"unique={d['_pid_key'].nunique()} (expected 6706/6706)"
        )

    for c, expected_one, expected_missing in [
        (ABX_SOURCE, 90, 234),
        (PPI_SOURCE, 254, 234),
    ]:
        x = pd.to_numeric(d[c], errors="coerce")
        other = int((x.notna() & ~x.isin([0, 1])).sum())
        one = int(x.eq(1).sum())
        missing_n = int(x.isna().sum())
        if other:
            raise RuntimeError(f"{c} contains non-binary non-missing values.")
        if one != expected_one or missing_n != expected_missing:
            raise RuntimeError(
                f"{c} count mismatch: one={one}, missing={missing_n}; "
                f"expected one={expected_one}, missing={expected_missing}"
            )

    abx_map = d.set_index("_pid_key")[ABX_SOURCE]
    ppi_map = d.set_index("_pid_key")[PPI_SOURCE]
    return d, abx_map, ppi_map


def patch_covariate_list_function(module):
    """
    If a module exposes get_covariate_list(score, available_columns), preserve its
    existing return value and append ABX/PPI only when those columns are available.
    """
    if not hasattr(module, "get_covariate_list"):
        return False

    old = module.get_covariate_list
    if getattr(old, "_step18e2_wrapped", False):
        return True

    def wrapped(score, available_columns, *args, **kwargs):
        out = list(old(score, available_columns, *args, **kwargs))
        available = set(map(str, available_columns))
        for c in (ABX_MODEL, PPI_MODEL):
            if c in available and c not in out:
                out.append(c)
        return out

    wrapped._step18e2_wrapped = True
    wrapped._step18e2_original = old
    module.get_covariate_list = wrapped
    return True


def append_to_global_covariate_lists(module):
    """
    Conservative fallback for modules that store a module-level base covariate list.
    Only touches globals whose NAME looks covariate-related and whose values already
    contain several known core covariates.
    """
    changed = []
    for name, value in list(vars(module).items()):
        lname = name.lower()
        if "cov" not in lname:
            continue
        if not isinstance(value, (list, tuple)):
            continue
        if not all(isinstance(x, str) for x in value):
            continue

        overlap = len(set(value) & CORE_COV_NAMES)
        if overlap < 3:
            continue

        new = list(value)
        for c in (ABX_MODEL, PPI_MODEL):
            if c not in new:
                new.append(c)

        if isinstance(value, tuple):
            new = tuple(new)
        setattr(module, name, new)
        changed.append(name)
    return changed


def main():
    print("=== STEP 18e2 PAPER-20 PROTOCOL-WINDOW ABX/PPI MEDIATION ===")
    print("MODELS_REFIT=True")
    print("BOOTSTRAP_RERUN=True")
    print("FDR_RECALCULATED_WITHIN_SAME_FROZEN_FAMILIES=True")
    print("PRIMARY_DISCOVERY_REDEFINED=False")
    print("SENSITIVITY_ONLY=True")
    print(f"PRIMARY_SCRIPT={PRIMARY_SCRIPT}")
    print(f"PROTOCOL_SOURCE={PROTOCOL_SOURCE}")

    if not PRIMARY_SCRIPT.is_file():
        raise FileNotFoundError(PRIMARY_SCRIPT)

    protocol, abx_map, ppi_map = load_and_validate_protocol()
    print("PROTOCOL_SOURCE_QC=True")
    print("PROTOCOL_ROWS=6706")
    print("PROTOCOL_ABX_POSITIVE=90")
    print("PROTOCOL_PPI_POSITIVE=254")
    print("PROTOCOL_MISSING_EACH=234")

    primary_dir = latest_primary_dir()
    primary_file = primary_dir / "models" / "18b_primary_mediation_paths.csv"
    primary = pd.read_csv(primary_file, low_memory=False)
    print(f"PRIMARY_REFERENCE={primary_dir}")

    # ------------------------------------------------------------------
    # 1) Install a narrow pd.read_csv wrapper.
    #
    # Any dataframe that clearly looks like a covariate/cohort dataframe
    # (participant_id + >=3 known core covariates) receives the two generic
    # modeling columns. Existing generic medication columns are overwritten,
    # ensuring that broad baseline flags cannot leak into this sensitivity.
    # ------------------------------------------------------------------
    original_read_csv = pd.read_csv
    injection_log = []

    def protocol_read_csv(*args, **kwargs):
        df = original_read_csv(*args, **kwargs)

        if PID not in df.columns:
            return df

        core_overlap = len(set(map(str, df.columns)) & CORE_COV_NAMES)
        if core_overlap < 3:
            return df

        pid_key = norm_id(df[PID])
        df = df.copy()
        df[ABX_MODEL] = pid_key.map(abx_map)
        df[PPI_MODEL] = pid_key.map(ppi_map)

        path = str(args[0]) if args else str(kwargs.get("filepath_or_buffer", ""))
        injection_log.append(
            (
                path,
                len(df),
                int(df[ABX_MODEL].eq(1).sum()),
                int(df[PPI_MODEL].eq(1).sum()),
                int(df[ABX_MODEL].isna().sum()),
                int(df[PPI_MODEL].isna().sum()),
            )
        )
        return df

    pd.read_csv = protocol_read_csv

    # ------------------------------------------------------------------
    # 2) Make a runtime-only copy of Step18b with a distinct output prefix.
    # No statistical lines are changed.
    # ------------------------------------------------------------------
    source = PRIMARY_SCRIPT.read_text(encoding="utf-8")
    # IMPORTANT: only rename Step18b's OUTPUT directory prefix.
    # Do NOT globally replace "paper20_mediation_", because Step18b also uses
    # "paper20_mediation_interface_audit_*" to locate the already completed
    # Step18a input. A global replacement would incorrectly make it look for a
    # nonexistent "paper20_mediation_protocol_abxppi_interface_audit_*".
    output_line = 'outdir = outputs / f"paper20_mediation_{stamp}{suffix}"'
    replacement_line = (
        'outdir = outputs / '
        'f"paper20_mediation_protocol_abxppi_{stamp}{suffix}"'
    )
    if runtime_source_count := source.count(output_line):
        if runtime_source_count != 1:
            raise RuntimeError(
                f"Expected exactly one Step18b output line, found "
                f"{runtime_source_count}."
            )
    else:
        raise RuntimeError(
            "Could not identify Step18b output-directory line; refusing to run."
        )

    runtime_source = source.replace(output_line, replacement_line, 1)
    runtime_source = runtime_source.replace(
        'print("ANTIBIOTIC_PPI_INCLUDED=False")',
        'print("ANTIBIOTIC_PPI_INCLUDED=True")',
        1,
    )
    runtime_source = runtime_source.replace(
        "STEP 18b", "STEP 18e2 [Step18b core + protocol ABX/PPI]"
    )

    # Step18b contains a hard "current fitted sample must equal Step18a
    # N/hash" guard. That is correct for the primary analysis but incompatible
    # with adding medication covariates that have genuine missingness.
    #
    # Do NOT match a brittle literal code block. Locate the guard structurally
    # with Python's AST and neutralize exactly that If statement. A separate
    # wrapper below reconstructs the original pre-medication sample, requires
    # exact Step18a N/hash agreement, and then requires the medication-adjusted
    # sample to be a subset. Thus the integrity check is preserved, just moved
    # to the sensitivity-aware location.
    import ast as _ast

    def _neutralize_step18a_sample_guard(src_text: str):
        tree = _ast.parse(src_text)
        lines = src_text.splitlines()
        matches = []

        for node in _ast.walk(tree):
            if not isinstance(node, _ast.If):
                continue
            seg = _ast.get_source_segment(src_text, node) or ""
            low = seg.lower()

            # Identify the primary Step18a sample N/hash mismatch guard by
            # semantics, not exact formatting.
            has_step18a = "step18a" in low or "audit_n" in low
            has_sample = "sample" in low
            has_hash = "hash" in low or "audit_h" in low or "sha256" in low
            has_raise = "runtimeerror" in low or "raise" in low

            if has_step18a and has_sample and has_hash and has_raise:
                matches.append(node)

        # Prefer the smallest matching If block in case a parent If also happens
        # to contain the same text.
        if matches:
            matches.sort(
                key=lambda n: (
                    (getattr(n, "end_lineno", n.lineno) - n.lineno),
                    n.lineno,
                )
            )
            target = matches[0]
        else:
            # Fallback: search for an If whose source contains the known audit
            # variable semantics even if its error text does not say "sample".
            fallback = []
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.If):
                    continue
                seg = _ast.get_source_segment(src_text, node) or ""
                low = seg.lower()
                if (
                    ("audit_n" in low or "audit_sample" in low)
                    and ("audit_h" in low or "sha256" in low or "hash" in low)
                    and ("raise" in low)
                ):
                    fallback.append(node)
            if not fallback:
                return src_text, None
            fallback.sort(
                key=lambda n: (
                    (getattr(n, "end_lineno", n.lineno) - n.lineno),
                    n.lineno,
                )
            )
            target = fallback[0]

        start = target.lineno - 1
        end = getattr(target, "end_lineno", target.lineno)
        indent = " " * int(getattr(target, "col_offset", 0))
        replacement = (
            indent
            + "pass  # Step18e2: original Step18a exact-sample guard moved "
              "to sensitivity-aware subset guard below"
        )
        patched_lines = lines[:start] + [replacement] + lines[end:]
        patched = "\n".join(patched_lines) + ("\n" if src_text.endswith("\n") else "")
        _ast.parse(patched)  # compile-level structural check
        return patched, (target.lineno, end)

    runtime_source, _qc_patch = _neutralize_step18a_sample_guard(runtime_source)
    if _qc_patch is None:
        raise RuntimeError(
            "Could not structurally locate the Step18b Step18a N/hash sample guard."
        )
    print(
        f"STEP18B_N_HASH_GUARD_RELOCATED=True "
        f"lines={_qc_patch[0]}-{_qc_patch[1]}"
    )

    runtime_dir = OUTPUTS / "_runtime_step18e2"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_script = runtime_dir / "18e2_runtime_paper20_protocol_abxppi.py"
    runtime_script.write_text(runtime_source, encoding="utf-8")
    compile(runtime_source, str(runtime_script), "exec")

    # ------------------------------------------------------------------
    # 3) Import the runtime Step18b module without invoking __main__ yet.
    # ------------------------------------------------------------------
    module_name = "hpp_step18e2_runtime"
    spec = importlib.util.spec_from_file_location(module_name, runtime_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create runtime import spec.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    # Patch any obvious module-level covariate lists.
    changed_globals = append_to_global_covariate_lists(mod)

    # Locate canonical statistical-core modules already loaded by Step18b.
    canonical_modules = []
    for smod in list(sys.modules.values()):
        f = getattr(smod, "__file__", None)
        if f and Path(str(f)).name == "10_diet_microbiome_cgm_mediation.py":
            canonical_modules.append(smod)

    get_cov_patched = False
    for cmod in canonical_modules:
        if patch_covariate_list_function(cmod):
            get_cov_patched = True
        changed_globals.extend(
            [f"{getattr(cmod, '__name__', 'canonical')}.{x}"
             for x in append_to_global_covariate_lists(cmod)]
        )

    print(f"CANONICAL_MODULES_FOUND={len(canonical_modules)}")
    print(f"GET_COVARIATE_LIST_PATCHED={get_cov_patched}")
    print(
        "MODULE_LEVEL_COVARIATE_LISTS_PATCHED="
        + (";".join(changed_globals) if changed_globals else "NONE")
    )

    # Strong sensitivity-specific sample guard:
    # For every path, first reconstruct the ORIGINAL Step18b complete-case
    # sample without ABX/PPI and demand exact N/hash agreement with Step18a.
    # Then construct the ABX/PPI-adjusted sample and demand that it is a true
    # subset of that original sample. This preserves Step18b's sample-integrity
    # guarantee while allowing protocol-medication missingness to reduce N.
    if not hasattr(mod, "prepare_path_data") or not hasattr(mod, "sample_hash"):
        raise RuntimeError(
            "Step18b runtime module lacks prepare_path_data/sample_hash."
        )

    original_prepare_path_data = mod.prepare_path_data
    subset_guard_stats = {
        "calls": 0,
        "n_dropped_paths": 0,
        "max_n_drop": 0,
    }

    def protocol_prepare_path_data(row, master, micro):
        saved_base = list(mod.BASE_COVARIATES)

        # Reconstruct the original primary Step18b sample.
        mod.BASE_COVARIATES = [
            c for c in saved_base if c not in {ABX_MODEL, PPI_MODEL}
        ]
        try:
            base_d, _base_covs = original_prepare_path_data(
                row, master, micro
            )
        finally:
            mod.BASE_COVARIATES = saved_base

        base_n = len(base_d)
        base_h = mod.sample_hash(base_d[PID])
        audit_n = int(row["audit_N"])
        audit_h = str(row["audit_sample_sha256"])

        if base_n != audit_n or base_h != audit_h:
            raise RuntimeError(
                "Pre-medication Step18b sample no longer matches Step18a for "
                f"{row['exposure']} / {row['outcome_field']} / "
                f"{row['species']}: reconstructed={base_n}/{base_h}, "
                f"audit={audit_n}/{audit_h}"
            )

        # Now build the actual protocol-window ABX/PPI-adjusted sample.
        adj_d, adj_covs = original_prepare_path_data(row, master, micro)

        base_ids = set(norm_id(base_d[PID]))
        adj_ids = set(norm_id(adj_d[PID]))
        if not adj_ids.issubset(base_ids):
            extra = sorted(adj_ids - base_ids)[:5]
            raise RuntimeError(
                "ABX/PPI-adjusted sample is not a subset of the primary "
                f"complete-case sample; example unexpected IDs={extra}"
            )
        if len(adj_d) > len(base_d):
            raise RuntimeError(
                "ABX/PPI-adjusted sample unexpectedly increased N."
            )

        drop = len(base_d) - len(adj_d)
        subset_guard_stats["calls"] += 1
        if drop > 0:
            subset_guard_stats["n_dropped_paths"] += 1
        subset_guard_stats["max_n_drop"] = max(
            subset_guard_stats["max_n_drop"], drop
        )

        return adj_d, adj_covs

    mod.prepare_path_data = protocol_prepare_path_data

    # If canonical exposes get_covariate_list, prove the two fields are returned
    # when available before spending ~1 hour on the full run.
    preflight_ok = False
    for cmod in canonical_modules:
        fn = getattr(cmod, "get_covariate_list", None)
        if fn is None:
            continue
        test_available = list(CORE_COV_NAMES) + [ABX_MODEL, PPI_MODEL]
        try:
            test = list(fn("AHEI", test_available))
            if ABX_MODEL in test and PPI_MODEL in test:
                preflight_ok = True
                break
        except Exception:
            pass

    # Module-level list fallback can also be sufficient, but canonical preflight
    # is preferred. Abort if neither route is visible.
    if not preflight_ok and not changed_globals:
        raise RuntimeError(
            "Could not verify that antibiotic_use and ppi_use will enter the "
            "model covariate list. Refusing to launch the full run."
        )
    print(f"ABX_PPI_COVARIATE_PREFLIGHT={preflight_ok or bool(changed_globals)}")

    # ------------------------------------------------------------------
    # 4) Run Step18b main using its own defaults (same bootstrap/workers/seed).
    # ------------------------------------------------------------------
    if not hasattr(mod, "main"):
        raise RuntimeError("Step18b runtime module has no main().")

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(runtime_script)]
        rc = mod.main()
        if rc not in (None, 0):
            raise RuntimeError(f"Step18b runtime main returned {rc}")
    finally:
        sys.argv = old_argv
        pd.read_csv = original_read_csv

    print(f"READ_CSV_PROTOCOL_INJECTIONS={len(injection_log)}")
    print(f"PRIMARY_SAMPLE_GUARD_CALLS={subset_guard_stats['calls']}")
    print(
        f"PATHS_WITH_PROTOCOL_N_DROP_DURING_FIT="
        f"{subset_guard_stats['n_dropped_paths']}"
    )
    print(f"MAX_PROTOCOL_N_DROP={subset_guard_stats['max_n_drop']}")
    if not injection_log:
        raise RuntimeError(
            "No covariate/cohort dataframe received protocol ABX/PPI fields."
        )

    print("--- PROTOCOL INJECTION EXAMPLES ---")
    for row in injection_log[:10]:
        print(
            f"path={row[0]} rows={row[1]} "
            f"abx1={row[2]} ppi1={row[3]} "
            f"abx_missing={row[4]} ppi_missing={row[5]}"
        )

    # ------------------------------------------------------------------
    # 5) Find new protocol output and run post-run guards.
    # ------------------------------------------------------------------
    candidates = []
    for p in OUTPUTS.glob("paper20_mediation_protocol_abxppi_*_primary"):
        f = p / "models" / "18b_primary_mediation_paths.csv"
        if f.is_file():
            candidates.append((f.stat().st_mtime, p))
    if not candidates:
        raise RuntimeError("Protocol run output directory not found.")
    candidates.sort(reverse=True)
    protocol_dir = candidates[0][1]
    protocol_file = protocol_dir / "models" / "18b_primary_mediation_paths.csv"
    adjusted = original_read_csv(protocol_file, low_memory=False)

    keys = ["exposure", "outcome_field", "species"]
    for label, df in [("primary", primary), ("protocol", adjusted)]:
        miss = [c for c in keys if c not in df.columns]
        if miss:
            raise RuntimeError(f"{label} result missing keys: {miss}")
        if df.duplicated(keys).any():
            raise RuntimeError(f"{label} result has duplicate path identities.")

    pkeys = set(map(tuple, primary[keys].to_numpy()))
    akeys = set(map(tuple, adjusted[keys].to_numpy()))
    exact = pkeys == akeys
    if not exact or len(akeys) != 1490:
        raise RuntimeError(
            f"Frozen path guard failed: primary={len(pkeys)}, "
            f"protocol={len(akeys)}, exact={exact}"
        )

    cmp = primary.merge(
        adjusted, on=keys, suffixes=("_primary", "_protocol"),
        validate="one_to_one"
    )

    if "N_primary" not in cmp.columns or "N_protocol" not in cmp.columns:
        raise RuntimeError("N columns unavailable for medication complete-case guard.")

    n1 = pd.to_numeric(cmp["N_primary"], errors="coerce")
    n2 = pd.to_numeric(cmp["N_protocol"], errors="coerce")
    n_increase = int((n2 > n1).fillna(False).sum())
    n_drop = int((n2 < n1).fillna(False).sum())
    if n_increase:
        raise RuntimeError(
            f"Protocol-adjusted N exceeded primary N on {n_increase} paths."
        )
    if n_drop == 0:
        raise RuntimeError(
            "No path lost complete-case N after adding protocol medication "
            "covariates; adjustment may not have entered the fitted models."
        )

    ps = as_bool(cmp["mediation_FDR05_primary_primary"])
    qs = as_bool(cmp["mediation_FDR05_primary_protocol"])

    pcon_col = (
        "bridge_direction_consistent_mediator_primary"
        if "bridge_direction_consistent_mediator_primary" in cmp.columns
        else "candidate_mediator_consistent_primary"
    )
    qcon_col = (
        "bridge_direction_consistent_mediator_protocol"
        if "bridge_direction_consistent_mediator_protocol" in cmp.columns
        else "candidate_mediator_consistent_protocol"
    )
    pc = as_bool(cmp[pcon_col])
    qc = as_bool(cmp[qcon_col])

    pind = pd.to_numeric(cmp["indirect_effect_primary"], errors="coerce")
    qind = pd.to_numeric(cmp["indirect_effect_protocol"], errors="coerce")
    flip = (
        pind.notna() & qind.notna()
        & (np.sign(pind) != np.sign(qind))
    )

    retained_sig = int((ps & qs).sum())
    lost_sig = int((ps & ~qs).sum())
    gained_sig = int((~ps & qs).sum())
    retained_con = int((pc & qc).sum())
    lost_con = int((pc & ~qc).sum())
    gained_con = int((~pc & qc).sum())

    rho_all = safe_spearman(pind, qind)
    rho_primary_sig = safe_spearman(pind[ps], qind[ps])

    print()
    print("=== STEP 18e2 PROTOCOL ABX/PPI POST-RUN SUMMARY ===")
    print(f"OUTPUT_DIR={protocol_dir}")
    print(f"EXACT_1490_PATH_SET_MATCH={exact}")
    print(f"PATHS_COMPARED={len(cmp)}")
    print(f"PATHS_WITH_N_DROP={n_drop}")
    print(f"PATHS_WITH_N_INCREASE={n_increase}")
    print()
    print(f"PRIMARY_SIGNIFICANT={int(ps.sum())}")
    print(f"PROTOCOL_SIGNIFICANT={int(qs.sum())}")
    print(f"PRIMARY_SIGNIFICANT_RETAINED_PROTOCOL={retained_sig}")
    print(f"PRIMARY_SIGNIFICANT_LOST_PROTOCOL={lost_sig}")
    print(f"PROTOCOL_SIGNIFICANT_GAINED={gained_sig}")
    print()
    print(f"PRIMARY_CONSISTENT={int(pc.sum())}")
    print(f"PROTOCOL_CONSISTENT={int(qc.sum())}")
    print(f"PRIMARY_CONSISTENT_RETAINED_PROTOCOL={retained_con}")
    print(f"PRIMARY_CONSISTENT_LOST_PROTOCOL={lost_con}")
    print(f"PROTOCOL_CONSISTENT_GAINED={gained_con}")
    print()
    print(f"INDIRECT_RHO_ALL1490={rho_all:.9f}")
    print(f"INDIRECT_RHO_PRIMARY_SIG={rho_primary_sig:.9f}")
    print(f"INDIRECT_SIGN_FLIPS_ALL1490={int(flip.sum())}")
    print(f"INDIRECT_SIGN_FLIPS_PRIMARY_SIG={int(flip[ps].sum())}")

    # Save comparison report next to protocol run.
    report_dir = protocol_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    cmp["primary_significant"] = ps
    cmp["protocol_significant"] = qs
    cmp["primary_consistent"] = pc
    cmp["protocol_consistent"] = qc
    cmp["indirect_sign_flip"] = flip
    cmp["N_lost_protocol"] = n1 - n2
    cmp.to_csv(
        report_dir / "18e2_primary_vs_protocol_path_comparison.csv",
        index=False
    )

    inject_df = pd.DataFrame(
        injection_log,
        columns=[
            "read_path", "rows", "antibiotic_positive",
            "ppi_positive", "antibiotic_missing", "ppi_missing"
        ],
    )
    inject_df.to_csv(
        report_dir / "18e2_protocol_covariate_injection_audit.csv",
        index=False
    )

    print("ABX_PPI_MODEL_GUARD=PASS")
    print("STEP18E2_STATUS=PASS")


if __name__ == "__main__":
    main()
