"""Simultaneous, one-sided discovery screen against the fixed historical share."""
import numpy as np

from v7_reference import HISTORICAL_RATIO


def annotate(records, names, boot, family_names, reliable):
    """Apply a basic maximum-deviation household bootstrap on the R² scale.

    On positive full R², a positive margin is equivalent to exceeding the
    historical percentage. All predefined families participate in the maximum,
    even when their ratio is unstable. All columns share the household draws.
    """
    if not family_names or len(set(family_names)) != len(family_names):
        raise ValueError("Expected distinct prespecified families")
    indices = [names.index(name) for name in family_names]
    margin = np.array([
        100 * records[name]["delta_r2"] - HISTORICAL_RATIO * records[name]["full_r2"]
        for name in family_names
    ])
    critical = None
    if reliable:
        simulated = 100 * boot[:, 2, :][:, indices] - HISTORICAL_RATIO * boot[:, 1, :][:, indices]
        deviations = np.max(simulated - margin, axis=1)
        critical = max(0.0, float(np.quantile(deviations, 0.95)))
    supported = []
    for name, estimate in zip(family_names, margin, strict=True):
        row = records[name]
        lower = None if critical is None else float(estimate - critical)
        passed = None
        if lower is not None and row["ratio_over_full_status"] == "stable":
            passed = lower > 1e-8
        row.update(threshold_excess_pp=float(estimate),
                   threshold_excess_simul_lower95_pp=lower,
                   exceeds_benchmark_simul_lower=passed)
        if passed:
            supported.append(name)
    return dict(family_count=len(family_names), benchmark_pct=HISTORICAL_RATIO,
                critical_margin_pp=critical, supported_families=supported,
                method="Per-repeat, one-sided 95% basic max-deviation household bootstrap "
                       "on 100*delta_R2 - benchmark_pct*full_R2, conditional on fitted predictions")
