"""Audit registered off/on pairs against individual held-out errors before declaring a hit."""

from math import fsum, isclose

import numpy as np
from v8_scores import EPS as R2_FLOOR
from v9_fixed import evaluate as household_evaluate


def sse(y, prediction):
    # Compensated scalar sum is independent of household aggregation, matrix layout,
    # BLAS scheduling, and the column blocking used in the simultaneous bootstrap.
    return fsum(np.square(y - prediction))


def apply_point(score, s0, s1, sst, target, critical):
    """Recompute both R²s; avoid a divided or display-rounded test at the boundary."""
    off, on = (sst - s0) / sst, (sst - s1) / sst
    delta = (s0 - s1) / sst
    if not all(
        isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-11)
        for actual, expected in (
            (score["off_r2"], off),
            (score["on_r2"], on),
            (score["delta_r2_pp"] / 100, delta),
        )
    ):
        raise ArithmeticError("Individual and household paired R2 disagree")
    # Below this floating-point error scale equality and a threshold hit cannot be
    # distinguished. This is in R² percentage points, not a fixed tolerance on ratio%.
    margin = fsum((100.0 * (s0 - s1), -target * (sst - s1))) / sst
    tol = 8 * np.finfo(float).eps * ((100.0 * (s0 + s1) + target * (sst + s1)) / sst)
    lower = None if critical is None else margin - critical
    available = min(off, on) > R2_FLOOR and score["ratio_pct"] is not None
    ratio = 100.0 * (s0 - s1) / (sst - s1) if available else None
    if not available:
        score.update(ratio_status="nonpositive_or_near_zero_R2", ratio_ci95_pct=None, ratio_ci97_5_pct=None)
    score.update(
        off_r2=off,
        on_r2=on,
        delta_r2_pp=100.0 * delta,
        ratio_pct=ratio,
        threshold_margin_r2_pp=margin,
        threshold_roundoff_r2_pp=float(tol),
        simultaneous_lower_margin_r2_pp=lower,
        above_target_point=None if not available else bool(margin > tol),
        above_target_adjusted=(
            None if lower is None or score["ratio_status"] != "stable" else bool(lower > tol)
        ),
    )


def evaluate(batches, keys, off_indices, off_names, target, draws, seed, width=64):
    """Check the name-based off column, duplicate ids, paired people and all point results.

    The bootstrap retains one family error bound for all fixed AND adaptive predictions
    in both repeats. An adaptive policy has a literal name; fixed names include @value.
    """
    keys, off_names = tuple(keys), tuple(off_names)
    if len(off_names) != len(set(off_names)) or len(keys) != len(set(keys)):
        raise ValueError("Duplicate forecast names")
    by_name = {name: j for j, name in enumerate(off_names)}
    families = [k.rsplit("@", 1)[0] for k in keys]
    if any(name not in by_name for name in families):
        raise ValueError("Unregistered off family for a paired forecast")
    expected = [by_name[name] for name in families]
    if not np.array_equal(expected, off_indices):
        raise ValueError("A setting does not map to its own named off column")
    for b in batches:
        if len(np.unique(b["rows"])) != len(b["rows"]) or b["setting_off"].shape != (
            len(b["rows"]),
            len(by_name),
        ):
            raise ValueError("Duplicate rows or missing named baseline columns")
    result, critical, valid = household_evaluate(batches, keys, off_indices, target, draws, seed, width)
    for r, b in enumerate(batches):
        y = np.asarray(b["y"], dtype=np.float64)
        sst = fsum(np.square(y - fsum(y) / len(y)))
        if not np.isfinite(sst) or sst <= 0:
            raise ValueError("Unscorable held-out outcome")
        nulls = {f: sse(y, b["setting_off"][:, i]) for f, i in by_name.items()}
        for j, (key, name) in enumerate(zip(keys, families, strict=True)):
            apply_point(
                result[key]["repeats"][r], nulls[name], sse(y, b["setting_on"][:, j]), sst, target, critical
            )
    for key, name in zip(keys, families, strict=True):
        row = result[key]
        a, b = row["repeats"]
        if (a["repeat"], b["repeat"]) != (0, 1):
            raise ValueError("Expected two ordered repartitions for the same setting")
        row["off_family"] = name
        row["stable_both"] = all(s["ratio_status"] == "stable" for s in (a, b))
        ratios = a["ratio_pct"], b["ratio_pct"]
        row["minimum_ratio_pct"] = None if None in ratios else min(ratios)
        if not row["stable_both"]:
            row["minimum_ratio_ci95_pct"] = None
        for level in ("point", "adjusted"):
            flags = a["above_target_" + level], b["above_target_" + level]
            row["above_target_both_" + level] = None if None in flags else all(flags)
    return result, critical, valid
