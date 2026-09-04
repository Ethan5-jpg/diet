#!/usr/bin/env python3
"""Fit six unadjusted diet-score to primary-CGM linear regressions."""

from __future__ import print_function

import argparse
import math
import os

import numpy as np
import pandas as pd

from analysis_utils import (
    AMED_EXPOSURE,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_REPORT_DIR,
    EXPOSURES,
    HPDI_EXPOSURE,
    PRIMARY_CGM_OUTCOMES,
    numeric_series,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


def _beta_continued_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""

    max_iterations = 300
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c_value = 1.0
    d_value = 1.0 - qab * x / qap
    if abs(d_value) < tiny:
        d_value = tiny
    d_value = 1.0 / d_value
    result = d_value

    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        numerator = iteration * (b - iteration) * x
        denominator = (qam + even) * (a + even)
        adjustment = numerator / denominator
        d_value = 1.0 + adjustment * d_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = 1.0 + adjustment / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        result *= d_value * c_value

        numerator = -(a + iteration) * (qab + iteration) * x
        denominator = (a + even) * (qap + even)
        adjustment = numerator / denominator
        d_value = 1.0 + adjustment * d_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = 1.0 + adjustment / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        delta = d_value * c_value
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ValueError("Incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x, a, b):
    """Evaluate I_x(a,b) without requiring SciPy."""

    if x < 0.0 or x > 1.0:
        raise ValueError("Incomplete beta x must lie in [0, 1]")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Incomplete beta parameters must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    split = (a + 1.0) / (a + b + 2.0)
    if x < split:
        value = factor * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def student_t_two_sided_p(t_statistic, degrees_of_freedom):
    """Exact two-sided Student-t P value via the incomplete beta identity."""

    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    absolute_t = abs(float(t_statistic))
    if math.isinf(absolute_t):
        return 0.0
    x_value = degrees_of_freedom / (
        degrees_of_freedom + absolute_t * absolute_t
    )
    return regularized_incomplete_beta(
        x_value, degrees_of_freedom / 2.0, 0.5
    )


def student_t_cdf(t_value, degrees_of_freedom):
    if t_value == 0.0:
        return 0.5
    two_sided = student_t_two_sided_p(t_value, degrees_of_freedom)
    if t_value > 0.0:
        return 1.0 - two_sided / 2.0
    return two_sided / 2.0


def student_t_quantile(probability, degrees_of_freedom):
    """Invert the Student-t CDF by deterministic bisection."""

    if not (0.5 < probability < 1.0):
        raise ValueError("Only upper-half Student-t quantiles are supported")
    lower = 0.0
    upper = 1.0
    while student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
        if upper > 1.0e6:
            raise ValueError("Could not bracket Student-t quantile")
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if student_t_cdf(middle, degrees_of_freedom) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def fit_simple_ols(exposure, outcome):
    """Fit outcome = intercept + beta * exposure using pairwise finite rows."""

    x_values = pd.to_numeric(exposure, errors="coerce").astype(float)
    y_values = pd.to_numeric(outcome, errors="coerce").astype(float)
    valid = (
        x_values.notna()
        & y_values.notna()
        & np.isfinite(x_values)
        & np.isfinite(y_values)
    )
    x_array = x_values.loc[valid].to_numpy(dtype=float)
    y_array = y_values.loc[valid].to_numpy(dtype=float)
    count = int(len(x_array))
    if count < 3:
        raise ValueError("Unadjusted OLS requires at least three complete observations")

    x_mean = float(np.mean(x_array))
    y_mean = float(np.mean(y_array))
    centered_x = x_array - x_mean
    centered_y = y_array - y_mean
    sum_xx = float(np.sum(centered_x ** 2))
    if not sum_xx > 0.0:
        raise ValueError("Exposure has no variation in the analytic sample")
    total_yy = float(np.sum(centered_y ** 2))
    if not total_yy > 0.0:
        raise ValueError("Outcome has no variation in the analytic sample")

    beta = float(np.sum(centered_x * centered_y) / sum_xx)
    intercept = y_mean - beta * x_mean
    fitted = intercept + beta * x_array
    residuals = y_array - fitted
    residual_sum_squares = float(np.sum(residuals ** 2))
    degrees_of_freedom = count - 2
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = math.sqrt(max(0.0, residual_variance / sum_xx))
    if standard_error == 0.0:
        t_statistic = math.copysign(math.inf, beta) if beta != 0.0 else 0.0
    else:
        t_statistic = beta / standard_error
    p_value = student_t_two_sided_p(t_statistic, degrees_of_freedom)
    critical = student_t_quantile(0.975, degrees_of_freedom)
    ci_lower = beta - critical * standard_error
    ci_upper = beta + critical * standard_error
    r_squared = 1.0 - residual_sum_squares / total_yy

    return {
        "n": count,
        "residual_df": degrees_of_freedom,
        "intercept": intercept,
        "beta": beta,
        "standard_error": standard_error,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "r_squared": r_squared,
        "exposure_mean": x_mean,
        "exposure_sample_sd": float(np.std(x_array, ddof=1)),
        "outcome_mean": y_mean,
        "outcome_sample_sd": float(np.std(y_array, ddof=1)),
    }


def benjamini_hochberg(p_values):
    """Return BH-adjusted P values in the original order."""

    values = [float(value) for value in p_values]
    if any(value < 0.0 or value > 1.0 or not math.isfinite(value) for value in values):
        raise ValueError("All P values must be finite and lie in [0, 1]")
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [0.0] * count
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        original_index = order[reverse_index]
        rank = reverse_index + 1
        candidate = values[original_index] * count / rank
        running = min(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def run_unadjusted_models(aligned):
    """Run two diet exposures by three primary CGM outcomes."""

    aligned = aligned.copy()
    require_columns(
        aligned,
        ["participant_id"] + EXPOSURES + PRIMARY_CGM_OUTCOMES,
        "aligned diet-CGM table",
    )
    require_unique(aligned, ["participant_id"], "aligned diet-CGM table")
    for column in EXPOSURES + PRIMARY_CGM_OUTCOMES:
        aligned[column] = numeric_series(
            aligned, column, "aligned diet-CGM table", allow_missing=True
        )

    exposure_labels = {
        AMED_EXPOSURE: "AMED",
        HPDI_EXPOSURE: "hPDI",
    }
    rows = []
    for exposure in EXPOSURES:
        for outcome in PRIMARY_CGM_OUTCOMES:
            estimates = fit_simple_ols(aligned[exposure], aligned[outcome])
            row = {
                "diet_score": exposure_labels[exposure],
                "exposure": exposure,
                "outcome": outcome,
                "model_formula": "%s ~ %s" % (outcome, exposure),
                "model_type": "unadjusted_ordinary_least_squares",
                "covariate_count": 0,
            }
            row.update(estimates)
            rows.append(row)

    results = pd.DataFrame(rows)
    results["fdr_bh"] = benjamini_hochberg(results["p_value"].tolist())
    results["nominal_p_lt_0_05"] = results["p_value"].lt(0.05)
    results["fdr_bh_lt_0_05"] = results["fdr_bh"].lt(0.05)
    results = results.sort_values(
        ["diet_score", "outcome"], kind="mergesort"
    ).reset_index(drop=True)

    summary = {
        "aligned_participants_input": int(len(aligned)),
        "diet_exposures": int(len(EXPOSURES)),
        "primary_cgm_outcomes": int(len(PRIMARY_CGM_OUTCOMES)),
        "models_fitted": int(len(results)),
        "covariate_count": 0,
        "minimum_model_n": int(results["n"].min()),
        "maximum_model_n": int(results["n"].max()),
        "nominal_p_lt_0_05_models": int(results["nominal_p_lt_0_05"].sum()),
        "fdr_bh_lt_0_05_models": int(results["fdr_bh_lt_0_05"].sum()),
    }
    return results, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit six unadjusted diet-score to primary-CGM OLS models."
    )
    parser.add_argument(
        "--aligned-csv",
        default=os.path.join(DEFAULT_DATA_DIR, "00_diet_cgm_aligned.csv"),
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    aligned = read_csv_preserve_ids(args.aligned_csv)
    results, summary = run_unadjusted_models(aligned)

    result_path = os.path.join(
        args.model_dir, "01_unadjusted_diet_cgm_models.csv"
    )
    summary_path = os.path.join(
        args.report_dir, "01_unadjusted_model_summary.csv"
    )
    write_csv(results, result_path)
    write_summary(summary, summary_path)
    print_summary("01 UNADJUSTED DIET-CGM MODELS", summary)
    print("MODEL_FILE=%s" % os.path.abspath(result_path))


if __name__ == "__main__":
    main()
