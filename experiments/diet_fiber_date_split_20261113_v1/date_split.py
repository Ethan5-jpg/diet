"""Build strictly disjoint person-level calendar periods."""

import numpy as np


COLUMNS = ("Y_value", "fiber", "age_calendar_years", "sex_code", "energy_kcal", "carb", "protein")
PAIRINGS = {"early_to_late": ("early", "late"), "late_to_early": ("late", "early"),
            "early_same": ("early", "early"), "late_same": ("late", "late"), "all_same": ("all", "all")}


def period_means(rows, dates, values):
    """Assign odd extra days to the later half. Require two days per half."""
    if not (np.isfinite(values).all() and len(rows) == len(dates) == len(values)):
        raise ValueError("Invalid matched day inputs")
    order = np.lexsort((dates, rows))
    rows, dates, values = rows[order], dates[order], values[order]
    same = rows[1:] == rows[:-1]
    if np.isnat(dates).any() or np.any(same & (dates[1:] <= dates[:-1])):
        raise ValueError("Missing or repeated person-date")
    people, start, counts = np.unique(rows, return_index=True, return_counts=True)
    keep = counts >= 4
    people, start, counts = people[keep], start[keep], counts[keep]
    if not len(people):
        raise ValueError("No participants with at least two days in each period")
    middle, end = start + counts // 2, start + counts
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    means = {key: (cumulative[b] - cumulative[a]) / (b - a)[:, None]
             for key, a, b in (("early", start, middle), ("late", middle, end), ("all", start, end))}
    return people, means, counts // 2, counts - counts // 2


def design(exposure, outcome, dose_terms):
    fiber, age, sex, energy, carb, protein = exposure[:, 1:].T
    age = (age - 50) / 10
    dose = dose_terms(fiber)
    x = np.column_stack((np.ones(len(dose)), age, sex, energy / 1000, carb / 10, protein / 10,
                         dose, dose * age[:, None], dose * sex[:, None]))
    return outcome[:, 0], x, dose


def gap(first, second, iqr, critical):
    beta = (first[0] - second[0]) * iqr
    se = np.linalg.norm(first[1] - second[1], axis=0) * iqr
    return dict(mg_dl=float(beta[0]), ci95=[float(beta[0] - critical * se[0]), float(beta[0] + critical * se[0])])


def effect(fit, vector, iqr, critical):
    contrast = np.asarray(vector) * iqr
    point = float(fit[0] @ contrast)
    se = float(np.linalg.norm(fit[1] @ contrast))
    return dict(mg_dl=point, se=se, ci95=[point - critical * se, point + critical * se])
