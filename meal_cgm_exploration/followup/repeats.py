"""Repeated meal structure and participant-cluster bootstrap intervals."""
import numpy as np
import pandas as pd

from hppmeal.common import KEYS, require
from hppmeal.report import repeat_pairs

COHORTS = ["all_descriptive", "primary", "strict"]
OUTCOMES = ["baseline_glucose", "mean_glucose", "mean_increment", "peak_rise_0_120"]
GROUP = KEYS + ["signature"]
METRICS = ["pair_median_absolute_difference", "pair_p90_absolute_difference",
           "pair_mean_signed_difference", "person_mean_of_median_absolute_difference",
           "person_mean_of_mean_signed_difference"]


def checked_pairs(d, old):
    eligible = d.loc[d.analysis_ready & d.signature.fillna("").ne("")].copy()
    eligible = eligible.loc[eligible.groupby(GROUP).meal_id.transform("size").ge(2)]
    expected = repeat_pairs(eligible)
    cols = ["participant_id", "meal_a", "meal_b"]
    require(not old.duplicated(cols).any(), "Duplicate stored pairs")
    x, y = expected.sort_values(cols).reset_index(drop=True), old.sort_values(cols).reset_index(drop=True)
    require(len(x) == len(y) and x[cols].equals(y[cols]), "Stored pairs do not match the original pairing rule")
    for col in ["iauc_a", "iauc_b", "delta", "clock_difference_hours"]:
        require(np.allclose(x[col].to_numpy(float), y[col].to_numpy(float), rtol=1e-9, atol=1e-7),
                "Stored pair values differ from verified meal table: " + col)
    require(pd.concat([x.meal_a, x.meal_b]).is_unique, "A meal was used in multiple pairs")
    m = d.set_index("meal_id")
    a, b = m.loc[x.meal_a].reset_index(drop=True), m.loc[x.meal_b].reset_index(drop=True)
    result = x[cols].copy()
    for key in ["cohort", "research_stage", "signature"]:
        require(a[key].eq(b[key]).all(), "Pair identity/template mismatch")
    for cohort in ["primary", "strict"]:
        result[cohort + "_eligible"] = a[cohort + "_eligible"] & b[cohort + "_eligible"]
    for outcome in OUTCOMES:
        result[outcome] = b[outcome] - a[outcome]
    result["complete"] = np.isfinite(result[OUTCOMES].to_numpy(float)).all(axis=1)
    result["same_connection"] = a.connection_id.notna() & b.connection_id.notna() & a.connection_id.eq(b.connection_id)
    result["days_apart"] = (b.start - a.start) / 86400
    return result


def integer_weighted_quantile(sorted_values, sorted_weights, q):
    """Exactly numpy's linear quantile on the integer-weight expanded sample."""
    cumulative = np.cumsum(sorted_weights)
    total = int(cumulative[-1])
    require(total > 0, "Empty bootstrap sample")
    position = (total - 1) * q
    low, high = int(np.floor(position)), int(np.ceil(position))
    lo = sorted_values[np.searchsorted(cumulative, low, side="right")]
    hi = sorted_values[np.searchsorted(cumulative, high, side="right")]
    return float(lo + (hi - lo) * (position - low))


def difference_summary(pairs, cohort, draws=1000, seed=20260925, minimum_people=50):
    d = pairs.loc[pairs.complete].copy()
    if cohort != "all_descriptive":
        d = d.loc[d[cohort + "_eligible"]].copy()
    codes, people = pd.factorize(d.participant_id, sort=True)
    n = len(people)
    rows = []
    for oi, outcome in enumerate(OUTCOMES):
        values = d[outcome].to_numpy(float)
        status = "ESTIMATED" if n >= minimum_people else ("DESCRIPTIVE_ONLY" if n else "NO_PAIRS")
        base = dict(cohort=cohort, outcome=outcome, unit="mg/dL", pairs=len(d), people=n,
                    status=status, bootstrap_draws=draws if status == "ESTIMATED" else 0,
                    ci_method="participant_cluster_percentile", ci_scope="exploratory; fixed selected pairs")
        if not n:
            rows.extend(dict(base, metric=k, estimate=np.nan, ci_low=np.nan, ci_high=np.nan) for k in METRICS)
            continue
        counts = np.bincount(codes, minlength=n)
        signed = np.bincount(codes, weights=values, minlength=n) / counts
        medians = pd.Series(np.abs(values)).groupby(codes).median().reindex(range(n)).to_numpy()
        estimates = [np.median(np.abs(values)), np.quantile(np.abs(values), 0.9),
                     values.mean(), medians.mean(), signed.mean()]
        ci = np.full((2, len(METRICS)), np.nan)
        if status == "ESTIMATED":
            order = np.argsort(np.abs(values), kind="mergesort")
            absolute, sorted_codes = np.abs(values)[order], codes[order]
            sums = signed * counts
            rng = np.random.RandomState(seed)  # Same participant draws for all outcomes.
            boot = np.empty((draws, len(METRICS)))
            for i in range(draws):
                multiplicity = np.bincount(rng.randint(0, n, size=n), minlength=n)
                weights = multiplicity[sorted_codes]
                boot[i] = [integer_weighted_quantile(absolute, weights, 0.5),
                           integer_weighted_quantile(absolute, weights, 0.9),
                           np.dot(multiplicity, sums) / np.dot(multiplicity, counts),
                           np.dot(multiplicity, medians) / n, np.dot(multiplicity, signed) / n]
            ci = np.quantile(boot, [0.025, 0.975], axis=0)
        rows.extend(dict(base, metric=k, estimate=float(estimates[i]), ci_low=ci[0, i], ci_high=ci[1, i])
                    for i, k in enumerate(METRICS))
    return pd.DataFrame(rows)


def template_structure(d):
    """Full logged template repetitions, without imposing the pair clock caliper."""
    d = d.loc[d.analysis_ready & d.signature.fillna("").ne("")].copy()
    columns = GROUP + ["meals", "dates", "connections", "clock_span_hours", "primary_eligible", "strict_eligible"]
    sizes = d.groupby(GROUP).meal_id.transform("size")
    rows = []
    for key, g in d.loc[sizes.ge(2)].groupby(GROUP, sort=True):
        hours = np.sort(g.local_hour.dropna().to_numpy(float))
        span = 24 - np.max(np.diff(np.r_[hours, hours[0] + 24])) if len(hours) else np.nan
        item = dict(zip(GROUP, key))
        item.update(meals=len(g), dates=g.local_date.nunique(), connections=g.connection_id.nunique(),
                    clock_span_hours=span, primary_eligible=bool(g.primary_eligible.all()),
                    strict_eligible=bool(g.strict_eligible.all()))
        rows.append(item)
    return pd.DataFrame(rows, columns=columns)


def structure_summary(templates):
    rows = []
    for cohort in COHORTS:
        t = templates if cohort == "all_descriptive" else templates.loc[templates[cohort + "_eligible"].eq(True)]
        for minimum_dates in [2, 3, 4]:
            g = t.loc[t.dates.ge(minimum_dates)]
            rows.append(dict(cohort=cohort, minimum_distinct_dates=minimum_dates, templates=len(g),
                             people=g.participant_id.nunique(), meals=int(g.meals.sum()),
                             templates_clock_span_le_1h=int(g.clock_span_hours.le(1).sum()),
                             interpretation="Full templates, not the disjoint-pair sample; no ICC estimated"))
    return pd.DataFrame(rows)
