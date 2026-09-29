"""Exploratory within-person models, participant-clustered covariance, no mediation."""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from .common import KEYS, norm, number, read_csv, require, unique, resolve_taxa, select_participant_scope


def bh_planned(pvalues):
    p = np.array(pvalues, dtype=float, copy=True)
    missing = ~np.isfinite(p)
    p[missing] = 1.0  # Keep unestimable planned tests in the family denominator.
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty(len(p))
    out[order] = adjusted
    out[missing] = np.nan
    return out


def within_fit(df, outcome, columns, min_df=30):
    """FWL person-visit FE; CR1 correction includes absorbed FE degrees of freedom."""
    require(len(df) > 0, "No complete observations")
    group = pd.MultiIndex.from_frame(df[KEYS]).factorize()[0]
    cols = [outcome] + list(columns)
    raw = df[cols].astype(float).reset_index(drop=True)
    require(np.isfinite(raw.to_numpy()).all(), "Nonfinite regression input")
    demeaned = raw - raw.groupby(group).transform("mean")
    y = demeaned[outcome].to_numpy()
    X = demeaned[list(columns)].to_numpy()
    scale = np.sqrt(np.mean(X ** 2, axis=0))
    active = scale > 1e-10
    dropped = [name for name, ok in zip(columns, active) if not ok]
    names = [name for name, ok in zip(columns, active) if ok]
    require("nova_per_10pp" in names, "No within-person NOVA variation")
    X, scale = X[:, active], scale[active]
    Z = X / scale
    n, k, n_fe = len(df), Z.shape[1], len(np.unique(group))
    u, singular, vt = np.linalg.svd(Z, full_matrices=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    require(np.linalg.matrix_rank(Z) == k and condition < 1e8, "Collinear/unstable within-person design; inspect variation")
    require(n - n_fe - k >= min_df, "Insufficient residual degrees of freedom")
    bz = vt.T.dot(u.T.dot(y) / singular)
    residual = y - Z.dot(bz)
    cluster, ids = pd.factorize(df.participant_id)
    n_clusters = len(ids)
    require(n_clusters >= 3, "Too few participant clusters")
    scores = np.zeros((n_clusters, k))
    np.add.at(scores, cluster, Z * residual[:, None])
    bread = (vt.T / singular ** 2).dot(vt)
    correction = n_clusters / (n_clusters - 1) * (n - 1) / (n - n_fe - k)
    cov_z = correction * bread.dot(scores.T.dot(scores)).dot(bread)
    beta = bz / scale
    se = np.sqrt(np.maximum(np.diag(cov_z), 0)) / scale
    critical = stats.t.ppf(0.975, n_clusters - 1)
    result = {}
    for name, b, s in zip(names, beta, se):
        require(np.isfinite(s) and s > 0, "Degenerate cluster standard error")
        result[name] = dict(estimate=float(b), se=float(s), ci_low=float(b - critical * s),
                            ci_high=float(b + critical * s), p=float(2 * stats.t.sf(abs(b / s), n_clusters - 1)))
    meal_hash = hashlib.sha256("\n".join(sorted(df.meal_id.astype(str))).encode()).hexdigest()
    return dict(coefficients=result, meals=n, people=n_clusters, fixed_effect_groups=n_fe,
                residual_df=n - n_fe - k, cluster_df=n_clusters - 1,
                design_condition_number=condition,
                dropped_constant_controls=";".join(dropped), sample_sha256=meal_hash)


def design(df):
    d = df.copy()
    d["nova_per_10pp"] = d.nova4_pct / 10
    d["carb_per_10g"] = d.carbohydrate_g / 10
    d["carb_per_10g_squared"] = d.carb_per_10g ** 2
    d["energy_per_100kcal"] = d.calories_kcal / 100
    d["baseline_per_100"] = d.baseline_glucose / 100
    d["hour_sin"] = np.sin(2 * np.pi * d.local_hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.local_hour / 24)
    d["previous_gap_hours_capped12"] = np.minimum(d.previous_gap_minutes / 60, 12)
    first = d.groupby(KEYS).start.transform("min")
    d["observation_day"] = (d.start - first) / 86400
    d["meal_duration_per_30min"] = d.duration_minutes / 30
    columns = ["nova_per_10pp", "carb_per_10g", "carb_per_10g_squared", "energy_per_100kcal",
               "baseline_per_100", "pre_slope_per_min", "hour_sin", "hour_cos",
               "previous_gap_hours_capped12", "observation_day", "meal_duration_per_30min"]
    return d, columns


def filter_variation(d, c):
    by = d.groupby(KEYS).nova4_pct
    keep = (by.transform("size").ge(c["minimum_meals_per_person"])
            & (by.transform("max") - by.transform("min")).ge(c["minimum_nova_range_pp"]))
    return d.loc[keep].copy()


def covariates(run):
    p = run.path("covariates")
    require(p is not None and p.is_file(), "Covariate master missing; cohort not established")
    run.record(p, "covariates")
    cols = run.c["analysis"]["require_covariate_columns"]
    d = read_csv(p, ["participant_id", "exclude_diabetes_or_a10"] + cols)
    d = select_participant_scope(run, d, "covariates")
    original = norm(d.exclude_diabetes_or_a10)
    flag = number(original)
    require(not (original.notna() & ~flag.isin([0, 1])).any(), "Invalid diabetes/A10 exclusion flag")
    complete = pd.Series(True, index=d.index)
    categorical = {"sex", "education_level", "smoking_status", "cgm_device_type"}
    for col in cols:
        valid = norm(d[col]).notna() if col in categorical else number(d[col]).notna()
        complete &= valid
    d["primary_eligible"] = ~flag.eq(1) & complete
    d["strict_eligible"] = flag.eq(0) & complete
    # People absent from this table are never treated as an unknown-but-eligible flag.
    return d[["participant_id", "primary_eligible", "strict_eligible"]]


def load_microbiome(run):
    p = run.path("microbiome")
    if p is None or not p.is_file():
        return None, [{"label": t["label"], "status": "MISSING_MICROBIOME"} for t in run.c["analysis"]["taxa"]]
    run.record(p, "microbiome")
    from .common import header
    all_cols = header(p)
    mappings = resolve_taxa(all_cols, run.c["analysis"]["taxa"])
    selected = [k for k in KEYS if k in all_cols] + [m["column"] for m in mappings if m["status"] == "FOUND"]
    require(len(selected) == len(set(selected)), "Two taxa resolve to the same microbiome column")
    d = read_csv(p, ["participant_id"], usecols=selected)
    d = select_participant_scope(run, d, "microbiome")
    for m in mappings:
        if m["status"] == "FOUND":
            d[m["column"]] = number(d[m["column"]])
    return d, mappings


def attempt(df, outcome, label, config, micro_column=None, coverage=0.8, strict=False):
    row = dict(model=label, outcome=outcome, status="NOT_ESTIMATED", estimate=np.nan,
               se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, people=0, meals=0)
    focus = "microbiome_x_nova" if micro_column else "nova_per_10pp"
    row["term"] = focus
    try:
        keep = df.primary_eligible.eq(True) & df.nova_coverage.ge(coverage)
        if strict:
            keep &= df.strict_eligible.eq(True)
        d, columns = design(df.loc[keep].copy())
        if micro_column:
            # Microbe measured once/person: M_i * within-person X deviation after FE.
            d["microbiome_x_nova"] = d[micro_column] * d.nova_per_10pp
            columns.append("microbiome_x_nova")
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + [outcome])
        row["complete_meals_before_variation_filter"] = len(d)
        row["complete_people_before_variation_filter"] = d.participant_id.nunique()
        d = filter_variation(d, config)
        row.update(people=d.participant_id.nunique(), meals=len(d))
        require(row["people"] >= config["minimum_people"], "Too few people after completeness/within-person variation filters")
        fit = within_fit(d, outcome, columns, config["minimum_residual_df"])
        require(focus in fit["coefficients"], "Focal term has no within-person variation")
        row.update(fit["coefficients"][focus])
        row.update({k: v for k, v in fit.items() if k != "coefficients"})
        row.update(status="ESTIMATED", reason="")
    except ValueError as e:
        row["reason"] = str(e)
    return row


def analyze(run, meals):
    if not run.c["time"]["consumption_time_verified"]:
        return dict(status="REVIEW_REQUIRED", reason="CONSUMPTION_TIME_NOT_VERIFIED; build outputs are provisional")
    try:
        cov = covariates(run)
        micro, mapping = load_microbiome(run)
    except ValueError as e:
        return dict(status="REVIEW_REQUIRED", reason=str(e))
    d = meals.loc[meals.analysis_ready].copy()
    d = d.merge(cov, on="participant_id", how="left", validate="many_to_one")
    timing = audit_microbiome_dates(run, d)
    flow = []
    for label, subset in [("valid_isolated_meals", d),
                          ("primary_covariate_eligible", d.loc[d.primary_eligible.eq(True)]),
                          ("strict_covariate_eligible", d.loc[d.strict_eligible.eq(True)])]:
        flow.append(dict(stage=label, meals=len(subset), people=subset.participant_id.nunique()))
    run.table(pd.DataFrame(flow), "03_cohort_flow.csv")
    a = run.c["analysis"]
    coverage = run.c["meal"]["nova_min_coverage"]
    rows = [attempt(d, "iauc_net_0_120", "primary_net_iauc", a, coverage=coverage),
            attempt(d, "peak_rise_0_120", "secondary_peak", a, coverage=coverage),
            attempt(d, "iauc_net_0_120", "sensitivity_strict", a, coverage=coverage, strict=True),
            attempt(d, "iauc_net_0_120", "sensitivity_coverage95", a, coverage=max(coverage, 0.95))]
    run.table(pd.DataFrame(rows), "03_within_person_models.csv")
    run.table(pd.DataFrame(mapping), "03_taxon_mapping.csv")
    interactions = []
    for m in mapping:
        label = m["label"]
        if m["status"] != "FOUND":
            interactions.append(dict(model=label, status="NOT_ESTIMATED", p=np.nan, reason=m["status"]))
            continue
        joined = d.merge(micro[["participant_id", m["column"]]], on="participant_id", how="left", validate="many_to_one")
        row = attempt(joined, "iauc_net_0_120", label, a, micro_column=m["column"], coverage=coverage)
        row["microbiome_scale"] = "original CLR-Z; not re-standardized by meal count"
        interactions.append(row)
    if interactions:
        iq = pd.DataFrame(interactions)
        iq["q_planned_taxa"] = bh_planned(iq.p.to_numpy(float))
        iq["planned_family_size"] = len(mapping)
        run.table(iq, "03_microbiome_interactions.csv")
    all_estimated = all(r["status"] == "ESTIMATED" for r in rows + interactions)
    return dict(status="PASS" if all_estimated else "PARTIAL", main_model=rows[0]["status"],
                estimated_models=sum(r["status"] == "ESTIMATED" for r in rows + interactions),
                planned_models=len(rows + interactions), microbiome_timing=timing,
                claim="Exploratory nutrient-conditional association, not a causal effect or mediator proof")


def audit_microbiome_dates(run, d):
    p = run.path("microbiome_dates")
    if p is None:
        return "UNKNOWN: no sample date file; do not call microbiome baseline"
    require(p.is_file(), "Configured microbiome_dates file missing: " + str(p))
    run.record(p, "microbiome_dates")
    dates = read_csv(p, KEYS + ["collection_date"])
    for key in KEYS:
        dates[key] = norm(dates[key])
    unique(dates, KEYS, "microbiome dates")
    # Explicit DATE-only input; reject mixed timestamps instead of dropping timezone silently.
    text = dates.collection_date.fillna("")
    require(text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(), "Microbiome dates must use YYYY-MM-DD")
    dates["sample_date"] = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    require(dates.sample_date.notna().all(), "Invalid microbiome dates")
    joined = d.merge(dates[KEYS + ["sample_date"]], on=KEYS, how="left", validate="many_to_one")
    delta = (joined.sample_date - pd.to_datetime(joined.local_date)).dt.days
    labels = pd.Series("unknown", index=joined.index)
    labels.loc[delta.lt(0)] = "before_meal"
    labels.loc[delta.eq(0)] = "same_date_order_unknown"
    labels.loc[delta.gt(0)] = "after_meal"
    run.table(pd.DataFrame({"timing": labels}).value_counts().rename("meal_count").reset_index(),
              "03_microbiome_timing.csv")
    return "DATE_AUDITED; same-date ordering unknown; interactions remain observational"
