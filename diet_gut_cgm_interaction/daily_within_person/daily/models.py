"""Prespecified repeated-day associations; FE and CR1 covariance in native units."""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from .common import require

EXPOSURES = {
    "NOVA4": dict(field="nova4_classified_pct", unit=10.0, range_key="minimum_nova_range_pp", label="10 percentage points of classified energy"),
    "Fiber": dict(field="fiber_density", unit=5.0, range_key="minimum_fiber_range_g_per_1000kcal", label="5 g/1000 kcal")
}
OUTCOMES = {"SD": "daily_sd", "Mean": "daily_mean", "CV": "daily_cv"}
SPECS = ["primary", "strict_known", "coverage95", "days7", "nutrient_context", "baseline_slopes", "equal_person_weight"]


def bh(pvalues):
    p = np.asarray(pvalues, float).copy()
    missing = ~np.isfinite(p)
    p[missing] = 1.0
    require(((p >= 0) & (p <= 1)).all(), "p-values outside [0,1]")
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1].clip(0, 1)
    out = np.empty(len(p))
    out[order] = q
    out[missing] = np.nan
    return out


def within_fit(d, outcome, columns, equal_person=False, min_df=100, max_condition=1e8):
    group, labels = pd.factorize(d.participant_id, sort=True)
    g, n = len(labels), len(d)
    require(g >= 3, "Too few participant clusters")
    names = [outcome] + list(columns)
    values = d[names].astype(float).reset_index(drop=True)
    require(np.isfinite(values.to_numpy()).all(), "Nonfinite regression values")
    counts = np.bincount(group)
    w = 1.0 / counts[group] if equal_person else np.ones(n)
    # General weighted projection onto the absorbed participant intercepts.
    sw = pd.Series(w).groupby(group).transform("sum")
    means = values.mul(w, axis=0).groupby(group).transform("sum").div(sw, axis=0)
    within = values - means
    y = within[outcome].to_numpy() * np.sqrt(w)
    X = within[list(columns)].to_numpy() * np.sqrt(w[:, None])
    scale = np.sqrt(np.mean(X ** 2, axis=0))
    active = scale > 1e-10
    dropped = [col for col, ok in zip(columns, active) if not ok]
    kept = [col for col, ok in zip(columns, active) if ok]
    require("x_within" in kept and "interaction" in kept, "No variation in focal within-person or interaction term")
    X, scale = X[:, active], scale[active]
    Z = X / scale
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    k = Z.shape[1]
    condition = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    require(np.linalg.matrix_rank(Z) == k and condition < max_condition, "Collinear or ill-conditioned within-person design")
    residual_df = n - g - k
    require(residual_df >= min_df, "Insufficient residual degrees of freedom")
    bz = vt.T.dot(u.T.dot(y) / s)
    residual = y - Z.dot(bz)
    scores = np.zeros((g, k))
    np.add.at(scores, group, Z * residual[:, None])
    bread = (vt.T / s ** 2).dot(vt)
    correction = g / (g - 1) * (n - 1) / residual_df
    covz = correction * bread.dot(scores.T.dot(scores)).dot(bread)
    beta, covariance = bz / scale, covz / np.outer(scale, scale)
    covariance = (covariance + covariance.T) / 2
    require(np.isfinite(covariance).all() and np.diag(covariance).min() >= -1e-12, "Invalid cluster covariance")
    critical = stats.t.ppf(0.975, g - 1)
    coefficients = {}
    for col, b, var in zip(kept, beta, np.diag(covariance)):
        se = float(np.sqrt(max(0, var)))
        require(se > 0, "Degenerate cluster standard error")
        coefficients[col] = dict(beta=float(b), se=se, low=float(b - critical * se), high=float(b + critical * se), p=float(2 * stats.t.sf(abs(b / se), g - 1)))
    leverage = np.sum(u ** 2, axis=1) + w / sw.to_numpy()
    focal = Z[:, kept.index("interaction")] ** 2
    person_info = np.bincount(group, weights=focal)
    total = float(y.dot(y))
    return dict(coefficients=coefficients, names=kept, covariance=covariance, n_days=n, n_people=g,
                residual_df=residual_df, cluster_df=g - 1, condition=condition,
                max_leverage=float(leverage.max()), max_person_focal_information_share=float(person_info.max() / person_info.sum()),
                within_r2=float(1 - residual.dot(residual) / total) if total > 0 else np.nan,
                dropped_constants=";".join(dropped),
                sample_sha256=hashlib.sha256("\n".join(sorted(d.day_id)).encode()).hexdigest())


def choose_sample(panel, config, exposure, taxon, outcome, spec):
    meta = EXPOSURES[exposure]
    field = meta["field"]
    c, a = config["diet"], config["analysis"]
    flow = []
    def record(label, frame):
        flow.append(dict(stage=label, n_days=len(frame), n_people=frame.participant_id.nunique()))
    d = panel.copy()
    record("matched_dates", d)
    d = d.loc[d.primary_eligible & d.diet_pass & d.cgm_pass].copy()
    if spec == "strict_known": d = d.loc[d.strict_eligible].copy()
    if spec == "coverage95": d = d.loc[d.cgm_strict].copy()
    coverage = c["strict_nova_coverage"] if spec == "coverage95" else c["minimum_nova_coverage"]
    if exposure == "NOVA4" or spec == "nutrient_context":
        d = d.loc[d.nova_coverage.ge(coverage)].copy()
    record("eligible_quality", d)
    cols = [field, taxon, OUTCOMES[outcome], "energy_kcal", "alcohol_g", "weekday", "observation_day"]
    if spec == "nutrient_context":
        cols += ["carb_density", "protein_density", "fiber_density", "nova4_classified_pct"]
    if spec == "baseline_slopes": cols += ["age_years", "bmi", "sex"]
    d = d.replace([np.inf, -np.inf], np.nan)
    # These are overlapping reasons within eligible-quality days, before the
    # complete-case and minimum-day/range filters. Never add them as a total.
    flow[-1].update(missing_exposure_days=int(d[field].isna().sum()),
                    missing_microbe_days=int(d[taxon].isna().sum()),
                    missing_alcohol_days=int(d.alcohol_g.isna().sum()),
                    missing_any_model_field_days=int(d[list(set(cols))].isna().any(axis=1).sum()))
    d = d.dropna(subset=list(set(cols))).copy()
    record("complete_model_fields", d)
    days = a["strict_days"] if spec == "days7" else a["minimum_days"]
    by = d.groupby("participant_id")[field]
    keep = by.transform("size").ge(days) & (by.transform("max") - by.transform("min")).ge(a[meta["range_key"]])
    d = d.loc[keep].copy()
    record("days_and_within_range", d)
    return d, flow


def make_design(d, exposure, taxon, spec, anchors):
    d = d.copy()
    meta = EXPOSURES[exposure]
    x = d[meta["field"]] / meta["unit"]
    xbar = x.groupby(d.participant_id).transform("mean")
    d["x_within"] = x - xbar
    m = d[taxon] - anchors["mean"]
    d["interaction"] = d.x_within * m
    d["energy_1000"] = d.energy_kcal / 1000
    d["alcohol_10g"] = d.alcohol_g / 10
    controls = ["energy_1000", "alcohol_10g", "observation_day"]
    for k in range(1, 7):
        col = "weekday_" + str(k)
        d[col] = d.weekday.eq(k).astype(float)
        controls.append(col)
    if spec == "nutrient_context":
        d["carb_density_50"] = d.carb_density / 50
        d["protein_density_25"] = d.protein_density / 25
        d["other_diet"] = d.fiber_density / 5 if exposure == "NOVA4" else d.nova4_classified_pct / 10
        controls += ["carb_density_50", "protein_density_25", "other_diet"]
    columns = ["x_within", "interaction"] + controls
    # Time-varying adjustment variables may themselves have microbe-dependent slopes.
    for col in controls:
        term = "microbe_x_" + col
        d[term] = m * d[col]
        columns.append(term)
    if spec == "baseline_slopes":
        for col in ["age_years", "bmi"]:
            centered = d[col] - d.drop_duplicates("participant_id")[col].mean()
            term = "diet_x_" + col
            d[term] = d.x_within * centered
            columns.append(term)
        for i, level in enumerate(sorted(d.sex.unique())[1:]):
            term = "diet_x_sex_" + str(i + 1)
            d[term] = d.x_within * d.sex.eq(level).astype(float)
            columns.append(term)
        d["diet_x_person_mean_diet"] = d.x_within * (xbar - xbar.groupby(d.participant_id).first().mean())
        columns.append("diet_x_person_mean_diet")
    return d, columns


def linear_contrast(fit, weights):
    vector = np.array([weights.get(name, 0.0) for name in fit["names"]])
    beta = np.array([fit["coefficients"][name]["beta"] for name in fit["names"]])
    estimate = float(vector.dot(beta))
    se = float(np.sqrt(max(0, vector.dot(fit["covariance"]).dot(vector))))
    critical = stats.t.ppf(0.975, fit["cluster_df"])
    return dict(beta=estimate, se=se, low=estimate - critical * se, high=estimate + critical * se)


def fit_all(panel, config):
    results, all_coefs, flows, slopes, anchors_out = [], [], [], [], []
    registry = [(e, t["id"], y) for e in EXPOSURES for t in config["taxa"] for y in OUTCOMES]
    for exposure, taxon, outcome in registry:
        primary, _ = choose_sample(panel, config, exposure, taxon, outcome, "primary")
        pm = primary.drop_duplicates("participant_id")[taxon]
        anchors = dict(mean=float(pm.mean()), q25=float(pm.quantile(.25)), q50=float(pm.quantile(.50)), q75=float(pm.quantile(.75)))
        anchors_out.append(dict(exposure=exposure, taxon=taxon, outcome=outcome, people=len(pm), **anchors))
        for spec in SPECS:
            row = dict(spec=spec, exposure=exposure, taxon=taxon, outcome=outcome,
                       unit="percentage_points" if outcome == "CV" else "mg/dL", exposure_unit=EXPOSURES[exposure]["label"],
                       status="NOT_ESTIMATED", reason="", n_days=0, n_people=0, beta=np.nan, low=np.nan, high=np.nan, p=np.nan,
                       limited_support=False, q_primary4=np.nan, q_all12=np.nan)
            d, flow = choose_sample(panel, config, exposure, taxon, outcome, spec)
            flows.extend(dict(spec=spec, exposure=exposure, taxon=taxon, outcome=outcome, **x) for x in flow)
            row.update(n_days=len(d), n_people=d.participant_id.nunique())
            try:
                require(row["n_people"] >= config["analysis"]["minimum_people"], "Too few people after daily quality/completeness/variation filters")
                require(all(np.isfinite(x) for x in anchors.values()), "Primary-sample microbial anchors unavailable")
                design, columns = make_design(d, exposure, taxon, spec, anchors)
                fit = within_fit(design, OUTCOMES[outcome], columns, spec == "equal_person_weight",
                                 config["analysis"]["minimum_residual_df"], config["analysis"]["maximum_condition_number"])
                row.update(fit["coefficients"]["interaction"])
                row.update({k: v for k, v in fit.items() if k not in ["coefficients", "covariance", "names"]})
                people = d.drop_duplicates("participant_id")
                counts = d.groupby("participant_id").size()
                ranges = d.groupby("participant_id")[EXPOSURES[exposure]["field"]].agg(lambda x: x.max() - x.min())
                low_n = int(people[taxon].le(anchors["q25"]).sum())
                high_n = int(people[taxon].ge(anchors["q75"]).sum())
                row.update(days_median=float(counts.median()), days_min=int(counts.min()), within_range_median=float(ranges.median()),
                           tail_low_people=low_n, tail_high_people=high_n,
                           microbial_iqr=anchors["q75"] - anchors["q25"],
                           sample_before_days=int(d.sample_minus_day.lt(0).sum()),
                           sample_same_day=int(d.sample_minus_day.eq(0).sum()), sample_after_days=int(d.sample_minus_day.gt(0).sum()),
                           sample_timing_unknown_days=int(d.sample_minus_day.isna().sum()))
                row["limited_support"] = (min(low_n, high_n) < config["analysis"]["minimum_tail_people"] or
                    anchors["q75"] <= anchors["q25"] or fit["max_person_focal_information_share"] > .1)
                row["status"] = "ESTIMATED"
                for term, coef in fit["coefficients"].items():
                    all_coefs.append(dict(spec=spec, exposure=exposure, taxon=taxon, outcome=outcome, term=term, **coef))
                # Marginal slopes in primary models: controls are held constant.
                # Baseline-slope sensitivity has extra interactions, so avoid presenting
                # its reference-category slope as an average-person estimate.
                if spec == "primary":
                    for label in ["q25", "q50", "q75"]:
                        value = linear_contrast(fit, {"x_within": 1, "interaction": anchors[label] - anchors["mean"]})
                        slopes.append(dict(exposure=exposure, taxon=taxon, outcome=outcome, point=label, microbe=anchors[label], **value))
                    value = linear_contrast(fit, {"interaction": anchors["q75"] - anchors["q25"]})
                    slopes.append(dict(exposure=exposure, taxon=taxon, outcome=outcome, point="q75_minus_q25", microbe=np.nan, **value))
            except ValueError as exc:
                # An unsuccessful diagnostic/contrast must not leave a usable
                # coefficient or p-value behind in the prespecified family.
                row.update(status="NOT_ESTIMATED", beta=np.nan, se=np.nan,
                           low=np.nan, high=np.nan, p=np.nan, limited_support=False)
                all_coefs = [r for r in all_coefs if (r["spec"], r["exposure"], r["taxon"], r["outcome"]) != (spec, exposure, taxon, outcome)]
                if spec == "primary":
                    slopes = [r for r in slopes if (r["exposure"], r["taxon"], r["outcome"]) != (exposure, taxon, outcome)]
                row["reason"] = str(exc)
            results.append(row)
        print("MODELS {}/{} hypotheses (7 specifications each)".format(len(results) // len(SPECS), len(registry)), flush=True)
    frame = pd.DataFrame(results)
    for spec in SPECS:
        ix = frame.spec.eq(spec)
        require(ix.sum() == 12, "Planned family must retain all 12 rows")
        frame.loc[ix, "q_all12"] = bh(frame.loc[ix, "p"])
        iy = ix & frame.outcome.eq("SD")
        require(iy.sum() == 4, "Primary endpoint family must retain four rows")
        frame.loc[iy, "q_primary4"] = bh(frame.loc[iy, "p"])
    coefficient_columns = ["spec", "exposure", "taxon", "outcome", "term", "beta", "se", "low", "high", "p"]
    slope_columns = ["exposure", "taxon", "outcome", "point", "microbe", "beta", "se", "low", "high"]
    return frame, pd.DataFrame(all_coefs, columns=coefficient_columns), pd.DataFrame(flows), pd.DataFrame(slopes, columns=slope_columns), pd.DataFrame(anchors_out)


def feasibility(panel, config):
    rows = []
    for exposure in EXPOSURES:
        for taxon in [t["id"] for t in config["taxa"]]:
            d, flow = choose_sample(panel, config, exposure, taxon, "SD", "primary")
            counts = d.groupby("participant_id").size()
            quality = next(r for r in flow if r["stage"] == "eligible_quality")
            rows.append(dict(exposure=exposure, taxon=taxon, people=d.participant_id.nunique(), days=len(d),
                             days_median=float(counts.median()) if len(counts) else np.nan,
                             days_p10=float(counts.quantile(.1)) if len(counts) else np.nan,
                             quality_days=quality["n_days"],
                             **{k: v for k, v in quality.items() if k.startswith("missing_")},
                             ready=d.participant_id.nunique() >= config["analysis"]["minimum_people"]))
    return pd.DataFrame(rows)
