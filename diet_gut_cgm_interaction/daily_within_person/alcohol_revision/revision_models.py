"""Revised source policies; numerical FE/CR1/BH core is the verified parent core."""
import hashlib
import numpy as np
import pandas as pd
from daily.common import require
from daily.models import EXPOSURES, OUTCOMES, bh, within_fit, linear_contrast

ORIGINAL_SPECS = ["primary", "strict_known", "coverage95", "days7", "nutrient_context", "baseline_slopes", "equal_person_weight"]
SPECS = ORIGINAL_SPECS + ["alcohol_recovered", "primary_no_alcohol_control", "recovered_no_alcohol_control", "all_days_no_alcohol_control"]
POLICIES = {s: dict(mask="alcohol_observed_ok", adjust=True) for s in ORIGINAL_SPECS}
POLICIES.update({
    "alcohol_recovered": dict(mask="alcohol_recovered_ok", adjust=True),
    "primary_no_alcohol_control": dict(mask="alcohol_observed_ok", adjust=False),
    "recovered_no_alcohol_control": dict(mask="alcohol_recovered_ok", adjust=False),
    "all_days_no_alcohol_control": dict(mask=None, adjust=False),
})

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
    policy = POLICIES[spec]
    cols = [field, taxon, OUTCOMES[outcome], "energy_kcal", "weekday", "observation_day"]
    if policy["adjust"]:
        cols.append("alcohol_amed_g")
    if spec == "nutrient_context":
        cols += ["carb_density", "protein_density", "fiber_density", "nova4_classified_pct"]
    if spec == "baseline_slopes": cols += ["age_years", "bmi", "sex"]
    numeric = d.select_dtypes(include=[np.number]).columns
    d[numeric] = d[numeric].replace([np.inf, -np.inf], np.nan)
    # These are overlapping reasons within eligible-quality days, before the
    # complete-case and minimum-day/range filters. Never add them as a total.
    flow[-1].update(missing_exposure_days=int(d[field].isna().sum()),
                    missing_microbe_days=int(d[taxon].isna().sum()),
                    missing_alcohol_days=int(d.alcohol_amed_g.isna().sum()),
                    unavailable_policy_alcohol_days=int((~d[policy["mask"]]).sum()) if policy["mask"] else 0,
                    missing_any_model_field_days=int(d[list(set(cols))].isna().any(axis=1).sum()))
    if policy["mask"]:
        d = d.loc[d[policy["mask"]]].copy()
    record("alcohol_source_policy", d)
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
    controls = ["energy_1000", "observation_day"]
    if POLICIES[spec]["adjust"]:
        d["alcohol_10g"] = d.alcohol_amed_g / 10
        controls.insert(1, "alcohol_10g")
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
        print("MODELS {}/{} hypotheses ({} specifications each)".format(len(results) // len(SPECS), len(registry), len(SPECS)), flush=True)
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
                             unavailable_policy_alcohol_days=quality["unavailable_policy_alcohol_days"],
                             **{k: v for k, v in quality.items() if k.startswith("missing_")},
                             ready=d.participant_id.nunique() >= config["analysis"]["minimum_people"]))
    return pd.DataFrame(rows)
