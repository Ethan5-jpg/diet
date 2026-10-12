"""Frozen common-sample regressions and covariance-aware comparisons."""
import itertools

import numpy as np
import pandas as pd

from .common import (BASE_CATEGORY, BASE_NUMERIC, TAXA, OUTCOMES, bh, people_hash,
                     require, sample_hash)
from .estimators import contrast, fit_linear, paired_contrast

VARIANTS = [
    dict(name="F_joint", sample="F", taxa=TAXA, field="nova4_total_pct", nutrition=False, mean_adjust=False, family="primary"),
    dict(name="F_T068", sample="F", taxa=["T068"], field="nova4_total_pct", nutrition=False, mean_adjust=False, family="single"),
    dict(name="F_T096", sample="F", taxa=["T096"], field="nova4_total_pct", nutrition=False, mean_adjust=False, family="single"),
    dict(name="F_classified_joint", sample="F", taxa=TAXA, field="nova4_classified_pct", nutrition=False, mean_adjust=False, family="definition"),
    dict(name="N0_joint", sample="N", taxa=TAXA, field="nova4_total_pct", nutrition=False, mean_adjust=False, family="nutrition_sample"),
    dict(name="N1_joint", sample="N", taxa=TAXA, field="nova4_total_pct", nutrition=True, mean_adjust=False, family="nutrition_adjusted"),
    dict(name="F_mean_adjusted_joint", sample="F", taxa=TAXA, field="nova4_total_pct", nutrition=False, mean_adjust=True, family="conditional_sd"),
]
SCALES = ["B", "W"]


def select_samples(panel, config, protocol):
    flows = []
    def record(stage, d, sample):
        flows.append(dict(sample=sample, stage=stage, n_people=d.participant_id.nunique(), n_days=len(d)))
    d = panel.copy()
    record("saved_panel", d, "F")
    d = d.loc[d.primary_eligible & d.diet_pass & d.cgm_pass & d.alcohol_observed_ok &
              d.nova_coverage.ge(config["diet"]["minimum_nova_coverage"])].copy()
    record("eligibility_quality_alcohol", d, "F")
    fields = ["nova4_total_pct", "nova4_classified_pct", "energy_kcal", "alcohol_amed_g",
              "weekday", "observation_day"] + TAXA + list(OUTCOMES.values()) + BASE_NUMERIC
    d = d.dropna(subset=fields)
    record("joint_complete_fields", d, "F")
    def filter_person(frame):
        g = frame.groupby("participant_id").nova4_total_pct
        keep = g.transform("size").ge(protocol["minimum_days"]) & (g.transform("max") - g.transform("min")).ge(protocol["minimum_total_nova_range_pp"])
        return frame.loc[keep].sort_values(["participant_id", "date"]).reset_index(drop=True)
    full = filter_person(d)
    record("minimum_days_total_nova_range", full, "F")
    n = full.dropna(subset=["carb_density", "protein_density", "fiber_density"]).copy()
    record("complete_nutrition_within_F", n, "N")
    n = filter_person(n)
    record("minimum_days_total_nova_range", n, "N")
    require(set(n.day_id).issubset(full.day_id), "Nutrition sample is not nested in F")
    if len(n):
        require(n[["carb_density", "protein_density", "fiber_density"]].ge(0).all().all(), "Negative nutrient density")
    return {"F": full, "N": n}, pd.DataFrame(flows)


def anchors_for(full):
    people = full.drop_duplicates("participant_id")
    return {t: dict(mean=float(people[t].mean()), q25=float(people[t].quantile(.25)),
                    q50=float(people[t].quantile(.5)), q75=float(people[t].quantile(.75))) for t in TAXA}


def design_for(d, variant, scale, anchors):
    """Use precisely the supplied days; recompute means after selection."""
    z = d[["participant_id", "day_id"] + TAXA + BASE_NUMERIC + BASE_CATEGORY + list(OUTCOMES.values())].copy()
    z["x"] = d[variant["field"]].to_numpy() / 10
    z["energy"] = d.energy_kcal.to_numpy() / 1000
    z["alcohol"] = d.alcohol_amed_g.to_numpy() / 10
    z["observation_day"] = d.observation_day.to_numpy()
    controls = ["energy", "alcohol", "observation_day"]
    for k in range(1, 7):
        name = "weekday_" + str(k)
        z[name] = d.weekday.eq(k).astype(float).to_numpy()
        controls.append(name)
    if variant["nutrition"]:
        for source, name, unit in [("carb_density", "carb", 50), ("protein_density", "protein", 25), ("fiber_density", "fiber", 5)]:
            z[name] = d[source].to_numpy() / unit
            controls.append(name)
    if scale == "B":
        # Each eligible date has equal weight within a person's window; each
        # person has equal weight in the between regression.
        averaged = z.groupby("participant_id", sort=True)[["x"] + controls + list(OUTCOMES.values())].mean()
        baseline = z.drop_duplicates("participant_id").set_index("participant_id")[TAXA + BASE_NUMERIC + BASE_CATEGORY]
        z = averaged.merge(baseline, left_index=True, right_index=True, validate="one_to_one").reset_index()
        z["intercept"] = 1.0
    if variant["mean_adjust"]:
        # B uses square of the person's mean, W uses daily Mean squared.
        z["mean_control"] = (z.daily_mean - 100) / 50
        z["mean_control_squared"] = z.mean_control ** 2
        controls += ["mean_control", "mean_control_squared"]
    if scale == "B":
        # A constant control's microbe product duplicates a microbe main effect.
        # Drop this redundant block explicitly, rather than silently using pinv.
        controls = [c for c in controls if z[c].max() - z[c].min() > 1e-10]
    columns = (["intercept"] if scale == "B" else []) + ["x"]
    focal = ["x"]
    for t in variant["taxa"]:
        z["m_" + t] = z[t] - anchors[t]["mean"]
        if scale == "B":
            columns.append("m_" + t)
        name = "xm_" + t
        z[name] = z.x * z["m_" + t]
        columns.append(name)
        focal.append(name)
    columns += controls
    for t in variant["taxa"]:
        for c in controls:
            name = t + "_x_" + c
            z[name] = z["m_" + t] * z[c]
            columns.append(name)
    if scale == "B":
        columns += BASE_NUMERIC
        for c in BASE_CATEGORY:
            for i, level in enumerate(sorted(z[c].unique())[1:]):
                name = c + "_level_" + str(i + 1)
                z[name] = z[c].eq(level).astype(float)
                columns.append(name)
    return z, columns, focal


def model_registry():
    return [(v, scale, y) for v in VARIANTS for scale in SCALES
            for y in (["SD"] if v["mean_adjust"] else list(OUTCOMES))]


def comparison_registry():
    rows = []
    for y, t in itertools.product(OUTCOMES, TAXA):
        rows.append(dict(family="scale", label="W_minus_B", taxon=t, outcome=y, scale="W-B",
                         left=("F_joint", "W", y), right=("F_joint", "B", y)))
    for scale, y, t in itertools.product(SCALES, OUTCOMES, TAXA):
        for family, label, left, right in [
            ("joint", "joint_minus_single", "F_joint", "F_" + t),
            ("definition", "classified_minus_total", "F_classified_joint", "F_joint"),
            ("selection", "N0_minus_F", "N0_joint", "F_joint"),
            ("nutrition", "N1_minus_N0", "N1_joint", "N0_joint")]:
            rows.append(dict(family=family, label=label, taxon=t, outcome=y, scale=scale,
                             left=(left, scale, y), right=(right, scale, y)))
    for scale, t in itertools.product(SCALES, TAXA):
        rows.append(dict(family="conditional_sd", label="conditional_minus_base", taxon=t, outcome="SD", scale=scale,
                         left=("F_mean_adjusted_joint", scale, "SD"), right=("F_joint", scale, "SD")))
    require(len(rows) == 40, "Comparison registry changed")
    return rows


def sample_audit(samples):
    audits = []
    for v, scale, y in model_registry():
        d = samples[v["sample"]]
        audits.append(dict(variant=v["name"], scale=scale, outcome=y, sample=v["sample"],
                           n_people=d.participant_id.nunique(), n_days=len(d),
                           sample_sha256=sample_hash(d), people_sha256=people_hash(d)))
    audit = pd.DataFrame(audits)
    for sample in ["F", "N"]:
        require(audit.loc[audit["sample"].eq(sample), "sample_sha256"].nunique() == 1, "Common sample audit failed")
    return audit


def run_models(samples, protocol, synthetic=False):
    anchors = anchors_for(samples["F"])
    audit = sample_audit(samples)
    rows, coefs, diagnostics, slopes = [], [], [], []
    fits = {}
    minimum_people = protocol["synthetic_minimum_people"] if synthetic else protocol["minimum_people"]
    minimum_tail = protocol["synthetic_minimum_tail_people"] if synthetic else protocol["minimum_tail_people"]
    registry = model_registry()
    for i, (v, scale, y) in enumerate(registry, 1):
        key = (v["name"], scale, y)
        d = samples[v["sample"]]
        meta = dict(variant=v["name"], scale=scale, outcome=y, family=v["family"], sample=v["sample"],
                    n_people=d.participant_id.nunique(), n_days=len(d), sample_sha256=sample_hash(d),
                    exposure_field=v["field"], unit="mg/dL per NOVA4 +10pp x original CLR-Z +1")
        fit, failure = None, ""
        try:
            require(meta["n_people"] >= minimum_people, "Too few people after common sample filters")
            require(all(np.isfinite(a) for t in v["taxa"] for a in anchors[t].values()), "Missing F microbial anchors")
            design, columns, focal = design_for(d, v, scale, anchors)
            min_df = (protocol["minimum_within_residual_df"] if scale == "W" else
                      protocol["synthetic_minimum_between_residual_df"] if synthetic else protocol["minimum_between_residual_df"])
            fit = fit_linear(design, OUTCOMES[y], columns, focal, scale == "W", True, min_df,
                             protocol["maximum_condition_number"])
            fits[key] = fit
            for term, coef in fit["coefficients"].items():
                coefs.append(dict(variant=v["name"], scale=scale, outcome=y, term=term, **coef))
        except (ValueError, np.linalg.LinAlgError) as exc:
            failure = str(exc)
        diag = dict(meta, status="ESTIMATED" if fit else "NOT_ESTIMATED", reason=failure)
        if fit:
            diag.update({k: fit[k] for k in ["residual_df", "condition", "max_leverage", "correction", "dropped_constants"]})
        diagnostics.append(diag)
        people = d.drop_duplicates("participant_id")
        for t in v["taxa"]:
            row = dict(meta, taxon=t, status=diag["status"], reason=failure, beta=np.nan, se=np.nan, low=np.nan,
                       high=np.nan, p=np.nan, limited_support=False, q_family=np.nan, q_main_sd4=np.nan,
                       q_main_all8=np.nan, q_all44=np.nan)
            a = anchors[t]
            row.update(tail_low_people=int(people[t].le(a["q25"]).sum()), tail_high_people=int(people[t].ge(a["q75"]).sum()),
                       microbial_iqr=a["q75"] - a["q25"], joint_low_people=np.nan, joint_high_people=np.nan)
            if len(v["taxa"]) == 2:
                other = next(x for x in v["taxa"] if x != t)
                oa = anchors[other]
                near = (people[other] - oa["mean"]).abs().le((oa["q75"] - oa["q25"]) / 4)
                row["joint_low_people"] = int((near & people[t].le(a["q25"])).sum())
                row["joint_high_people"] = int((near & people[t].ge(a["q75"])).sum())
            if fit:
                name = "xm_" + t
                row.update(fit["coefficients"][name])
                row["max_person_partial_information_share"] = fit["focal_information_shares"][name]
                row["limited_support"] = (min(row["tail_low_people"], row["tail_high_people"]) < minimum_tail or
                                           not row["microbial_iqr"] > 0 or row["max_person_partial_information_share"] > .1)
                if len(v["taxa"]) == 2:
                    row["limited_support"] |= min(row["joint_low_people"], row["joint_high_people"]) < minimum_tail
                for point in ["q25", "q50", "q75"]:
                    value = contrast(fit, {"x": 1.0, name: a[point] - a["mean"]})
                    slopes.append(dict(variant=v["name"], scale=scale, outcome=y, taxon=t, point=point,
                                       microbe=a[point], other_microbe="F participant mean if included", **value))
                value = contrast(fit, {name: row["microbial_iqr"]})
                slopes.append(dict(variant=v["name"], scale=scale, outcome=y, taxon=t, point="q75_minus_q25", microbe=np.nan,
                                   other_microbe="F participant mean if included", **value))
            rows.append(row)
        if i % 4 == 0 or i == len(registry):
            print("COMMON MODELS {}/26".format(i), flush=True)
    results = pd.DataFrame(rows)
    require(len(results) == 44 and len(diagnostics) == 26, "Frozen model registry incomplete")
    for family, indices in results.groupby("family").groups.items():
        require(len(indices) == (4 if family == "conditional_sd" else 8), "Interaction family size changed")
        results.loc[indices, "q_family"] = bh(results.loc[indices, "p"])
    main = results.variant.eq("F_joint")
    results.loc[main, "q_main_all8"] = bh(results.loc[main, "p"])
    sd = main & results.outcome.eq("SD")
    results.loc[sd, "q_main_sd4"] = bh(results.loc[sd, "p"])
    results["q_all44"] = bh(results.p)
    comparisons = []
    for c in comparison_registry():
        row = {k: v for k, v in c.items() if k not in ["left", "right"]}
        row.update(left_variant=c["left"][0], right_variant=c["right"][0], status="NOT_ESTIMATED", reason="Required model not estimated",
                   beta=np.nan, se=np.nan, low=np.nan, high=np.nan, p=np.nan, q_family=np.nan, q_scale_sd2=np.nan,
                   q_all40=np.nan, limited_support=False, unit="difference of mg/dL interaction coefficients on original CLR-Z scale")
        if c["left"] in fits and c["right"] in fits:
            value = paired_contrast(fits[c["left"]], fits[c["right"]], {"xm_" + c["taxon"]: 1}, {"xm_" + c["taxon"]: 1})
            row.update(value, status="ESTIMATED", reason="")
            for model_key in [c["left"], c["right"]]:
                part = results.loc[results.variant.eq(model_key[0]) & results.scale.eq(model_key[1]) &
                                   results.outcome.eq(model_key[2]) & results.taxon.eq(c["taxon"])]
                row["limited_support"] |= bool(part.iloc[0].limited_support)
        comparisons.append(row)
    comparisons = pd.DataFrame(comparisons)
    for family, indices in comparisons.groupby("family").groups.items():
        require(len(indices) == (4 if family in ["scale", "conditional_sd"] else 8), "Contrast family size changed")
        comparisons.loc[indices, "q_family"] = bh(comparisons.loc[indices, "p"])
    sd = comparisons.family.eq("scale") & comparisons.outcome.eq("SD")
    comparisons.loc[sd, "q_scale_sd2"] = bh(comparisons.loc[sd, "p"])
    comparisons["q_all40"] = bh(comparisons.p)
    anchor_table = pd.DataFrame([dict(taxon=t, **a) for t, a in anchors.items()])
    coefficient_columns = ["variant", "scale", "outcome", "term", "beta", "se", "low", "high", "p"]
    slope_columns = ["variant", "scale", "outcome", "taxon", "point", "microbe", "other_microbe", "beta", "se", "low", "high", "p"]
    return dict(interactions=results, comparisons=comparisons, coefficients=pd.DataFrame(coefs, columns=coefficient_columns),
                diagnostics=pd.DataFrame(diagnostics), slopes=pd.DataFrame(slopes, columns=slope_columns), anchors=anchor_table, sample_audit=audit)


def describe_samples(samples):
    rows = []
    for name, d in samples.items():
        people = d.drop_duplicates("participant_id")
        days = d.groupby("participant_id").size()
        ranges = d.groupby("participant_id").nova4_total_pct.agg(lambda x: x.max() - x.min())
        rows.append(dict(sample=name, n_people=len(people), n_days=len(d), median_days=float(days.median()),
                         min_days=int(days.min()) if len(days) else 0, median_total_nova_range=float(ranges.median()),
                         microbe_pearson=float(people[TAXA].corr().iloc[0, 1]) if len(people) > 1 else np.nan,
                         sample_sha256=sample_hash(d), people_sha256=people_hash(d)))
    return pd.DataFrame(rows)
