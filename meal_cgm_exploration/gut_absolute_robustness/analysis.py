"""Fixed exploratory questions, common participant means, original HC3 inference."""
import numpy as np
import pandas as pd

from hppmeal.common import require
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import CATEGORIES
from seven_diet_meals.models import encode, linear_fit, sample_hash
from gut_meal_bridge_v2.analysis import estimate as original_estimate, model_columns
from gut_meal_bridge_v2.inputs import contract as original_contract
from .inputs import contract, selected, MODEL_KEYS


def original_fdr(rows):
    d = rows.copy()
    meal = original_contract()["meal_outcomes"]
    d["family"] = np.where(d.outcome.eq("mean_increment"), "net_primary",
                           np.where(d.outcome.isin(meal), "baseline_peak_secondary", "whole_cgm_reference"))
    d["q_family"] = np.nan
    d["q_meal12"] = np.nan
    for (_, family), ix in d.groupby(["setting", "family"]).groups.items():
        expected = {"net_primary": 4, "baseline_peak_secondary": 8, "whole_cgm_reference": 12}[family]
        require(len(ix) == expected, "Missing original FDR specifications")
        d.loc[ix, "q_family"] = bh_planned(d.loc[ix, "p"])
        d.loc[ix, "planned_family_size"] = expected
    for _, g in d.groupby("setting"):
        ix = g.index[g.outcome.isin(meal)]
        require(len(ix) == 12, "Missing original meal specifications")
        d.loc[ix, "q_meal12"] = bh_planned(d.loc[ix, "p"])
    return d


def compare_models(current, previous):
    require(len(current) == len(previous) == 48, "Expected 48 reproduction specifications")
    require(not current.duplicated(MODEL_KEYS).any() and not previous.duplicated(MODEL_KEYS).any(), "Duplicate reproduction specifications")
    merged = current.merge(previous, on=MODEL_KEYS, suffixes=("_new", "_old"), validate="one_to_one")
    require(len(merged) == 48, "Source reproduction keys differ")
    for name in ["status", "people", "meals", "person_sha256", "meal_sha256", "adjustment", "unit", "family"]:
        require(merged[name + "_new"].eq(merged[name + "_old"]).all(), "Source reproduction differs: " + name)
    for name in ["estimate", "se", "ci_low", "ci_high", "p", "q_family", "q_meal12"]:
        require(np.allclose(merged[name + "_new"], merged[name + "_old"],
                            equal_nan=True, **contract()["source_reproduction"]), "Source numerical reproduction differs: " + name)
    return merged[MODEL_KEYS + ["people_new", "meals_new"]].rename(columns={"people_new": "people", "meals_new": "meals"}).assign(reproduction="MATCHED")


def reproduce(persons, members, previous, progress=print):
    rows = []
    old = original_contract()
    for setting in contract()["settings"]:
        p, m = selected(persons, members, setting)
        for taxon in contract()["taxa"]:
            for outcome in old["meal_outcomes"] + old["reference_outcomes"]:
                rows.append(original_estimate(p, m, setting, taxon, outcome)[0])
        progress("REPRODUCE " + setting + ": 24 original specifications")
    current = original_fdr(pd.DataFrame(rows))
    return compare_models(current, previous)


def common_columns(taxon):
    cols = model_columns(taxon, "baseline_glucose")
    require("baseline_glucose" not in cols and "pre_slope_per_min" not in cols,
            "Common comparison accidentally adjusts for pre-meal outcome/slope")
    return cols


def estimate_common(persons, members, setting, taxon, outcome):
    c = contract()
    require(outcome in c["absolute_outcomes"], "Unknown absolute-comparison outcome")
    cols = common_columns(taxon)
    require(outcome not in cols, "Outcome included in own adjustment set")
    row = dict(setting=setting, threshold="m5d3", taxon=taxon, outcome=outcome,
               status="NOT_ESTIMATED", reason="", people=len(persons), meals=len(members),
               person_sha256=sample_hash(persons.participant_id), meal_sha256=sample_hash(members.meal_id),
               estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
               unit="mg/dL per original CLR-Z", adjustment=";".join(cols[1:]),
               model="common_adjustment_without_pre_or_slope",
               role="original_pre_reference" if outcome == "baseline_glucose" else "exploratory_extension")
    try:
        require(len(persons) >= c["minimum_people"], "Fewer than 50 common eligible participants")
        require(persons[taxon].nunique() >= 3, "Fewer than three distinct taxon values")
        require(persons[outcome].std(ddof=0) > 1e-10, "Constant outcome")
        fitted, info = linear_fit(persons[outcome], encode(persons, cols), taxon)
        row.update(next(x for x in fitted if x["term"] == taxon))
        row.update(info)
        row["status"] = "ESTIMATED"
    except (ValueError, np.linalg.LinAlgError) as error:
        row["reason"] = str(error)
    return row


def absolute_fdr(rows):
    d = rows.copy()
    d["q_panel12"] = np.nan
    for _, ix in d.groupby("setting").groups.items():
        require(len(ix) == contract()["fdr_absolute_per_setting"], "Missing A panel tests")
        d.loc[ix, "q_panel12"] = bh_planned(d.loc[ix, "p"])
    d["planned_family_size"] = contract()["fdr_absolute_per_setting"]
    return d


def sensitivity_fdr(rows):
    d = rows.copy()
    d["family"] = np.where(d.outcome.eq("mean_increment"), "net_primary", "baseline_peak_secondary")
    d["q_family"] = np.nan
    d["q_meal12"] = np.nan
    for (_, _, family), ix in d.groupby(["setting", "threshold", "family"]).groups.items():
        size = contract()["fdr_sensitivity_per_setting_and_threshold"][family]
        require(len(ix) == size, "Missing sensitivity FDR specifications")
        d.loc[ix, "q_family"] = bh_planned(d.loc[ix, "p"])
        d.loc[ix, "planned_family_size"] = size
    for _, ix in d.groupby(["setting", "threshold"]).groups.items():
        require(len(ix) == 12, "Missing sensitivity meal specifications")
        d.loc[ix, "q_meal12"] = bh_planned(d.loc[ix, "p"])
    return d


def identity_checks(absolute):
    rows = []
    for (setting, taxon), g in absolute.groupby(["setting", "taxon"]):
        require(len(g) == 3 and g.person_sha256.nunique() == g.meal_sha256.nunique() == g.adjustment.nunique() == 1,
                "A models do not share sample and design")
        status, delta = "NOT_EVALUABLE", np.nan
        if g.status.eq("ESTIMATED").all():
            b = g.set_index("outcome").estimate
            delta = b.absolute_post_mean - b.baseline_glucose - b.mean_increment
            require(np.isclose(b.absolute_post_mean, b.baseline_glucose + b.mean_increment,
                               rtol=contract()["identity_rtol"], atol=contract()["identity_atol"]), "Common-X algebra identity failed")
            status = "PASS_ALGEBRA_ONLY"
        rows.append(dict(setting=setting, taxon=taxon, status=status, coefficient_identity_residual=delta))
    require(len(rows) == 8, "Missing identity checks")
    return pd.DataFrame(rows)


def selection_balance(base, retained, setting, threshold):
    numeric = ["age_years", "bmi", "meals", "dates", "hPDI", "NOVA4", "mean_daily_energy_kcal",
               "baseline_glucose", "mean_increment", "absolute_post_mean", "peak_rise_0_120"] + contract()["taxa"]
    values = base.set_index("participant_id")[numeric].copy()
    for col in CATEGORIES:
        for level in sorted(base[col].astype(str).unique()):
            values[col + "=" + level] = base.set_index("participant_id")[col].astype(str).eq(level).astype(float)
    keep = values.index.isin(retained.participant_id)
    rows = []
    for field in values:
        a, b = values.loc[keep, field], values.loc[~keep, field]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd, smd_status = np.nan, "INSUFFICIENT_GROUP_SIZE"
        if len(a) >= 2 and len(b) >= 2:
            if a.nunique() == b.nunique() == 1:
                smd_status = "IDENTICAL_CONSTANTS" if a.iloc[0] == b.iloc[0] else "ZERO_VARIANCE_SEPARATION"
            elif np.isfinite(pooled) and pooled > 0:
                smd, smd_status = (a.mean() - b.mean()) / pooled, "ESTIMATED"
            else:
                smd_status = "UNDEFINED_VARIANCE"
        rows.append(dict(setting=setting, threshold=threshold, field=field, retained_people=len(a), excluded_people=len(b),
                         retained_mean=a.mean(), excluded_mean=b.mean(), retained_sd=a.std(ddof=1),
                         excluded_sd=b.std(ddof=1), standardized_difference=smd, smd_status=smd_status))
    return rows


def support_tables(persons, members):
    flows, balances, memberships = [], [], []
    for setting in contract()["settings"]:
        base, _ = selected(persons, members, setting)
        previous = set(base.participant_id)
        for t in contract()["thresholds"]:
            p, m = selected(persons, members, setting, t["meals"], t["dates"])
            ids = set(p.participant_id)
            require(ids.issubset(previous), "Support thresholds not nested")
            previous = ids
            flows.append(dict(setting=setting, threshold=t["name"], min_meals=t["meals"], min_dates=t["dates"],
                              people=len(p), meals=len(m), retained_fraction=len(p) / len(base) if len(base) else np.nan,
                              excluded_people=len(base) - len(p), meals_p50=p.meals.median(), dates_p50=p.dates.median(),
                              person_sha256=sample_hash(p.participant_id), meal_sha256=sample_hash(m.meal_id)))
            memberships.extend(dict(setting=setting, threshold=t["name"], participant_id=pid) for pid in p.participant_id)
            if t["name"] != "m5d3":
                balances.extend(selection_balance(base, p, setting, t["name"]))
    return pd.DataFrame(flows), pd.DataFrame(balances), pd.DataFrame(memberships, columns=["setting", "threshold", "participant_id"])


def analyze(persons, members, source_models, progress=print):
    absolute_rows, sensitivity_rows = [], []
    for setting in contract()["settings"]:
        p, m = selected(persons, members, setting)
        for taxon in contract()["taxa"]:
            for outcome in contract()["absolute_outcomes"]:
                absolute_rows.append(estimate_common(p, m, setting, taxon, outcome))
        progress("A " + setting + ": common-adjustment absolute/Pre/Net complete")
        for t in contract()["thresholds"]:
            p, m = selected(persons, members, setting, t["meals"], t["dates"])
            for taxon in contract()["taxa"]:
                for outcome in contract()["original_meal_outcomes"]:
                    if t["name"] == "m5d3":
                        row = source_models.loc[source_models.setting.eq(setting) & source_models.taxon.eq(taxon) & source_models.outcome.eq(outcome)].iloc[0].to_dict()
                    else:
                        row = original_estimate(p, m, setting, taxon, outcome)[0]
                    row.update(threshold=t["name"], model="original_adjustment", role="original_reference" if t["name"] == "m5d3" else "support_sensitivity")
                    sensitivity_rows.append(row)
            progress("B {} {}: {} people, {} meals".format(setting, t["name"], len(p), len(m)))
    absolute = absolute_fdr(pd.DataFrame(absolute_rows))
    sensitivity = sensitivity_fdr(pd.DataFrame(sensitivity_rows))
    require(len(absolute) == 24 and len(sensitivity) == 72, "Planned model specifications missing")
    require(not absolute.duplicated(MODEL_KEYS).any() and not sensitivity.duplicated(MODEL_KEYS + ["threshold"]).any(), "Duplicate model specification")
    for _, g in sensitivity.groupby(["setting", "threshold"]):
        require(g.person_sha256.nunique() == g.meal_sha256.nunique() == 1, "Sensitivity outcomes use different members")
    # A Pre is an explicit original reference; only its NEW panel q is different.
    old_pre = source_models.loc[source_models.outcome.eq("baseline_glucose")].set_index(MODEL_KEYS).sort_index()
    new_pre = absolute.loc[absolute.outcome.eq("baseline_glucose")].set_index(MODEL_KEYS).sort_index()
    for field in ["estimate", "se", "ci_low", "ci_high", "p"]:
        require(np.allclose(new_pre[field], old_pre[field], equal_nan=True, **contract()["source_reproduction"]), "A Pre differs from original reference")
    identity = identity_checks(absolute)
    return dict(absolute=absolute, sensitivity=sensitivity, identity=identity)
