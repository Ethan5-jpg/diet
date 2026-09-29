"""Matched participants, equal person weights, bounded HC3 associations."""
import numpy as np
import pandas as pd
from hppmeal.common import require
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import BASE
from seven_diet_meals.models import features, encode, linear_fit, sample_hash, NUTRIENTS
from .inputs import contract

TIMING = ["hour_sin", "hour_cos", "gap_hours", "observation_day", "duration_30"]
PREMEAL = ["baseline_glucose", "pre_slope_per_min"]


def prepare(meals, people):
    c = contract()
    taxa = [t["id"] for t in c["taxa"]]
    d = features(meals)
    person = people.reset_index()
    joined = d.merge(person, on="participant_id", how="left", validate="many_to_one")
    flow = []

    def mark(stage, frame):
        flow.append(dict(stage=stage, people=frame.participant_id.nunique(), meals=len(frame)))

    mark("analysis_ready_meals", joined)
    joined = joined.loc[joined.primary_eligible.eq(True)].copy()
    mark("primary_background_eligible", joined)
    complete = taxa + BASE + c["diet_covariates"]
    joined = joined.loc[joined[complete].notna().all(axis=1)].copy()
    mark("four_taxa_and_diet_complete", joined)
    connection = joined.connection_id.eq(joined.reference_connection) & joined.reference_connection.notna()
    mismatch = dict(people=int(joined.loc[~connection, "participant_id"].nunique()), meals=int((~connection).sum()))
    joined = joined.loc[connection & joined[c["reference_outcomes"]].notna().all(axis=1)].copy()
    mark("same_connection_and_reference_complete", joined)
    fields = list(dict.fromkeys(c["meal_outcomes"] + TIMING + PREMEAL + NUTRIENTS))
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna(subset=fields).copy()
    mark("same_complete_meal_fields", joined)
    group = joined.groupby("participant_id")
    enough = group.meal_id.transform("size").ge(c["minimum_meals"]) & group.local_date.transform("nunique").ge(c["minimum_dates"])
    joined = joined.loc[enough].copy().reset_index(drop=True)
    mark("at_least_5_meals_3_dates", joined)
    counts = joined.groupby("participant_id").agg(meals=("meal_id", "size"), dates=("local_date", "nunique"),
        first_date=("local_date", "min"), last_date=("local_date", "max"))
    averages = joined.groupby("participant_id")[fields].mean()
    fixed = taxa + BASE + c["diet_covariates"] + c["reference_outcomes"] + ["strict_eligible", "sample_date", "timing_status", "reference_connection"]
    static = joined.drop_duplicates("participant_id").set_index("participant_id")[fixed]
    persons = averages.join(static).join(counts).reset_index()
    if len(persons):
        persons["sample_date"] = pd.to_datetime(persons.sample_date)
        persons["stool_minus_first_meal_days"] = (persons.sample_date - pd.to_datetime(persons.first_date)).dt.days
        persons["stool_minus_last_meal_days"] = (persons.sample_date - pd.to_datetime(persons.last_date)).dt.days
    else:
        persons["stool_minus_first_meal_days"] = np.nan
        persons["stool_minus_last_meal_days"] = np.nan
    members = joined[["participant_id", "meal_id", "strict_eligible"]].copy()
    return persons, members, pd.DataFrame(flow), mismatch


def model_columns(taxon, outcome):
    c = contract()
    columns = [taxon] + BASE + c["diet_covariates"]
    if outcome in c["meal_outcomes"]:
        columns += TIMING + NUTRIENTS
        if outcome != "baseline_glucose":
            columns += PREMEAL
    require(outcome not in columns, "Outcome accidentally included in its own adjustment set")
    return columns


def estimate(persons, members, setting, taxon, outcome):
    c = contract()
    columns = model_columns(taxon, outcome)
    row = dict(setting=setting, taxon=taxon, outcome=outcome, status="NOT_ESTIMATED", reason="",
        people=len(persons), meals=len(members), person_sha256=sample_hash(persons.participant_id),
        meal_sha256=sample_hash(members.meal_id), estimate=np.nan, se=np.nan, ci_low=np.nan,
        ci_high=np.nan, p=np.nan, unit="mg/dL per original CLR-Z" if outcome in c["meal_outcomes"]
        else "original whole-CGM Z per original CLR-Z", adjustment=";".join(columns[1:]))
    coefficients = []
    try:
        require(len(persons) >= c["minimum_people"], "Fewer than 50 common eligible participants")
        require(persons[taxon].nunique() >= 3, "Fewer than three distinct taxon values")
        require(persons[outcome].std(ddof=0) > 1e-10, "Constant outcome")
        rows, info = linear_fit(persons[outcome], encode(persons, columns), taxon)
        require(info["residual_df"] >= c["minimum_residual_df"], "Insufficient residual degrees of freedom")
        focus = next(r for r in rows if r["term"] == taxon)
        row.update(focus)
        row.update(info)
        row["status"] = "ESTIMATED"
        coefficients = [dict(setting=setting, taxon=taxon, outcome=outcome, **r) for r in rows]
    except (ValueError, np.linalg.LinAlgError) as error:
        row["reason"] = str(error)
    return row, coefficients


def analyze(persons, members, progress=print):
    c = contract()
    outcomes = c["meal_outcomes"] + c["reference_outcomes"]
    rows, coefficients, sample_rows, timing_rows, description = [], [], [], [], []
    for setting in c["settings"]:
        p = persons if setting == "primary_common" else persons.loc[persons.strict_eligible.eq(True)]
        p = p.sort_values("participant_id").reset_index(drop=True)
        m = members.loc[members.participant_id.isin(p.participant_id)].sort_values("meal_id")
        sample_rows.append(dict(setting=setting, people=len(p), meals=len(m),
            meals_p50=p.meals.median(), dates_p50=p.dates.median(),
            person_sha256=sample_hash(p.participant_id), meal_sha256=sample_hash(m.meal_id)))
        dates = p.sample_date.notna()
        known_gaps = p.stool_minus_first_meal_days.dropna()
        timing_rows.append(dict(setting=setting, people=len(p), known_date=int(dates.sum()),
            unknown_date=int((~dates).sum()), ambiguous=int(p.timing_status.eq("AMBIGUOUS_MULTIPLE_SAMPLES").sum()),
            before=int((dates & p.stool_minus_first_meal_days.lt(0)).sum()),
            within=int((dates & p.stool_minus_first_meal_days.ge(0) & p.stool_minus_last_meal_days.le(0)).sum()),
            after=int((dates & p.stool_minus_last_meal_days.gt(0)).sum()),
            first_gap_p50=known_gaps.median() if len(known_gaps) else np.nan,
            distant_over_30d=int((dates & ((p.stool_minus_first_meal_days < -30) | (p.stool_minus_last_meal_days > 30))).sum())))
        for field in outcomes + [x["id"] for x in c["taxa"]]:
            values = p[field]
            description.append(dict(setting=setting, field=field, people=len(values), mean=values.mean(),
                sd=values.std(), q25=values.quantile(.25), median=values.median(), q75=values.quantile(.75)))
        for taxon in c["taxa"]:
            progress("MODEL {} {} (3 meal + 3 reference outcomes)".format(setting, taxon["id"]))
            for outcome in outcomes:
                row, coef = estimate(p, m, setting, taxon["id"], outcome)
                rows.append(row)
                coefficients.extend(coef)
    results = pd.DataFrame(rows)
    results["family"] = np.where(results.outcome.eq("mean_increment"), "net_primary",
        np.where(results.outcome.isin(c["meal_outcomes"]), "baseline_peak_secondary", "whole_cgm_reference"))
    results["q_family"] = np.nan
    results["q_meal12"] = np.nan
    for (setting, family), indices in results.groupby(["setting", "family"]).groups.items():
        expected = c["fdr"][family]
        require(len(indices) == expected, "Incorrect planned multiple-testing family")
        results.loc[indices, "q_family"] = bh_planned(results.loc[indices, "p"])
        results.loc[indices, "planned_family_size"] = expected
    for setting, group in results.groupby("setting"):
        indices = group.index[group.outcome.isin(c["meal_outcomes"])]
        require(len(indices) == 12, "Expected twelve meal tests")
        results.loc[indices, "q_meal12"] = bh_planned(results.loc[indices, "p"])
        require(group.person_sha256.nunique() == 1 and group.meal_sha256.nunique() == 1,
                "Common sample differs across taxa or outcomes")
    require(len(results) == c["planned_models"], "Missing planned models")
    require(set(persons.loc[persons.strict_eligible.eq(True), "participant_id"]).issubset(set(persons.participant_id)),
            "Strict membership is not a subset")
    return dict(models=results, coefficients=pd.DataFrame(coefficients, columns=["setting", "taxon", "outcome", "term", "estimate", "se", "ci_low", "ci_high", "p"]),
        sample=pd.DataFrame(sample_rows), timing=pd.DataFrame(timing_rows), descriptive=pd.DataFrame(description))
