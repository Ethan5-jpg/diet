"""Reproduce frozen results before prespecified person/meal quality filtering."""
import numpy as np
import pandas as pd

from hppmeal.common import require
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import DIETS
from seven_diet_meals import models as diet
from gut_meal_bridge_v2 import analysis as gut
from gut_meal_bridge_v2.inputs import contract
from .coverage import attach, quality_mask

DIET_KEYS = ["diet", "branch", "outcome", "model"]
GUT_KEYS = ["setting", "taxon", "outcome"]
QUALITY_SETTINGS = ["qc_primary", "qc_strict", "qc10", "qc20"]


def settings(main_diet=.05):
    require(main_diet in [.05, .10], "Diet primary threshold must be fixed as 0.05 or 0.10")
    return [("qc_primary", .05, main_diet, False), ("qc_strict", .05, main_diet, True),
            ("qc10", .10, .10, False), ("qc20", .20, .20, False)]


def reproduce(current, previous, keys, qcols):
    require(len(current) == len(previous) and len(current) > 0, "Source model count differs")
    require(not current.duplicated(keys).any() and not previous.duplicated(keys).any(), "Duplicate source model specifications")
    d = current.merge(previous, on=keys, validate="one_to_one", suffixes=("_new", "_old"))
    require(len(d) == len(current), "Source model keys differ")
    for col in ["status", "people", "meals", "person_sha256", "meal_sha256"]:
        require(d[col+"_new"].eq(d[col+"_old"]).all(), "Frozen source mismatch: " + col)
    for col in ["estimate", "se", "ci_low", "ci_high", "p"] + qcols:
        require(np.allclose(d[col+"_new"].to_numpy(float), d[col+"_old"].to_numpy(float),
                            rtol=1e-6, atol=1e-9, equal_nan=True), "Frozen source coefficients differ: " + col)
    return d[keys + ["people_new", "meals_new"]].assign(reproduction="MATCHED")


def diet_fdr(table):
    d = table.copy()
    d["family"] = np.where(d.model.eq("background"), "background_14",
                           np.where(d.outcome.eq("mean_increment"), "primary_14", "peak_secondary_14"))
    d["q_family14"] = np.nan
    d["q_all42"] = np.nan
    for _, ix in d.groupby(["setting", "family"]).groups.items():
        require(len(ix) == 14, "Expected all fourteen planned diet family tests")
        d.loc[ix, "q_family14"] = bh_planned(d.loc[ix, "p"])
    for _, ix in d.groupby("setting").groups.items():
        require(len(ix) == 42, "Expected all forty-two planned diet tests")
        d.loc[ix, "q_all42"] = bh_planned(d.loc[ix, "p"])
    d["planned_family_size"] = 14
    return d


def gut_fdr(table):
    d = table.copy()
    c = contract()
    d["family"] = np.where(d.outcome.eq("mean_increment"), "net_primary",
        np.where(d.outcome.isin(c["meal_outcomes"]), "baseline_peak_secondary", "whole_cgm_reference"))
    for field in ["q_family", "q_meal12", "q_all24"]:
        d[field] = np.nan
    for (_, family), ix in d.groupby(["setting", "family"]).groups.items():
        require(len(ix) == c["fdr"][family], "Incorrect planned taxon family size")
        d.loc[ix, "q_family"] = bh_planned(d.loc[ix, "p"])
        d.loc[ix, "planned_family_size"] = len(ix)
    for _, group in d.groupby("setting"):
        meal_ix = group.index[group.outcome.isin(c["meal_outcomes"])]
        require(len(meal_ix) == 12 and len(group) == 24, "Expected all planned taxon tests")
        d.loc[meal_ix, "q_meal12"] = bh_planned(d.loc[meal_ix, "p"])
        d.loc[group.index, "q_all24"] = bh_planned(group.p)
        require(group.person_sha256.nunique() == 1 and group.meal_sha256.nunique() == 1,
                "Taxon/outcome common membership differs")
    return d


def fit_diet_sample(sample, score, label, branch):
    rows, coefficients = [], []
    for outcome in diet.OUTCOMES:
        for model in (["background", "meal_adjusted"] if branch == "A" else ["carb_interaction"]):
            row, coef = diet.attempt(sample, score, label, branch, outcome, model)
            rows.append(row)
            coefficients.extend(coef)
    return rows, coefficients


def selection_balance(base, retained, module, score, branch, setting):
    """Describe participant selection; SMD does not test away selection bias."""
    gut_fields = ["hPDI", "NOVA4"] + [x["id"] for x in contract()["taxa"]] if module == "gut" else []
    fields = [c for c in ["age_years", "bmi", score, "mean_daily_energy_kcal", "mean_increment",
                          "baseline_glucose", "carbohydrate_g", "calories_kcal"] + gut_fields if c and c in base]
    persons = base.groupby("participant_id")[fields].mean()
    if "meal_id" in base:
        persons["meals"] = base.groupby("participant_id").meal_id.size()
    static = base.drop_duplicates("participant_id").set_index("participant_id")
    for col in ["sex", "cgm_device_type"]:
        if col in static:
            for level in sorted(static[col].dropna().astype(str).unique()):
                persons[col+"="+level] = static[col].astype(str).eq(level).astype(float)
    keep = persons.index.isin(retained)
    rows = []
    for field in persons:
        a, b = persons.loc[keep, field].dropna(), persons.loc[~keep, field].dropna()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (a.mean() - b.mean()) / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append(dict(module=module, setting=setting, diet=score, branch=branch, field=field,
            retained_people=len(a), excluded_people=len(b), retained_mean=a.mean(), excluded_mean=b.mean(),
            retained_sd=a.std(), excluded_sd=b.std(), standardized_difference=smd))
    return rows


def aggregate_gut(full_features, source_persons, source_members, pool, strict=False):
    """Keep features computed BEFORE filtering; update per-person averages/counts."""
    c = contract()
    selected = source_persons.loc[source_persons.strict_eligible.eq(True)] if strict else source_persons
    ids = set(pool.meal_id) & set(source_members.loc[source_members.participant_id.isin(selected.participant_id), "meal_id"])
    d = full_features.loc[full_features.meal_id.isin(ids)].copy()
    g = d.groupby("participant_id")
    d = d.loc[g.meal_id.transform("size").ge(5) & g.local_date.transform("nunique").ge(3)].copy()
    fields = list(dict.fromkeys(c["meal_outcomes"] + gut.TIMING + gut.PREMEAL + diet.NUTRIENTS))
    averages = d.groupby("participant_id")[fields].mean()
    static_fields = [x for x in source_persons.columns if x not in fields + ["meals", "dates", "first_date", "last_date",
                                      "stool_minus_first_meal_days", "stool_minus_last_meal_days"]]
    static = source_persons.set_index("participant_id")[[x for x in static_fields if x != "participant_id"]]
    counts = d.groupby("participant_id").agg(meals=("meal_id", "size"), dates=("local_date", "nunique"),
                                              first_date=("local_date", "min"), last_date=("local_date", "max"))
    persons = averages.join(static, how="left").join(counts).reset_index()
    persons["sample_date"] = pd.to_datetime(persons.sample_date)
    persons["stool_minus_first_meal_days"] = (persons.sample_date - pd.to_datetime(persons.first_date)).dt.days
    persons["stool_minus_last_meal_days"] = (persons.sample_date - pd.to_datetime(persons.last_date)).dt.days
    members = d[["participant_id", "meal_id"]].copy()
    return persons.sort_values("participant_id").reset_index(drop=True), members.sort_values("meal_id"), d


def threshold_summary(quality, tagged, main_diet=.05):
    rows = []
    for label, cg, dt, strict in settings(main_diet):
        # 'strict' refers to model covariate eligibility, not raw-cache QC universe.
        audit = quality.loc[quality.status.eq("AUDITED") & quality.connection_unambiguous.eq(True) & quality.unresolved_diet_time.eq(False)]
        kept = audit.loc[quality_mask(audit, cg, dt)]
        meal = tagged.loc[quality_mask(tagged, cg, dt, meal=True)]
        rows.append(dict(setting=label, cgm_missing_rate_lt=cg, diet_missing_day_rate_lt=dt,
            raw_auditable_people=audit.participant_id.nunique(), cgm_pass_people=audit.loc[audit.cgm_missing_rate.lt(cg)].participant_id.nunique(),
            diet_pass_people=audit.loc[audit.diet_missing_day_rate.lt(dt)].participant_id.nunique(),
            joint_pass_people=kept.participant_id.nunique(), in_window_analysis_ready_meals=len(meal),
            in_window_people=meal.participant_id.nunique(), strict_covariate_filter_applied="IN_MODELS_ONLY" if strict else "NO"))
    return pd.DataFrame(rows)


def timing(persons, quality, label, timezone):
    p = persons.merge(quality[["participant_id", "connection_id", "window_start", "window_end", "anchor_source"]],
        left_on=["participant_id", "reference_connection"], right_on=["participant_id", "connection_id"],
        how="left", validate="one_to_one")
    first = pd.to_datetime(p.window_start, unit="s", utc=True).dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize()
    last = pd.to_datetime(p.window_end - 1, unit="s", utc=True).dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize()
    sample = pd.to_datetime(p.sample_date)
    known = sample.notna() & first.notna() & last.notna()
    gaps = (sample - last).dt.days.where(known)
    row = dict(setting=label, people=len(p), known_window_and_stool_date=int(known.sum()),
        unknown_or_unmatched=int((~known).sum()), before=int((known & sample.lt(first)).sum()),
        within=int((known & sample.ge(first) & sample.le(last)).sum()), after=int((known & sample.gt(last)).sum()),
        gap_after_window_q25=gaps.quantile(.25), gap_after_window_median=gaps.median(), gap_after_window_q75=gaps.quantile(.75),
        sample_identity_verified=int(p.timing_status.eq("MATCHED_SAMPLE").sum()),
        single_visit_record=int(p.timing_status.eq("SINGLE_VISIT_RECORD").sum()),
        date_order_filter_applied=False)
    return row


def run_models(run, meals, people, quality, previous_diet, previous_gut, main_diet=.05, progress=print):
    data = diet.features(meals)  # Keep the original date origin in every setting.
    tagged = attach(data, quality)
    run.table(threshold_summary(quality, tagged, main_diet), "02_threshold_flow.csv")
    source_persons, source_members, source_flow, mismatch = gut.prepare(meals, people)
    old_gut = gut.analyze(source_persons, source_members, progress)
    gut_match = reproduce(old_gut["models"], previous_gut, GUT_KEYS, ["q_family", "q_meal12"])
    run.table(gut_match, "03_gut_reproduction.csv")

    diet_rows, coefficients, flows, membership, balance = [], [], [], [], []
    for score in DIETS:
        progress("REPRODUCE diet " + score)
        for branch in ["A", "B"]:
            sample, flow = diet.prepare(data, people, score, "primary_available", branch)
            rows, coefs = fit_diet_sample(sample, score, "source_primary", branch)
            diet_rows.extend(rows)
            coefficients.extend(coefs)
            flows.extend([dict(x, module="diet", setting="source_primary") for x in flow])
    source_diet = diet_fdr(pd.DataFrame(diet_rows))
    diet_match = reproduce(source_diet, previous_diet.loc[previous_diet.setting.eq("primary_available")],
                           DIET_KEYS, ["q_family14"])
    run.table(diet_match, "03_diet_reproduction.csv")
    progress("REPRODUCTION=PASS (42 diet + 48 gut models; identities and effects matched)")
    for score in DIETS:
        for branch in ["A", "B"]:
            base, _ = diet.prepare(data, people, score, "primary_available", branch)
            for label, cg, dt, strict in settings(main_diet):
                progress("DIET {} {} {}".format(label, score, branch))
                pool = tagged.loc[quality_mask(tagged, cg, dt, meal=True)]
                sample, flow = diet.prepare(pool, people, score, "strict_available" if strict else "primary_available", branch)
                rows, coefs = fit_diet_sample(sample, score, label, branch)
                diet_rows.extend(rows)
                coefficients.extend(coefs)
                flows.extend([dict(x, module="diet", setting=label) for x in flow])
                membership.append(sample[["participant_id", "meal_id"]].assign(module="diet", setting=label, diet=score, branch=branch))
                balance.extend(selection_balance(base, sample.participant_id, "diet", score, branch, label))
    diet_table = diet_fdr(pd.DataFrame(diet_rows))

    gut_rows, gut_coef, gut_samples, timing_rows = [], [], [], []
    background = ["age_years", "bmi", "sex", "cgm_device_type", "mean_daily_energy_kcal", "hPDI", "NOVA4"]
    background += [x["id"] for x in contract()["taxa"]]
    background = [x for x in background if x in source_persons and x not in data]
    gut_balance_base = data.loc[data.meal_id.isin(source_members.meal_id)].merge(
        source_persons[["participant_id"] + background], on="participant_id", how="left", validate="many_to_one")
    for old_label, new_label in [("primary_common", "source_primary"), ("strict_common", "source_strict")]:
        gut_rows.extend(old_gut["models"].loc[old_gut["models"].setting.eq(old_label)].assign(setting=new_label).to_dict("records"))
        gut_coef.extend(old_gut["coefficients"].loc[old_gut["coefficients"].setting.eq(old_label)].assign(setting=new_label).to_dict("records"))
        p = source_persons if old_label == "primary_common" else source_persons.loc[source_persons.strict_eligible.eq(True)]
        timing_rows.append(timing(p, quality, new_label, run.c["time"]["local_timezone"]))
    for label, cg, dt, strict in settings(main_diet):
        progress("GUT " + label)
        pool = tagged.loc[quality_mask(tagged, cg, dt, meal=True)]
        persons, members, selected = aggregate_gut(data, source_persons, source_members, pool, strict)
        membership.append(members.assign(module="gut", setting=label, diet="", branch=""))
        run.table(persons, "gut_person_inputs_" + label + ".csv", private=True)
        gut_samples.append(dict(setting=label, people=len(persons), meals=len(members),
            person_sha256=diet.sample_hash(persons.participant_id), meal_sha256=diet.sample_hash(members.meal_id)))
        balance.extend(selection_balance(gut_balance_base, persons.participant_id, "gut", None, "", label))
        timing_rows.append(timing(persons, quality, label, run.c["time"]["local_timezone"]))
        for taxon in contract()["taxa"]:
            for outcome in contract()["meal_outcomes"] + contract()["reference_outcomes"]:
                row, coefs = gut.estimate(persons, members, label, taxon["id"], outcome)
                gut_rows.append(row)
                gut_coef.extend(coefs)
    gut_table = gut_fdr(pd.DataFrame(gut_rows))
    run.table(pd.DataFrame(source_flow), "02_gut_original_flow.csv")
    run.table(pd.DataFrame(timing_rows), "04_stool_timing.csv")
    run.table(pd.DataFrame(gut_samples), "04_gut_samples.csv")
    run.table(pd.DataFrame(balance), "05_selection_balance.csv")
    run.table(pd.DataFrame(flows), "04_diet_sample_flow.csv")
    run.table(pd.concat(membership, ignore_index=True), "model_membership.csv", private=True)
    run.table(pd.DataFrame(coefficients, columns=diet.COEFFICIENT_COLUMNS), "diet_coefficients.csv")
    run.table(pd.DataFrame(gut_coef, columns=["setting", "taxon", "outcome", "term", "estimate", "se", "ci_low", "ci_high", "p"]), "gut_coefficients.csv")
    run.table(diet_table, "diet_models.csv")
    run.table(gut_table, "gut_models.csv")
    return diet_table, gut_table
