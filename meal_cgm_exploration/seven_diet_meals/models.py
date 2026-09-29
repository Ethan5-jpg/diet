"""Two identifiable questions: personal means and within-person carb interactions."""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from hppmeal.common import require
from hppmeal.models import bh_planned
from .inputs import DIETS, BASE, CATEGORIES

OUTCOMES = ["mean_increment", "peak_rise_0_120"]
CONTEXT = ["baseline_glucose", "pre_slope_per_min", "hour_sin", "hour_cos", "gap_hours", "observation_day", "duration_30"]
NUTRIENTS = ["carb_10", "carb_10_squared", "energy_100"]
SETTINGS = ["primary_available", "strict_available", "primary_common7"]
COEFFICIENT_COLUMNS = ["setting", "diet", "branch", "outcome", "model", "term", "estimate", "se", "ci_low", "ci_high", "p"]
DESCRIPTIVE_COLUMNS = ["setting", "diet", "field", "people", "mean", "sd", "p10", "median", "p90"]


def extras(diet):
    if diet == "hPDI":
        return ["alcohol_intake_g_day"]
    return ["mean_daily_energy_kcal"] if diet in ["EAT13", "NOVA4", "Carbohydrate_pct"] else []


def features(meals):
    d = meals.copy()
    d["carb_10"] = d.carbohydrate_g / 10
    d["carb_10_squared"] = d.carb_10 ** 2
    d["energy_100"] = d.calories_kcal / 100
    d["hour_sin"] = np.sin(2 * np.pi * d.local_hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.local_hour / 24)
    d["gap_hours"] = np.minimum(d.previous_gap_minutes / 60, 12)
    d["observation_day"] = (d.start - d.groupby("participant_id").start.transform("min")) / 86400
    d["duration_30"] = d.duration_minutes / 30
    return d


def sample_hash(values):
    return hashlib.sha256("\n".join(sorted(values.astype(str))).encode()).hexdigest()


def prepare(meals, people, diet, setting, branch):
    person = people.reset_index()
    columns = [diet] + BASE + extras(diet)
    if setting == "primary_common7":
        columns = DIETS + BASE + ["alcohol_intake_g_day", "mean_daily_energy_kcal"]
    flag = "strict_eligible" if setting == "strict_available" else "primary_eligible"
    valid = person[flag] & person[columns].notna().all(axis=1)
    selected = person.loc[valid]
    flow = [dict(stage="score_and_covariates_complete", people=len(selected), meals=np.nan)]
    d = meals.merge(selected, on="participant_id", how="inner", validate="many_to_one")
    flow.append(dict(stage="plus_valid_meals", people=d.participant_id.nunique(), meals=len(d)))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=OUTCOMES + CONTEXT + NUTRIENTS)
    flow.append(dict(stage="plus_common_meal_fields", people=d.participant_id.nunique(), meals=len(d)))
    g = d.groupby("participant_id")
    enough = g.meal_id.transform("size").ge(5 if branch == "A" else 10) & g.local_date.transform("nunique").ge(3)
    if branch == "B":
        enough &= (g.carbohydrate_g.transform("max") - g.carbohydrate_g.transform("min")).ge(30)
    d = d.loc[enough].copy().reset_index(drop=True)
    flow.append(dict(stage="plus_repeated_support", people=d.participant_id.nunique(), meals=len(d)))
    for item in flow:
        item.update(setting=setting, diet=diet, branch=branch)
    return d, flow


def encode(frame, columns):
    blocks = []
    for col in columns:
        if col in CATEGORIES:
            levels = sorted(frame[col].astype(str).unique())
            for level in levels[1:]:
                blocks.append(pd.Series(frame[col].astype(str).eq(level).astype(float).to_numpy(),
                                        name=col + "=" + level))
        else:
            blocks.append(pd.Series(frame[col].to_numpy(float), name=col))
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=range(len(frame)))


def linear_fit(y, X, focus, ids=None, weights=None):
    """HC3 for independent people; weighted absorbed FE + cluster CR1 for meals."""
    X = X.astype(float).reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    require(len(X) == len(y) and len(y) > 0, "Empty/misaligned regression sample")
    require(np.isfinite(y).all() and np.isfinite(X.to_numpy()).all(), "Nonfinite design")
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    require(len(w) == n and np.isfinite(w).all() and (w > 0).all(), "Invalid weights")
    absorbed = 0
    if ids is not None:
        codes, groups = pd.factorize(np.asarray(ids))
        absorbed = len(groups)
        require(absorbed >= 50, "Fewer than 50 participant clusters")
        # All meals for a participant have equal weight, and person total weight is 1.
        by = pd.Series(w).groupby(codes)
        require(np.allclose(by.min(), by.max()) and np.allclose(by.sum(), 1), "Expected equal total weight per participant")
        X = X - X.groupby(codes).transform("mean")
        y = y - pd.Series(y).groupby(codes).transform("mean").to_numpy()
    else:
        X.insert(0, "intercept", 1.0)
    constant = X.std(ddof=0).le(1e-10)
    if ids is None:
        constant.loc["intercept"] = False
    require(focus in X and not constant[focus], "Focal term constant/unidentifiable")
    dropped = list(X.columns[constant])
    X = X.loc[:, ~constant]
    names = list(X)
    scale = np.sqrt(np.average(X.to_numpy() ** 2, axis=0, weights=w))
    Z = X.to_numpy() / scale * np.sqrt(w[:, None])
    target = y * np.sqrt(w)
    u, singular, vt = np.linalg.svd(Z, full_matrices=False)
    k = len(names)
    condition = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
    require(n - absorbed - k >= 30, "Fewer than 30 residual degrees of freedom")
    require(np.linalg.matrix_rank(Z) == k and condition < 1e8, "Rank-deficient or unstable design")
    beta_z = vt.T.dot(u.T.dot(target) / singular)
    resid = target - Z.dot(beta_z)
    bread = (vt.T / singular ** 2).dot(vt)
    if ids is None:
        leverage = np.sum(u ** 2, axis=1)
        require((leverage < 1 - 1e-8).all(), "Extreme HC3 leverage")
        scores = Z * (resid / (1 - leverage))[:, None]
        covariance = bread.dot(scores.T.dot(scores)).dot(bread)
        df = n - k
        method = "HC3; one row per participant"
    else:
        scores = np.zeros((absorbed, k))
        np.add.at(scores, codes, Z * resid[:, None])
        correction = absorbed / (absorbed - 1) * (n - 1) / (n - absorbed - k)
        covariance = correction * bread.dot(scores.T.dot(scores)).dot(bread)
        df = absorbed - 1
        method = "participant-cluster CR1; person total weight 1; absorbed FE counted"
    beta = beta_z / scale
    se = np.sqrt(np.maximum(np.diag(covariance), 0)) / scale
    critical = stats.t.ppf(0.975, df)
    rows = []
    for name, b, s in zip(names, beta, se):
        if name == focus:
            require(np.isfinite(s) and s > 0, "Degenerate focal standard error")
        p = float(2 * stats.t.sf(abs(b / s), df)) if s > 0 else np.nan
        rows.append(dict(term=name, estimate=b, se=s, ci_low=b - critical * s, ci_high=b + critical * s, p=p))
    return rows, dict(design_condition=condition, design_columns=k, absorbed_effects=absorbed,
                      residual_df=n - absorbed - k, inference_df=df, se_method=method,
                      dropped_constants=";".join(dropped))


def aggregate_people(d):
    means = d.groupby("participant_id")[OUTCOMES + CONTEXT + NUTRIENTS + ["mean_glucose"]].mean()
    static = d.drop_duplicates("participant_id").set_index("participant_id")[DIETS + BASE + ["alcohol_intake_g_day", "mean_daily_energy_kcal"]]
    return means.join(static).reset_index()


def attempt(d, diet, setting, branch, outcome, model):
    row = dict(setting=setting, diet=diet, branch=branch, outcome=outcome, model=model, status="NOT_ESTIMATED",
               estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, reason="",
               people=d.participant_id.nunique(), meals=len(d), person_sha256=sample_hash(d.participant_id.drop_duplicates()),
               meal_sha256=sample_hash(d.meal_id), unit="mg/dL per original diet Z" if branch == "A"
               else "mg/dL per 10g carbs per original diet Z")
    coefficients = []
    try:
        require(row["people"] >= 50, "Fewer than 50 eligible participants")
        require(d.drop_duplicates("participant_id")[diet].nunique() >= 3, "Fewer than three distinct dietary values")
        if branch == "A":
            p = aggregate_people(d)
            cols = [diet] + BASE + extras(diet) + CONTEXT + (NUTRIENTS if model == "meal_adjusted" else [])
            X = encode(p, cols)
            rows, diagnostic = linear_fit(p[outcome], X, diet)
            focus = diet
        else:
            X = d[CONTEXT + NUTRIENTS].copy().reset_index(drop=True)
            focus = "diet_x_carb"
            X[focus] = d[diet].to_numpy() * d.carb_10.to_numpy()
            heterogeneity = encode(d, ["age_years", "bmi", "sex"] + extras(diet))
            for col in heterogeneity:
                X[col + "_x_carb"] = heterogeneity[col].to_numpy() * d.carb_10.to_numpy()
            weights = 1.0 / d.groupby("participant_id").meal_id.transform("size").to_numpy()
            rows, diagnostic = linear_fit(d[outcome], X, focus, d.participant_id, weights)
        selected = next(x for x in rows if x["term"] == focus)
        row.update(selected)
        row.update(diagnostic)
        row["status"] = "ESTIMATED"
        coefficients = [dict(setting=setting, diet=diet, branch=branch, outcome=outcome, model=model, **x) for x in rows]
    except (ValueError, np.linalg.LinAlgError) as e:
        row["reason"] = str(e)
    return row, coefficients


def run_models(meals, people, progress=print):
    d = features(meals)
    rows, flows, coef, membership, descriptives = [], [], [], [], []
    for setting in SETTINGS:
        for diet in DIETS:
            progress("MODELS {} {}".format(setting, diet))
            for branch in ["A", "B"]:
                sample, flow = prepare(d, people, diet, setting, branch)
                flows.extend(flow)
                membership.append(sample[["participant_id", "meal_id"]].assign(setting=setting, diet=diet, branch=branch))
                if branch == "A" and len(sample):
                    person = aggregate_people(sample)
                    for field in OUTCOMES + ["mean_glucose", diet, "carb_10", "energy_100"]:
                        values = person[field]
                        descriptives.append(dict(setting=setting, diet=diet, field=field, people=len(person),
                                                 mean=values.mean(), sd=values.std(), p10=values.quantile(.1),
                                                 median=values.median(), p90=values.quantile(.9)))
                for outcome in OUTCOMES:
                    for model in (["background", "meal_adjusted"] if branch == "A" else ["carb_interaction"]):
                        row, coefficients = attempt(sample, diet, setting, branch, outcome, model)
                        rows.append(row)
                        coef.extend(coefficients)
    table = pd.DataFrame(rows)
    table["family"] = np.where(table.model.eq("background"), "background_14",
                               np.where(table.outcome.eq("mean_increment"), "primary_14", "peak_secondary_14"))
    for _, indices in table.groupby(["setting", "family"]).groups.items():
        require(len(indices) == 14, "Planned FDR family must contain 14 rows")
        table.loc[indices, "q_family14"] = bh_planned(table.loc[indices, "p"])
    table["planned_family_size"] = 14
    # Same people and meals, not merely the same N, for seven-score comparisons.
    common = table.loc[table.setting.eq("primary_common7")]
    for _, group in common.groupby("branch"):
        require(group.meal_sha256.nunique() == 1 and group.person_sha256.nunique() == 1, "Common7 membership differs across diets")
    # Empty outputs must remain readable CSVs when no model/sample is supported.
    return (table, pd.DataFrame(flows), pd.DataFrame(coef, columns=COEFFICIENT_COLUMNS),
            pd.concat(membership, ignore_index=True), pd.DataFrame(descriptives, columns=DESCRIPTIVE_COLUMNS))
