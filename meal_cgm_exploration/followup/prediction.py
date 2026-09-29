"""Prospective-date holdout for already seen people and logged meal templates."""
import numpy as np
import pandas as pd

from hppmeal.common import KEYS, require
from .repeats import GROUP

MODELS = ["template_history", "plus_context", "plus_premeal", "premeal_persistence"]
FEATURES = ["hour_sin", "hour_cos", "gap_hours", "baseline_glucose", "pre_slope_per_min"]


def features(d):
    x = d.copy()
    x["hour_sin"] = np.sin(2 * np.pi * x.local_hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.local_hour / 24)
    x["gap_hours"] = np.minimum(x.previous_gap_minutes / 60, 12)
    return x


def temporal_split(d):
    """Date boundaries depend only on eligible dates, never on outcome values."""
    eligible = d.loc[d.signature.fillna("").ne("")].copy()
    require(pd.to_datetime(eligible.local_date, format="%Y-%m-%d", errors="coerce").notna().all(),
            "Invalid local dates in eligible meals")
    eligible["split"] = "insufficient_dates"
    for _, g in eligible.groupby(KEYS, sort=True):
        days = sorted(g.local_date.unique())
        if len(days) < 3:
            continue
        cut = min(len(days) - 1, max(2, int(np.floor(0.7 * len(days)))))
        eligible.loc[g.index[g.local_date.isin(days[:cut])], "split"] = "train"
        eligible.loc[g.index[g.local_date.isin(days[cut:])], "split"] = "test"
    return eligible


def prepare_prediction(d):
    rows = []

    def record(label, frame):
        rows.append(dict(stage=label, meals=len(frame), people=frame.participant_id.nunique(),
                         person_dates=len(frame[KEYS + ["local_date"]].drop_duplicates())))

    record("eligible_meals", d)
    split = temporal_split(d).reset_index(drop=True)
    record("valid_signature", split)
    record("insufficient_dates", split.loc[split.split.eq("insufficient_dates")])
    split = features(split)
    # The same feature-complete rows are used by all models. Date split happened first.
    complete = np.isfinite(split[FEATURES + ["mean_glucose"]].to_numpy(float)).all(axis=1)
    split["feature_outcome_complete"] = complete
    train = split.loc[split.split.eq("train") & complete].copy()
    test = split.loc[split.split.eq("test") & complete].copy()
    record("train_complete_before_template_support", train)
    record("test_complete_before_template_support", test)
    support = train.groupby(GROUP).local_date.nunique().rename("training_dates").reset_index()
    support = support.loc[support.training_dates.ge(2)]
    train = train.merge(support, on=GROUP, how="inner", validate="many_to_one")
    test = test.merge(support, on=GROUP, how="inner", validate="many_to_one")
    record("train_supported_templates", train)
    record("test_seen_supported_templates", test)
    assignment = split[KEYS + ["meal_id", "signature", "local_date", "split", "feature_outcome_complete"]].copy()
    supported = set(train.meal_id) | set(test.meal_id)
    assignment["used_after_template_support"] = assignment.meal_id.isin(supported)
    return train, test, pd.DataFrame(rows), assignment


def fit_predict(train, test, feature_cols, penalty=10.0):
    """Unpenalized template intercepts; train-only within-template ridge slopes."""
    require(len(train) > 0 and len(test) > 0, "Empty train/test set")
    require(np.isfinite(penalty) and penalty > 0, "Ridge penalty must be positive")
    means = train.groupby(GROUP)[["mean_glucose"] + feature_cols].mean()
    train_index = pd.MultiIndex.from_frame(train[GROUP])
    test_index = pd.MultiIndex.from_frame(test[GROUP])
    require(test_index.isin(means.index).all(), "Unseen test template; cannot extrapolate")
    tr_mean = means.reindex(train_index).reset_index(drop=True)
    te_mean = means.reindex(test_index).reset_index(drop=True)
    predicted = te_mean.mean_glucose.to_numpy().copy()
    coefficients = []
    if feature_cols:
        X = train[feature_cols].to_numpy(float) - tr_mean[feature_cols].to_numpy(float)
        T = test[feature_cols].to_numpy(float) - te_mean[feature_cols].to_numpy(float)
        y = train.mean_glucose.to_numpy(float) - tr_mean.mean_glucose.to_numpy(float)
        scale = np.sqrt(np.mean(X ** 2, axis=0))
        active = scale > 1e-10
        beta = np.zeros(len(feature_cols))
        if active.any():
            Z = X[:, active] / scale[active]
            bz = np.linalg.solve(Z.T.dot(Z) + penalty * np.eye(active.sum()), Z.T.dot(y))
            beta[active] = bz / scale[active]
        predicted += T.dot(beta)
        for j, col in enumerate(feature_cols):
            coefficients.append(dict(feature=col, coefficient=beta[j],
                                     train_within_template_scale=scale[j],
                                     status="FITTED" if active[j] else "CONSTANT_IN_TRAIN"))
    require(np.isfinite(predicted).all(), "Nonfinite predictions")
    return predicted, pd.DataFrame(coefficients, columns=["feature", "coefficient", "train_within_template_scale", "status"])


def evaluate_predictions(predictions, draws=1000, seed=20260925):
    d = predictions.copy()
    rows, gains = [], []
    errors = {}
    for name in MODELS:
        e = d[name] - d.mean_glucose
        errors[name] = e.abs()
        person = pd.DataFrame({"id": d.participant_id, "abs": e.abs(), "sq": e ** 2}).groupby("id").mean()
        rows.append(dict(model=name, people=len(person), test_meals=len(d), unit="mg/dL",
                         person_equal_mae=person["abs"].mean(), person_equal_rmse=np.sqrt(person.sq.mean()),
                         meal_weighted_mae=e.abs().mean(), meal_weighted_rmse=np.sqrt((e ** 2).mean())))
    for previous, newer in [("template_history", "plus_context"), ("plus_context", "plus_premeal"),
                            ("template_history", "plus_premeal"), ("premeal_persistence", "plus_premeal")]:
        # One difference per participant; every person gets equal weight.
        g = pd.DataFrame({"id": d.participant_id, "gain": errors[previous] - errors[newer]}).groupby("id").gain.mean()
        x, n = g.to_numpy(float), len(g)
        rng = np.random.RandomState(seed)
        boot = np.array([x[rng.randint(0, n, size=n)].mean() for _ in range(draws)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        gains.append(dict(previous=previous, newer=newer, people=n, test_meals=len(d),
                          mae_improvement=x.mean(), ci_low=lo, ci_high=hi, unit="mg/dL",
                          direction="positive means lower error for newer model",
                          ci_method="participant_cluster_percentile",
                          ci_scope="conditional on fixed trained models; no refitting", bootstrap_draws=draws))
    return pd.DataFrame(rows), pd.DataFrame(gains)


def predict_cohort(d, cohort, draws, seed, minimum_people=50):
    train, test, flow, assignment = prepare_prediction(d)
    flow.insert(0, "cohort", cohort)
    info = dict(cohort=cohort, status="NOT_ESTIMATED", train_people=train.participant_id.nunique(),
                test_people=test.participant_id.nunique(), train_meals=len(train), test_meals=len(test),
                target="two-hour mean absolute glucose, mg/dL", penalty=10.0,
                validation="later dates of known participants and known meal templates")
    if min(info["train_people"], info["test_people"]) < minimum_people:
        info["reason"] = "Fewer than {} supported train/test participants; no threshold relaxation".format(minimum_people)
        return info, flow, assignment, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    require(set(test.participant_id).issubset(set(train.participant_id)), "Unknown test participant")
    predictions = test[KEYS + ["meal_id", "signature", "local_date", "mean_glucose"]].copy()
    coefficients = []
    for name, cols in [(MODELS[0], []), (MODELS[1], FEATURES[:3]), (MODELS[2], FEATURES)]:
        values, coef = fit_predict(train, test, cols)
        predictions[name] = values
        coef.insert(0, "model", name)
        if not coef.empty:
            coefficients.append(coef)
    predictions["premeal_persistence"] = test.baseline_glucose.to_numpy(float)
    metrics, gains = evaluate_predictions(predictions, draws, seed)
    coef = pd.concat(coefficients, ignore_index=True)
    for table in [metrics, gains, coef]:
        table.insert(0, "cohort", cohort)
    info.update(status="ESTIMATED", reason="")
    return info, flow, assignment, predictions, metrics, gains, coef
