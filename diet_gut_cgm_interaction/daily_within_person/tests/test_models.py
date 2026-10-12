import numpy as np
import pandas as pd
import pytest

from daily.models import bh, choose_sample, linear_contrast, within_fit


def regression_data():
    rng = np.random.RandomState(44)
    records = []
    for i in range(45):
        m, intercept = rng.normal(size=2)
        for j in range(5 + i % 8):
            x, z = rng.normal(size=2)
            y = 10 * intercept + 1.2 * x - .4 * x * m + .5 * z + rng.normal(scale=.4)
            records.append(dict(participant_id=str(i), day_id="{}|{}".format(i, j), x_within=x, interaction=x * m, z=z, y=y))
    return pd.DataFrame(records)


@pytest.mark.parametrize("weighted", [False, True])
def test_fe_cluster_covariance_matches_explicit_dummy_regression(weighted):
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    d = regression_data()
    cols = ["x_within", "interaction", "z"]
    fit = within_fit(d, "y", cols, equal_person=weighted, min_df=30)
    dummy = pd.get_dummies(d.participant_id, dtype=float)
    x = pd.concat([dummy, d[cols]], axis=1)
    weights = 1.0 / d.groupby("participant_id").participant_id.transform("size") if weighted else np.ones(len(d))
    explicit = sm.WLS(d.y, x, weights=weights).fit()
    cov = cov_cluster(explicit, d.participant_id, use_correction=True)[-3:, -3:]
    np.testing.assert_allclose([fit["coefficients"][s]["beta"] for s in cols], explicit.params.iloc[-3:], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit["covariance"], cov, rtol=1e-9, atol=1e-10)
    assert fit["residual_df"] == explicit.df_resid
    assert fit["coefficients"]["interaction"]["beta"] == pytest.approx(-.4, abs=.08)
    contrast = linear_contrast(fit, {"x_within": 1, "interaction": 2})
    np.testing.assert_allclose(contrast["se"] ** 2, np.array([1, 2, 0]).dot(cov).dot(np.array([1, 2, 0])))


def test_person_level_outcome_shifts_are_absorbed():
    d = regression_data()
    cols = ["x_within", "interaction", "z"]
    a = within_fit(d, "y", cols, min_df=30)
    d.y += d.participant_id.astype(float) * 1000
    b = within_fit(d, "y", cols, min_df=30)
    np.testing.assert_allclose(a["covariance"], b["covariance"], rtol=1e-9, atol=1e-9)
    assert a["coefficients"]["interaction"]["beta"] == pytest.approx(b["coefficients"]["interaction"]["beta"], abs=1e-9)


def test_nonvarying_and_collinear_focal_terms_are_rejected():
    d = regression_data()
    d["interaction"] = d.participant_id.astype(float)
    with pytest.raises(ValueError, match="variation"):
        within_fit(d, "y", ["x_within", "interaction", "z"], min_df=30)
    d["interaction"] = d.x_within
    with pytest.raises(ValueError, match="Collinear"):
        within_fit(d, "y", ["x_within", "interaction", "z"], min_df=30)


def test_bh_keeps_unestimated_tests_in_family():
    q = bh([.01, np.nan, .04, .8])
    np.testing.assert_allclose(q[[0, 2, 3]], [.04, .08, 1])
    assert np.isnan(q[1])


def test_filter_applies_days_after_model_complete_cases(config):
    p = pd.DataFrame(dict(participant_id=["a"] * 6 + ["b"] * 6,
        primary_eligible=True, strict_eligible=True, diet_pass=True, cgm_pass=True, cgm_strict=True,
        nova_coverage=1., nova4_classified_pct=[10,20,30,40,50,60] * 2, T068=1., daily_sd=10.,
        energy_kcal=2000., alcohol_g=0., weekday=[0,1,2,3,4,5] * 2, observation_day=list(range(6)) * 2))
    p.loc[[0, 1], "alcohol_g"] = np.nan
    d, flow = choose_sample(p, config, "NOVA4", "T068", "SD", "primary")
    assert set(d.participant_id) == {"b"}
    assert flow[-2]["n_people"] == 2
    assert flow[-1]["n_people"] == 1
    quality = next(r for r in flow if r["stage"] == "eligible_quality")
    assert quality["missing_alcohol_days"] == 2
    assert quality["missing_exposure_days"] == 0
    assert quality["missing_any_model_field_days"] == 2
