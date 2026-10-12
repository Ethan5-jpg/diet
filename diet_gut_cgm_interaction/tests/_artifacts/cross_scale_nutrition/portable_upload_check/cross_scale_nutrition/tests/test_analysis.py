import numpy as np
import pandas as pd
import pytest

from crossscale.analysis import (VARIANTS, anchors_for, design_for, run_models, sample_audit, select_samples)
from crossscale.common import bh
from crossscale.inputs import load_panel
from crossscale.estimators import fit_linear, paired_contrast


def test_common_samples_recalculate_min_days_and_total_range(panel, protocol):
    ids = panel.participant_id.unique()
    panel.loc[panel.participant_id.eq(ids[0]), "fiber_density"] = np.nan
    index = panel.index[panel.participant_id.eq(ids[1])]
    panel.loc[index[4:], "fiber_density"] = np.nan
    panel.loc[panel.participant_id.eq(ids[2]), "nova4_total_pct"] = 30
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    assert ids[0] in set(samples["F"].participant_id) and ids[0] not in set(samples["N"].participant_id)
    assert ids[1] not in set(samples["N"].participant_id)
    assert ids[2] not in set(samples["F"].participant_id)
    assert set(samples["N"].day_id).issubset(samples["F"].day_id)
    audit = sample_audit(samples)
    assert len(audit) == 26
    assert audit.loc[audit["sample"].eq("N"), "sample_sha256"].nunique() == 1
    assert audit.loc[audit["sample"].eq("F"), "sample_sha256"].nunique() == 1


def test_nutrition_design_uses_identical_days_and_adds_microbe_products(panel, protocol):
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    anchors = anchors_for(samples["F"])
    variants = {v["name"]: v for v in VARIANTS}
    for scale in ["W", "B"]:
        a, cols0, _ = design_for(samples["N"], variants["N0_joint"], scale, anchors)
        b, cols1, _ = design_for(samples["N"], variants["N1_joint"], scale, anchors)
        np.testing.assert_allclose(a[cols0], b[cols0])
        for t in ["T068", "T096"]:
            for nutrient in ["carb", "protein", "fiber"]:
                assert t + "_x_" + nutrient in cols1
        assert a.participant_id.tolist() == b.participant_id.tolist()
    b, _, _ = design_for(samples["N"], variants["N0_joint"], "B", anchors)
    expected = samples["N"].groupby("participant_id").nova4_total_pct.mean() / 10
    np.testing.assert_allclose(b.set_index("participant_id").x, expected)


def test_full_pipeline_keeps_fixed_registries_and_fdr(panel, protocol):
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    tables = run_models(samples, protocol, True)
    r, c, d = tables["interactions"], tables["comparisons"], tables["diagnostics"]
    assert len(r) == 44 and len(c) == 40 and len(d) == 26
    assert d.status.eq("ESTIMATED").all(), d.loc[d.status.ne("ESTIMATED"), ["variant", "scale", "reason"]].to_dict("records")
    assert c.status.eq("ESTIMATED").all()
    np.testing.assert_allclose(r.q_all44, bh(r.p))
    np.testing.assert_allclose(c.q_all40, bh(c.p))
    assert r.loc[r.variant.eq("F_joint") & r.outcome.eq("Mean"), "q_main_sd4"].isna().all()
    for _, rows in r.groupby("family"):
        np.testing.assert_allclose(rows.q_family, bh(rows.p))


def test_known_between_within_interactions_are_recovered_in_native_units(panel, protocol):
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    d = samples["F"].copy()
    x = d.nova4_total_pct / 10
    xb = x.groupby(d.participant_id).transform("mean")
    xw = x - xb
    rng = np.random.default_rng(17)
    ids = sorted(d.participant_id.unique())
    person_error = d.participant_id.map(dict(zip(ids, rng.normal(scale=.01, size=len(ids)))))
    d["daily_sd"] = (20 + .2 * xb + .4 * xw + .8 * xb * d.T068 - .4 * xb * d.T096 -
                     .3 * xw * d.T068 + .7 * xw * d.T096 + person_error + rng.normal(scale=.01, size=len(d)))
    anchors = anchors_for(d)
    fits = {}
    for scale, expected in [("B", [.8, -.4]), ("W", [-.3, .7])]:
        design, cols, focal = design_for(d, VARIANTS[0], scale, anchors)
        fit = fit_linear(design, "daily_sd", cols, focal, scale == "W", min_df=8)
        fits[scale] = fit
        np.testing.assert_allclose([fit["coefficients"]["xm_" + t]["beta"] for t in ["T068", "T096"]], expected, atol=.015)
    for t, expected in [("T068", -1.1), ("T096", 1.1)]:
        row = paired_contrast(fits["W"], fits["B"], {"xm_" + t: 1}, {"xm_" + t: 1})
        assert abs(row["beta"] - expected) < .02


def test_no_valid_nutrition_keeps_failed_rows_and_main_results(panel, protocol):
    panel["fiber_density"] = np.nan
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    tables = run_models(samples, protocol, True)
    r, c = tables["interactions"], tables["comparisons"]
    assert len(r) == 44 and len(c) == 40
    n = r.loc[r["sample"].eq("N")]
    assert len(n) == 16 and n.status.eq("NOT_ESTIMATED").all()
    assert n[["beta", "p", "q_family", "q_all44"]].isna().all().all()
    assert r.loc[r.variant.eq("F_joint"), "status"].eq("ESTIMATED").all()
    np.testing.assert_allclose(r.q_all44, bh(r.p), equal_nan=True)


@pytest.mark.parametrize("failure", ["duplicate", "boolean", "denominator", "cv", "id_change", "date"])
def test_saved_panel_tampering_rejected(panel, protocol, tmp_path, failure):
    if failure == "duplicate": panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    elif failure == "boolean": panel["alcohol_observed_ok"] = "yes"
    elif failure == "denominator": panel.loc[0, "nova4_total_pct"] += 1
    elif failure == "cv": panel.loc[0, "daily_cv"] += 1
    elif failure == "id_change": panel.loc[0, "T068"] += 1
    elif failure == "date": panel.loc[0, "date"] = "2025"
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_panel(path, protocol["reference_config"])


def test_string_ids_survive_csv_roundtrip(panel, protocol, tmp_path):
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)
    loaded = load_panel(path, protocol["reference_config"])
    assert {"001", "NA"}.issubset(loaded.participant_id)
    np.testing.assert_allclose(loaded.T096, panel.T096)


def test_empty_full_sample_retains_all_failed_hypotheses(panel, protocol):
    panel["T096"] = np.nan
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    tables = run_models(samples, protocol, True)
    assert len(tables["interactions"]) == 44
    assert len(tables["comparisons"]) == 40
    assert tables["interactions"].status.eq("NOT_ESTIMATED").all()
    assert tables["interactions"][["beta", "p", "q_family", "q_all44"]].isna().all().all()
    assert {"term", "beta", "p"}.issubset(tables["coefficients"].columns)
    assert {"point", "beta", "p"}.issubset(tables["slopes"].columns)
