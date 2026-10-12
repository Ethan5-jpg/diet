"""Verify the saved revised panel, then reproduce historical checkpoints."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (BASE_CATEGORY, BASE_NUMERIC, TAXA, OUTCOMES, boolean,
                     numeric, read_csv, require, sample_hash, sha256)
from .estimators import fit_linear

REFERENCE_FILES = ["analysis_config.json", "protocol.json", "private/daily_panel.csv",
                   "tables/10_interactions.csv", "tables/02_paired_sample_identity.csv"]


def open_reference(reference, protocol, synthetic=False):
    reference = Path(reference).expanduser().resolve()
    digest = sha256(reference / "manifest.json")
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    kind = "SYNTHETIC" if synthetic else "SERVER_INPUTS"
    require(manifest["data_kind"] == kind, "Reference real/synthetic mode mismatch")
    require(manifest["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"], "Reference analysis incomplete")
    require(manifest["version"] == protocol["reference_version"], "Wrong reference version")
    require(manifest["code_sha256"] == protocol["reference_code_sha256"], "Reference runtime fingerprint mismatch")
    require(manifest["parent_code_sha256"] == protocol["parent_code_sha256"], "Reference parent fingerprint mismatch")
    if not synthetic:
        require(manifest["run_id"] == protocol["reference_run_id"] and reference.name == protocol["reference_run_id"],
                "Use the exact completed real revised run")
        require(manifest["amed_sha256"] == protocol["amed_sha256"], "Reference alcohol source fingerprint mismatch")
    def verify():
        require(sha256(reference / "manifest.json") == digest, "Reference manifest changed")
        for name in REFERENCE_FILES:
            require(sha256(reference / name) == manifest["output_sha256"][name], "Saved reference changed: " + name)
    verify()
    config = json.loads((reference / "analysis_config.json").read_text())
    require(config == manifest["config"], "Reference config/manifest mismatch")
    saved_protocol = json.loads((reference / "protocol.json").read_text())
    require(saved_protocol == manifest["protocol"], "Reference protocol/manifest mismatch")
    expected = json.loads(json.dumps(protocol["reference_config"]))
    if synthetic:
        expected["analysis"]["minimum_people"] = config["analysis"]["minimum_people"]
        expected["analysis"]["minimum_tail_people"] = config["analysis"]["minimum_tail_people"]
    require(config == expected, "Reference QC/analysis configuration differs from frozen settings")
    panel = load_panel(reference / "private/daily_panel.csv", config)
    results = read_csv(reference / "tables/10_interactions.csv")
    require(len(results) == 132 and not results.duplicated(["spec", "exposure", "taxon", "outcome"]).any(),
            "Reference 132-row registry invalid")
    pairs = read_csv(reference / "tables/02_paired_sample_identity.csv")
    require(len(pairs) == 24 and boolean(pairs.identical, "reference sample identity").all(),
            "Reference paired sample audit failed")
    verify()
    return panel, config, results, manifest, digest, verify


def load_panel(path, config):
    d = read_csv(path)
    text = ["participant_id", "connection_id", "day_id", "date", "cohort", "research_stage"] + BASE_CATEGORY
    nums = (BASE_NUMERIC + TAXA + list(OUTCOMES.values()) + ["daily_cv", "nova4_total_pct",
            "nova4_classified_pct", "nova_coverage", "energy_kcal", "alcohol_amed_g",
            "carb_density", "protein_density", "fiber_density", "weekday", "observation_day"])
    flags = ["primary_eligible", "strict_eligible", "diet_pass", "cgm_pass", "cgm_strict", "alcohol_observed_ok"]
    require(set(text + nums + flags).issubset(d), "Saved panel is missing required columns")
    for c in nums:
        d[c] = numeric(d[c], c)
    for c in flags:
        d[c] = boolean(d[c], c)
    require(d[text[:6]].ne("").all().all(), "Missing saved panel identity")
    require(not d.duplicated(["participant_id", "date"]).any() and d.day_id.is_unique, "Duplicate person-day records")
    require(d.cohort.eq(config["cohort"]).all() and d.research_stage.eq(config["research_stage"]).all(), "Wrong cohort/visit")
    dates = pd.to_datetime(d.date, format="%Y-%m-%d", errors="coerce")
    require(dates.notna().all() and dates.dt.strftime("%Y-%m-%d").eq(d.date).all(), "Invalid exact local date")
    require(d.weekday.eq(dates.dt.weekday).all(), "Saved weekday/date mismatch")
    constant = TAXA + BASE_NUMERIC + BASE_CATEGORY + ["connection_id", "primary_eligible", "strict_eligible"]
    require(d.groupby("participant_id")[constant].nunique(dropna=False).le(1).all().all(),
            "Time-invariant person fields changed within a participant")
    eligible = d.primary_eligible
    require(d.loc[eligible, BASE_NUMERIC].notna().all().all(), "Eligible records lack baseline numeric covariates")
    for c in BASE_CATEGORY:
        require(not d.loc[eligible, c].str.lower().isin(["", "nan", "none", "null", "<na>"]).any(),
                "Eligible records lack baseline categorical covariates")
    # Both denominator definitions must describe the same measured numerator.
    valid = d[["nova4_total_pct", "nova4_classified_pct", "nova_coverage"]].notna().all(axis=1)
    require(d.loc[valid, "nova4_total_pct"].between(0, 100 + 1e-8).all() and
            d.loc[valid, "nova4_classified_pct"].between(0, 100 + 1e-8).all() and
            d.loc[valid, "nova_coverage"].between(0, 1 + 1e-8).all(), "Invalid NOVA bounds")
    require(np.allclose(d.loc[valid, "nova4_total_pct"],
                        d.loc[valid, "nova4_classified_pct"] * d.loc[valid, "nova_coverage"], rtol=1e-9, atol=1e-8),
            "NOVA denominator identity failed")
    valid = d[["daily_mean", "daily_sd", "daily_cv"]].notna().all(axis=1)
    require(d.loc[valid, "daily_mean"].gt(0).all() and d.loc[valid, "daily_sd"].ge(0).all(), "Invalid CGM summaries")
    require(np.allclose(d.loc[valid, "daily_cv"], 100 * d.loc[valid, "daily_sd"] / d.loc[valid, "daily_mean"], rtol=1e-9, atol=1e-8),
            "Saved CV/SD/Mean identity failed")
    require(d.loc[d.alcohol_observed_ok, "alcohol_amed_g"].ge(0).all(), "Observed-source alcohol values invalid")
    return d


def historical_sample(panel, config, taxon, outcome, nutrient):
    d = panel.loc[panel.primary_eligible & panel.diet_pass & panel.cgm_pass & panel.alcohol_observed_ok &
                  panel.nova_coverage.ge(config["diet"]["minimum_nova_coverage"])].copy()
    cols = ["nova4_classified_pct", taxon, OUTCOMES[outcome], "energy_kcal", "alcohol_amed_g", "weekday", "observation_day"]
    if nutrient:
        cols += ["carb_density", "protein_density", "fiber_density"]
    d = d.dropna(subset=cols)
    x = d.groupby("participant_id").nova4_classified_pct
    d = d.loc[x.transform("size").ge(config["analysis"]["minimum_days"]) &
              (x.transform("max") - x.transform("min")).ge(config["analysis"]["minimum_nova_range_pp"])].copy()
    return d


def reproduce_reference(panel, config, results, protocol, synthetic=False):
    """Eight old primary/nutrient NOVA4 estimates, before any new comparison."""
    rows = []
    for taxon in TAXA:
        primary = historical_sample(panel, config, taxon, "SD", False)
        anchor = primary.drop_duplicates("participant_id")[taxon].mean()
        for outcome in OUTCOMES:
            for spec in ["primary", "nutrient_context"]:
                old = results.loc[results.spec.eq(spec) & results.exposure.eq("NOVA4") & results.taxon.eq(taxon) & results.outcome.eq(outcome)]
                require(len(old) == 1 and old.iloc[0].status == "ESTIMATED", "Missing completed reference checkpoint")
                old = old.iloc[0]
                d = historical_sample(panel, config, taxon, outcome, spec == "nutrient_context")
                require((len(d), d.participant_id.nunique()) == (int(old.n_days), int(old.n_people)), "Historical sample counts changed")
                require(sample_hash(d) == old.sample_sha256, "Historical person-day sample identity changed")
                if not synthetic:
                    expected = protocol["expected_historical_counts"].get(spec + "_" + taxon)
                    if expected is not None:
                        require([d.participant_id.nunique(), len(d)] == expected, "Historical screenshot counts changed")
                x = d.nova4_classified_pct / 10
                m = d[taxon] - anchor
                d["x_within"] = x - x.groupby(d.participant_id).transform("mean")
                d["interaction"] = d.x_within * m
                d["energy_1000"] = d.energy_kcal / 1000
                d["alcohol_10g"] = d.alcohol_amed_g / 10
                controls = ["energy_1000", "alcohol_10g", "observation_day"]
                for k in range(1, 7):
                    c = "weekday_" + str(k)
                    d[c] = d.weekday.eq(k).astype(float)
                    controls.append(c)
                if spec == "nutrient_context":
                    d["carb_density_50"] = d.carb_density / 50
                    d["protein_density_25"] = d.protein_density / 25
                    d["other_diet"] = d.fiber_density / 5
                    controls += ["carb_density_50", "protein_density_25", "other_diet"]
                columns = ["x_within", "interaction"] + controls
                for c in controls:
                    d["microbe_x_" + c] = m * d[c]
                    columns.append("microbe_x_" + c)
                fit = fit_linear(d, OUTCOMES[outcome], columns, ["x_within", "interaction"], True, False,
                                 config["analysis"]["minimum_residual_df"], config["analysis"]["maximum_condition_number"])
                coef = fit["coefficients"]["interaction"]
                for k in ["beta", "se", "low", "high", "p"]:
                    require(np.isclose(coef[k], float(old[k]), rtol=2e-6, atol=2e-9), "Historical numeric reproduction failed: " + k)
                rows.append(dict(spec=spec, taxon=taxon, outcome=outcome, n_people=fit["n_people"], n_days=len(d),
                                 sample_sha256=sample_hash(d), passed=True, **coef))
    return pd.DataFrame(rows)
