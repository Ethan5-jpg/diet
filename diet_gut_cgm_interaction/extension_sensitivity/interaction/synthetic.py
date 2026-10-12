"""Reproducible synthetic fixture. Never represents HPP observations."""
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import NEW_COLUMNS, resolve_paths


def create_fixture(root, config, n=900, seed=20261010):
    rng = np.random.RandomState(seed)
    ids = pd.Index(["SYNTH_{:06d}".format(i) for i in range(n)], name="participant_id")
    def z(x):
        return (x - np.mean(x)) / np.std(x, ddof=1)
    def save(label, frame):
        path = paths[label]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=True)
    paths = resolve_paths(config, root)
    scope = pd.DataFrame({"cohort": config["cohort"], "research_stage": config["research_stage"]}, index=ids)
    diet = pd.DataFrame({e: z(rng.normal(size=n)) for e in config["focus_exposures"] + config["extension_exposures"]}, index=ids)
    tax = pd.DataFrame({t["id"]: z(rng.normal(size=n)) for t in config["taxa"]}, index=ids)
    old = scope.copy()
    for e in ["AHEI", "AMED", "rEDIH"]:
        old[e + "_z"] = diet[e]
    old["hPDI_z"] = -diet.hPDI  # Wrong historical value: tests must use corrected source.
    save("current4", old)
    hp = scope.copy()
    hp["hpdi_score_energy_adjusted_z"] = diet.hPDI
    save("corrected_hpdi", hp)
    new, candidate = scope.copy(), scope.copy()
    for e, col in NEW_COLUMNS.items():
        new[e + "_z"] = diet[e]
        candidate[col] = diet[e]
    new["mean_daily_energy_kcal"] = 2200 + 180 * rng.normal(size=n)
    candidate["mean_daily_energy_kcal"] = new.mean_daily_energy_kcal
    save("new3", new)
    save("candidate_master", candidate)
    micro = scope.copy()
    for t in config["taxa"]:
        micro["k__Bacteria|g__Test|s__" + t["species"]] = tax[t["id"]]
    save("microbiome", micro)
    cov = scope.copy()
    cov["age_years"] = 52 + 8 * rng.normal(size=n)
    cov["sex"] = rng.choice(["Female", "Male"], n)
    cov["education_level"] = rng.choice(["1", "2", "3"], n)
    cov["smoking_status"] = rng.choice(["never", "former", "current"], n)
    cov["sleep_duration_hours_day"] = 7 + 0.7 * rng.normal(size=n)
    cov["physical_activity_met_h_week"] = rng.gamma(3, 4, n)
    cov["vitamin_use"] = rng.binomial(1, .4, n)
    cov["hormone_use"] = rng.binomial(1, .15, n)
    cov["bmi"] = 26 + 3 * rng.normal(size=n)
    cov["cgm_device_type"] = "Libre"
    cov["alcohol_intake_g_day"] = rng.gamma(1, 3, n)
    cov["exclude_diabetes_or_a10"] = 0.0
    cov.loc[ids[3:30], "exclude_diabetes_or_a10"] = np.nan
    cov.loc[ids[30:45], "exclude_diabetes_or_a10"] = 1.0
    save("covariates", cov.drop(ids[:3]))
    protocol = scope.copy()
    protocol["antibiotic_use_protocol_window"] = rng.binomial(1, .1, n).astype(float)
    protocol["ppi_use_protocol_window"] = rng.binomial(1, .15, n).astype(float)
    protocol.loc[ids[50:65], "antibiotic_use_protocol_window"] = np.nan
    save("protocol", protocol)
    cgm = scope.copy()
    cgm["connection_id"] = np.arange(n) + 1
    cgm["cgm_device_type"] = "Libre"
    values = {
        "Mean": 105 + 2 * diet.NOVA4 + 2 * tax.T068 + 5 * diet.NOVA4 * tax.T068 + .2 * (cov.bmi - 26) + rng.normal(0, 4, n),
        "CV": 24 - diet.hPDI - 2 * diet.hPDI * tax.T096 + rng.normal(0, 2, n),
        "SDw": 18 + diet.NOVA4 + tax.T023 + rng.normal(0, 2, n)}
    for outcome in config["outcomes"]:
        clean = np.asarray(values[outcome["id"]])
        if np.min(clean) <= 0:
            clean = clean - np.min(clean) + 1
        cgm[outcome["field"] + "_clean"] = clean
        cgm[outcome["field"] + "_z"] = z(clean)
    save("cgm", cgm)
    return dict(seed=seed, n=n, injected_native_interactions={"NOVA4:T068:Mean": 5.0, "hPDI:T096:CV": -2.0},
                note="SYNTHETIC; all effects and observations generated for software validation")
