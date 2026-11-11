"""Local integration test: generate a fake TRE layout and run the real CLI.

Uses historical source code only, entirely generated participants and outcomes.
Requires the local code archive alongside the experiment directory.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import pandas as pd

from exp2_inputs import EXPERIMENTS, PACKAGE, ROOT, V1, V4, adapters
from exp2_store import read_json, write_json


def put_table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def make(package):
    experiments = package / "experiments"
    experiment = experiments / ROOT.parent.name / ROOT.name
    shutil.copytree(ROOT, experiment,
                    ignore=shutil.ignore_patterns("__pycache__", "runs_TRE_ONLY", ".DS_Store"))
    for file in read_json(ROOT / "dependency_manifest.json")["experiments"]:
        dest = experiments / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXPERIMENTS / file, dest)
    for name in ("core.py", "run_pilot.py"):
        dest = experiments / V1 / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXPERIMENTS / "diet_response_heterogeneity_20261103_v1" / name, dest)
    prep = package / "preprocessing/code/preprocessing_v1_20261030.py"
    prep.parent.mkdir(parents=True)
    shutil.copyfile(PACKAGE / "01_CURRENT_CORE/preprocessing/preprocessing_v1_20261030.py", prep)

    bridge, _ = adapters()
    v4 = bridge.imported("exp2_fixture_v4", experiments / V4 / "run_nested.py")
    v4.dependencies(package, experiments / V1 / "code", experiments / "diet_representation_screen_20261103_v3")
    context = bridge.Context
    rng = np.random.default_rng(1053)
    n, good = 4427, 3343
    idx = np.arange(n)
    groups = np.r_[np.repeat(np.arange(284), 2), np.arange(284, n - 284)].astype(str)
    fiber = np.clip(rng.normal(13, 4.1, n), 3, 27)
    fold0, fold1 = (v4.folds(groups, 5, seed) for seed in (17, 20261104))
    energy = rng.uniform(1500, 2650, n)
    table = pd.DataFrame(dict(
        participant_id=idx + 10000, cohort="synthetic", research_stage="one",
        input_row=idx, final_role="discovery", split_unit_id=groups, cv_fold=fold0,
        n_fiber_days=np.where(idx < good, 3, 2), fiber_density=fiber,
        age_calendar_years=rng.uniform(20, 70, n), sex_code=rng.integers(0, 2, n),
        smoking_status=rng.choice(["never", "former", "current"], n), mean_daily_energy_fiber=energy,
        UPF_lower_equalday_fraction=rng.uniform(0, 0.25, n),
        UPF_upper_equalday_fraction=rng.uniform(0.3, 0.6, n),
    ))
    inputs = package / "inputs/discovery_v2"
    put_table(inputs / "discovery_covariates_food_TRE_ONLY.parquet", table)
    raw, taxa = {}, {}
    latent = rng.normal(size=n)
    for site, width in (("oral", 86), ("gut", 200)):
        a = rng.dirichlet(np.linspace(0.35, 1.3, width), size=n)
        if site == "oral":
            a[:, :8] *= np.exp(1.4 * latent[:, None])
            a[:, 8:16] *= np.exp(-1.4 * latent[:, None])
            a /= a.sum(1, keepdims=True)
        raw[site] = a
        taxa[site] = ["sim_%s_%d" % (site, i) for i in range(width)]
        np.savez_compressed(inputs / (site + "_raw_TRE_ONLY.npz"), abundance=a)
        write_json(inputs / (site + "_taxa.json"), taxa[site])
    y = (100 - 0.3 * fiber + 1.6 * latent + 3.4 * latent * (fiber - 13) / 5 +
         rng.normal(0, 4.4, n))
    outcomes = table[context.KEY + ["input_row"]].assign(Y_value=y, Y_unit="mg/dL")
    put_table(inputs / "D_Glu_raw_Y_TRE_ONLY.parquet", outcomes)
    reference = np.full((2, n), np.nan)
    outer = np.stack([fold0, fold1])
    for r, assignments in enumerate(outer):
        for f in range(5):
            valid = assignments == f
            _, processed, _, mask, _ = v4.context(table, raw, taxa, y, ~valid, valid)
            reference[r, mask] = processed[mask]
    assert np.isfinite(reference).sum(axis=1).tolist() == [good, good]
    path = experiments / V4 / "results/TRE_ONLY/final_predictions.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, y=reference, outer=outer,
                        hist_boost_oral=np.zeros_like(reference), hist_boost_oral_gut=np.zeros_like(reference))

    identity = table[context.KEY]
    rows = np.repeat(idx, 3)
    daily = identity.iloc[rows].reset_index(drop=True)
    daily["date"] = np.tile(pd.date_range("2020-01-01", periods=3), n)
    daily["energy_kcal"] = energy[rows]
    daily["fiber_g"] = fiber[rows] * energy[rows] / 1000
    daily["energy_eligible"] = True
    daily["fiber_eligible"] = True
    put_table(package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet", daily)
    nutrient = daily[context.DAY].assign(calories_kcal_complete_sum=daily.energy_kcal,
                                        dietary_fiber_g_complete_sum=daily.fiber_g)
    for j, col in enumerate(context.NUTRIENTS):
        nutrient[col + "_complete_sum"] = rng.uniform(50, 200 + 1000 * j, len(rows))
    shared = package.parent / "analysis/shared"
    put_table(shared / "food_day_qc_TRE_ONLY.parquet", nutrient)
    lifestyle = identity.assign(**{col: rng.uniform(0, hi, n) for col, hi in zip(context.LIFE, (7, 7, 24))})
    lifestyle[context.ALCOHOL] = rng.choice(["never", "weekly", None], n)
    put_table(shared / "covariate_registry_TRE_ONLY.parquet", lifestyle)
    return experiment


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="hpp_exp2_native_sim_") as temp:
        package = Path(temp) / "collaborator_handoff_20261102_v1"
        experiment = make(package)
        env = os.environ.copy()
        env["HPP_EXP2_PYTHON"] = sys.executable
        raw = package / "inputs/discovery_v2/D_Glu_raw_Y_TRE_ONLY.parquet"
        backup = raw.with_suffix(".fixture_backup")
        raw.rename(backup)
        try:
            failed = subprocess.run(["bash", "server_run.sh", "--run-id", "missing_fixture_source"],
                                    cwd=experiment, env=env, capture_output=True, text=True)
            assert failed.returncode == 1 and "Required existing discovery source missing" in failed.stderr
            assert not (experiment / "runs_TRE_ONLY/missing_fixture_source").exists()
        finally:
            backup.rename(raw)
        print("EXP2_NATIVE_MISSING_SOURCE_FIXTURE_OK | failed before reserving a run ID", flush=True)
        for args in [("--preflight",), ("--limit-outer", "1"), ("--resume",)] + [
                ("--screen", str(i)) for i in range(1, 11)]:
            subprocess.run(["bash", "server_run.sh", *args], cwd=experiment, env=env, check=True)
        run_id = read_json(ROOT / "config.json")["version"]
        summary = read_json(experiment / "runs_TRE_ONLY" / run_id / "summary.json")
        assert len(summary["core_routes"]) == 26 and len(summary["added_routes"]) == 20
        assert len(summary["v3_routes"]) == 10
        assert len(summary["release_added_routes"]) == 9
        assert len(summary["expansion_routes"]) == 18
        assert len(summary["followup_routes"]) == 18
        assert all(q["people"] == 3343 and q["households"] == 3059 for q in summary["quality"])
        assert all(s["select_ridge"]["delta"] > 0.05 for s in summary["scores"])
        print("EXP2_NATIVE_LAYOUT_SYNTHETIC_OK | 4427 fake rows, 3343 eligible, 3059 households; "
              "preflight, 1-fold pause, resume, all reports, planted linear signal", flush=True)
