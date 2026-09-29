"""Artificial inputs ONLY. Values do not describe HPP participants."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_data(root, config, people=12, days=6):
    root = Path(root)
    (root / "cgm").mkdir(parents=True)
    (root / "SYNTHETIC_TEST_ONLY.json").write_text('{"data_kind":"SYNTHETIC"}')
    rng = np.random.default_rng(390)
    meals, covs, microbes = [], [], []
    for person in range(people):
        pid = "{:05d}".format(person)
        offset = pd.Timestamp("2021-12-14 00:00:00", tz="Asia/Jerusalem")
        clock = pd.date_range(offset, periods=days * 96 + 1, freq="15min")
        baseline = 90 + person * 0.2 + np.sin(np.arange(len(clock)) / 60) * 4 + rng.normal(0, 0.5, len(clock))
        glucose = baseline.copy()
        micro = rng.normal()
        for day in range(days):
            for hour in [8, 13, 19]:
                t = offset + pd.Timedelta(days=day, hours=hour, minutes=int(rng.choice([-15, 0, 15])))
                nova = float(rng.choice([0.1, 0.3, 0.6, 0.9]))
                energy, carbs = rng.uniform(300, 700), rng.uniform(30, 90)
                rise = max(5, 20 + carbs * 0.2 + 12 * nova + 7 * micro * nova + rng.normal(0, 3))
                relative = (clock - t).total_seconds().to_numpy() / 60
                glucose += np.maximum(0, 1 - np.abs(relative - 60) / 60) * rise
                for food, share, delay in [("A", 1 - nova, 0), ("B", nova, 5)]:
                    meals.append(dict(participant_id=pid, cohort="10k", research_stage="00_00_visit",
                                      collection_timestamp=(t + pd.Timedelta(minutes=delay)).isoformat(),
                                      food_id=food, weight_g=100 * share, calories_kcal=energy * share,
                                      carbohydrate_g=carbs * share, lipid_g=15 * share, protein_g=20 * share,
                                      dietary_fiber_g=5 * share, timezone="Asia/Jerusalem"))
        cgm = pd.DataFrame(dict(participant_id=pid, cohort="10k", research_stage="00_00_visit",
                                connection_id="sensor" + pid, collection_timestamp=[x.isoformat() for x in clock], glucose=glucose))
        # Deliberately reverse order: downstream assembly must sort actual times.
        cgm.iloc[::-1].to_csv(root / "cgm" / (pid + ".csv"), index=False)
        cov = dict(participant_id=pid, cohort="10k", cgm_research_stage="00_00_visit", exclude_diabetes_or_a10="0")
        for name in config["analysis"]["require_covariate_columns"]:
            cov[name] = "1"
        covs.append(cov)
        microbes.append(dict(participant_id=pid, cohort="10k", research_stage="00_00_visit", Eubacterium_rectale=micro,
                             Ruthenibacterium_lactatiformans=rng.normal()))
    # Random ordering plus tiny chunks exercises cross-chunk meal aggregation.
    pd.DataFrame(meals).sample(frac=1, random_state=6).to_csv(root / "diet.csv", index=False)
    pd.DataFrame(covs).to_csv(root / "cov.csv", index=False)
    pd.DataFrame(microbes).to_csv(root / "micro.csv", index=False)
    pd.DataFrame(dict(food_id=["A", "B"], nova_group=[1, 4], nova_mapping_status=["mapped", "mapped"])).to_csv(root / "nova.csv", index=False)
    c = copy.deepcopy(config)
    c["data_kind"] = "SYNTHETIC"
    c["time"]["consumption_time_verified"] = True
    c["inputs"] = dict(diet="diet.csv", cgm_glob="cgm/*.csv", nova_mapping="nova.csv",
                       covariates="cov.csv", microbiome="micro.csv", microbiome_dates=None)
    c["analysis"].update(minimum_people=8, minimum_meals_per_person=4)
    c["plots"]["example_meals"] = 2
    c["ingest"]["chunksize"] = 137
    path = root.parent / "synthetic_config.json"
    path.write_text(json.dumps(c))
    return c, path
