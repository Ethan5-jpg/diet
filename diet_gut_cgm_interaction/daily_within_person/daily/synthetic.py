"""Deterministic fictional fixture; no local or server cohort records are read."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json


def create_fixture(root, config, n_people=80, n_days=12):
    root = Path(root)
    rng = np.random.RandomState(741)
    foods, sensors, people, cov, micro, dates = [], [], [], [], [], []
    for i in range(n_people):
        pid, connection = "SYNTH_{:04d}".format(i), "SYNTH_CONN_{:04d}".format(i)
        keys = dict(participant_id=pid, cohort=config["cohort"], research_stage=config["research_stage"])
        m1, m2 = rng.normal(size=2)
        age, bmi = rng.uniform(30, 70), rng.uniform(19, 35)
        people.append(dict(**keys, connection_id=connection, cgm_device_type="SYNTH_DEVICE"))
        flag = np.nan if i % 13 == 0 else 0
        cov.append(dict(**keys, connection_id=connection, age_years=age, sex="F" if i % 2 else "M",
                        education_level="SYNTH", smoking_status="never", sleep_duration_hours_day=7,
                        physical_activity_met_h_week=15, vitamin_use=0, hormone_use=0, bmi=bmi,
                        cgm_device_type="SYNTH_DEVICE", exclude_diabetes_or_a10=flag))
        micro.append(dict(**keys, Eubacterium_rectale=m1, Ruthenibacterium_lactatiformans=m2, sample_name="SYNTH_SAMPLE_" + str(i)))
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.randint(0, 80)))
        dates.append(dict(**keys, sample_name="SYNTH_SAMPLE_" + str(i), collection_date=(start + pd.Timedelta(days=n_days + 2)).strftime("%Y-%m-%d")))
        baseline = rng.normal(102, 5)
        for j in range(n_days):
            date = (start + pd.Timedelta(days=j)).strftime("%Y-%m-%d")
            midnight = pd.Timestamp(date, tz=config["time"]["local_timezone"])
            energy = rng.uniform(1500, 2900)
            nova = rng.uniform(.08, .85)
            fiber_density = rng.uniform(6, 23)
            carb_density, protein_density = rng.uniform(85, 145), rng.uniform(20, 65)
            alcohol = float(rng.choice([0, 0, 0, 5, 10, 15]))
            for hour in [7.5, 12.5, 18.0]:
                stamp = midnight + pd.Timedelta(hours=hour)
                for food_id, fraction in [("SYNTH_UPF", nova), ("SYNTH_OTHER", 1 - nova)]:
                    kcal = energy * fraction / 3
                    foods.append(dict(**keys, collection_timestamp=stamp.isoformat(), collection_date=date,
                        timezone=config["time"]["local_timezone"].lower(), food_id=food_id, calories_kcal=kcal,
                        carbohydrate_g=carb_density * kcal / 1000, protein_g=protein_density * kcal / 1000,
                        lipid_g=kcal * .3 / 9, dietary_fiber_g=fiber_density * kcal / 1000,
                        alcohol_g=alcohol * fraction / 3))
            x, f = nova * 10, fiber_density / 5
            sd = max(3, 17 + .5 * x - .3 * x * m1 + .15 * f * m2 + rng.normal(0, .7))
            mean = baseline + .5 * x + .1 * f + rng.normal(0, 1)
            phase = np.arange(96) * 2 * np.pi / 96
            wave = np.sin(phase) + .4 * np.sin(phase * 3 + .3)
            wave = (wave - wave.mean()) / wave.std(ddof=1)
            for k, glucose in enumerate(mean + sd * wave):
                stamp = midnight + pd.Timedelta(minutes=15 * k)
                sensors.append(dict(**keys, connection_id=connection, collection_timestamp=stamp.isoformat(), glucose=glucose))
    frames = {"diet": pd.DataFrame(foods), "connections": pd.DataFrame(people), "covariates": pd.DataFrame(cov),
              "microbiome": pd.DataFrame(micro), "microbiome_dates": pd.DataFrame(dates),
              "nova_mapping": pd.DataFrame([dict(food_id="SYNTH_UPF", nova_group=4, nova_mapping_status="mapped"),
                                             dict(food_id="SYNTH_OTHER", nova_group=1, nova_mapping_status="mapped")])}
    for label, frame in frames.items():
        p = root / config["inputs"][label]
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)
    p = root / "Transfer/cgm/timeseries/SYNTHETIC.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensors).to_csv(p, index=False)
    write_json(root / "SYNTHETIC_ONLY.json", dict(seed=741, people=n_people, days_per_person=n_days, truth="SD NOVA4 x T068 = -0.3 per 10pp per original CLR-Z; fictional"))
    return dict(seed=741, people=n_people, days_per_person=n_days)
