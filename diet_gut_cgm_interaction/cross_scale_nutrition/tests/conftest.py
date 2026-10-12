import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def make_panel(n_people=100, seed=812):
    """Fictional data only; no HPP participant is represented."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_people):
        pid = ["001", "NA"][i] if i < 2 else "synthetic_{:04d}".format(i)
        m1 = rng.normal()
        m2 = .35 * m1 + rng.normal(scale=.94)
        baseline = dict(participant_id=pid, connection_id="C" + pid, cohort="10k", research_stage="00_00_visit",
                        age_years=50 + rng.normal(scale=10), sex=str(rng.integers(1, 3)),
                        education_level=str(rng.integers(1, 4)), smoking_status=str(rng.integers(1, 4)),
                        cgm_device_type=str(rng.integers(1, 3)), sleep_duration_hours_day=7 + rng.normal(scale=.8),
                        physical_activity_met_h_week=rng.uniform(2, 80), vitamin_use=int(rng.integers(0, 2)),
                        hormone_use=int(rng.integers(0, 2)), bmi=25 + rng.normal(scale=3), T068=m1, T096=m2,
                        primary_eligible=True, strict_eligible=True)
        days = sorted(rng.choice(28, size=int(rng.integers(8, 16)), replace=False))
        xbar = rng.uniform(20, 60)
        person_sd, person_mean = rng.normal(scale=1), rng.normal(scale=3)
        for j in days:
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(j))
            x = float(np.clip(xbar + rng.normal(scale=13), 2, 79))
            coverage = rng.uniform(.81, 1)
            energy = rng.uniform(1500, 2900)
            alcohol = rng.choice([0., 0., 0., 6., 12.])
            carb = .65 * x + rng.uniform(60, 130)
            protein = rng.uniform(20, 65)
            fiber = rng.uniform(5, 22)
            sd = 17 + person_sd - .3 * xbar / 10 * m1 + .15 * (x - xbar) / 10 * m2 + .03 * x + .003 * carb + rng.normal(scale=.7)
            mean = 98 + person_mean + .1 * xbar / 10 * m2 + .25 * (x - xbar) / 10 * m2 + .05 * x + rng.normal(scale=1.5)
            row = dict(baseline, date=date.strftime("%Y-%m-%d"), day_id=pid + "|C" + pid + "|" + date.strftime("%Y-%m-%d"),
                       nova4_total_pct=x, nova4_classified_pct=x / coverage, nova_coverage=coverage,
                       energy_kcal=energy, alcohol_amed_g=alcohol, alcohol_g=alcohol,
                       alcohol_observed_ok=True, alcohol_recovered_ok=True, diet_pass=True, cgm_pass=True, cgm_strict=True,
                       carb_density=carb, protein_density=protein, fiber_density=fiber, daily_sd=sd, daily_mean=mean,
                       daily_cv=100 * sd / mean, weekday=date.weekday(), observation_day=float(j), sample_minus_day=10.)
            records.append(row)
    panel = pd.DataFrame(records)
    missing = rng.random(len(panel)) < .1
    panel.loc[missing, "fiber_density"] = np.nan
    return panel


@pytest.fixture
def protocol():
    return json.loads((PACKAGE / "config/protocol.json").read_text())


@pytest.fixture
def panel():
    return make_panel()
