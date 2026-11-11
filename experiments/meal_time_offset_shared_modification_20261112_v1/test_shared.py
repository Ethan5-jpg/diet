"""Synthetic checks for objective scaling and training-only offset pooling."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate import LOSS, WEIGHTS, WINDOWS, cohort_transfer
from shared_fit import ENDPOINTS, PENALTY, fit_shared, household_losses, summarize

MODELS = Path(__file__).resolve().parents[1] / "diet_daily_modification_20261112_v1/models.py"
SPEC = spec_from_file_location("archived_meal_models", MODELS)
MODEL = module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class SharedOffsetTest(unittest.TestCase):
    def test_objective_matches_explicit_stacking_and_excludes_held_outcomes(self):
        rng = np.random.default_rng(913)
        n = 90
        z = rng.normal(size=(n, 4))
        w = rng.uniform(.1, .5, n)
        train = np.arange(n) < 64
        data = {(ep, "carbs"): (rng.normal(size=n), rng.normal(size=(n, 5)), w) for ep in ENDPOINTS}
        _, shared = fit_shared(data, z, train, MODEL)
        residuals, targets = [], []
        for ep in ENDPOINTS:
            y, x, _ = data[ep, "carbs"]
            fit = MODEL.fit_background(x[train], y[train], w[train])
            zr, yr, _ = MODEL.orthogonal_block(z[train], fit)
            residuals.append(zr / np.sqrt(3))
            targets.append(yr / np.sqrt(3))
        residual, target = np.vstack(residuals), np.concatenate(targets)
        expected = np.linalg.solve(residual.T @ residual + PENALTY * w[train].sum() * np.eye(4),
                                   residual.T @ target)
        np.testing.assert_allclose(shared, expected, atol=1e-12, rtol=0)
        changed = {k: (y + 100 * ~train, x, w) for k, (y, x, w) in data.items()}
        np.testing.assert_array_equal(shared, fit_shared(changed, z, train, MODEL)[1])

    def test_identical_offsets_do_not_change_the_penalty(self):
        rng = np.random.default_rng(16)
        z = rng.normal(size=(60, 6))
        y, x = rng.normal(size=60), rng.normal(size=(60, 2))
        w = np.full(60, .5)
        data = {(ep, "carbs"): (y, x, w) for ep in ENDPOINTS}
        train = np.arange(60) < 40
        fits, shared = fit_shared(data, z, train, MODEL)
        np.testing.assert_allclose(shared, fits["net_p0"][2], atol=1e-12, rtol=0)
        losses = household_losses(data, z, fits, shared, ~train, np.arange(60), 60)
        np.testing.assert_allclose(losses["net_p0"][1], losses["net_p0"][2], atol=1e-12, rtol=0)

    def test_pooling_recovers_a_shared_signal_in_noisy_synthetic_offsets(self):
        rng = np.random.default_rng(61)
        n = 470
        z = rng.normal(size=(n, 10))
        signal = z[:, :2] @ np.array([.7, -.4])
        common = rng.normal(size=n)
        w = np.full(n, .25)
        train = np.arange(n) < 60
        data = {}
        for ep in ENDPOINTS:
            x = rng.normal(size=(n, 3))
            y = x[:, 0] + signal + common + 2 * rng.normal(size=n)
            data[ep, "carbs"] = (y, x, w)
        fits, shared = fit_shared(data, z, train, MODEL)
        losses = household_losses(data, z, fits, shared, ~train, np.arange(n), n)
        point = summarize(losses, 40)["net_p0"]
        self.assertGreater(point["shared_vs_separate"]["delta_r2"], .01)
        self.assertGreater(point["shared_vs_baseline"]["delta_r2"], .01)

    def test_transfer_requires_and_replays_the_original_receipts(self):
        from contracts import finish, load
        from frozen import fit_frozen, prepare_design, save_arrays

        rng = np.random.default_rng(169)
        index = pd.DataFrame({"input_row": np.arange(24), "split_unit_id": [f"f{i:02}" for i in range(24)],
                              "age_calendar_years": rng.uniform(30, 70, 24), "sex_code": np.arange(24) % 2})
        rows = np.repeat(index.input_row.to_numpy(), 9)
        n = len(rows)
        full = pd.DataFrame({
            "input_row": rows, "carbs": rng.uniform(15, 60, n), "carb_share": rng.uniform(30, 60, n),
            "protein": rng.uniform(7, 40, n), "protein_share": rng.uniform(10, 20, n),
            "energy": rng.uniform(150, 550, n), "fiber": rng.uniform(1, 8, n),
            "weekday": np.arange(n) % 7, "local_hour": rng.uniform(7, 19, n), "prior_gap": rng.uniform(135, 420, n),
            "peak_p0": rng.normal(25, 8, n),
        })
        for ep in ENDPOINTS:
            full[ep] = rng.normal(12, 4, n)
            tag = ep.rsplit("_", 1)[1]
            full[f"pre_{tag}"] = rng.normal(100, 9, n)
            full[f"trend_{tag}"] = rng.normal(0, .1, n)
        pop = np.arange(16)
        train = full.loc[full.input_row.isin(pop)].merge(index, on="input_row").assign(person_index=lambda d: d.input_row)
        held_raw = full.loc[~full.input_row.isin(pop)].copy()
        held = held_raw.merge(index, on="input_row").assign(person_index=lambda d: d.input_row - len(pop))
        scores = rng.normal(size=(len(index), 10))
        a, z = prepare_design(train, scores, MODEL)
        b, u = prepare_design(held, scores, MODEL)
        weights = dict(scores=scores, training_rows=pop, excluded_rows=pop)
        previous = {}
        for ep in ENDPOINTS:
            base, projection, beta = fit_frozen(*a[ep, "carbs"], z, MODEL)
            weights.update({f"{ep}_{k}": v for k, v in zip(("background", "projection", "beta"), (base, projection, beta))})
            y, x, w = b[ep, "carbs"]
            residual = y - x @ base
            adjustment = (u - x @ projection) @ beta
            previous[ep] = np.stack([np.bincount(held.person_index, weights=w * v)
                                     for v in (residual ** 2, (residual - adjustment) ** 2, y ** 2)])
        io = SimpleNamespace(write=lambda p, value: p.write_text(json.dumps(value), encoding="ascii"))
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            frozen, prepared, validated = [project / "runs_TRE_ONLY" / name for name in
                                            ("frozen-1", "prep-discovery-1", "validation-1")]
            for path in (frozen, prepared, validated):
                path.mkdir(parents=True)
            io.write(frozen / "registration.json", {"sources": {}})
            save_arrays(frozen / WEIGHTS, weights)
            finish(frozen, {}, frozen / WEIGHTS, io)
            flow = dict(people=8, households=8, meals=len(held_raw))
            io.write(prepared / "registration.json", {"sources": {}, "smoke": False})
            held_raw.to_parquet(prepared / WINDOWS, index=False)
            finish(prepared, flow, prepared / WINDOWS, io)
            io.write(validated / "registration.json", dict(
                sources={}, frozen_receipt=load(frozen / "COMPLETE.json"), windows_receipt=load(prepared / "COMPLETE.json"),
            ))
            save_arrays(validated / LOSS, previous)
            finish(validated, {"flow": flow}, validated / LOSS, io)
            out = project / "output"
            out.mkdir()
            with patch("evaluate.NEW", project):
                result, _, files = cohort_transfer(out, train, index, pop, MODEL, io, "synthetic")
            self.assertEqual(result["flow"], flow)
            self.assertEqual(set(result["prediction"]), set(ENDPOINTS))
            self.assertTrue(all(p.is_file() for p in files))
            self.assertFalse(load(out / "FROZEN.json")["additional_outcomes_read"])


if __name__ == "__main__":
    unittest.main()
