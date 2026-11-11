"""Synthetic checks: no HPP native data. python -B exp2_test.py --full."""

import argparse
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from exp2_design import Projection, RoleDesign, anchored_curve, fourier, nystrom, prevalent_clr
from exp2_engine import Engine, SELECTORS, earliest_best, fit_blend, household_loss_guard
from exp2_expansion import component_path, detection_features, rank_curve_fit, soft_profiles
from exp2_followup import FOLLOWUP_FAMILIES, balance_profiles, dose_weights, local_dose_features, two_part_profiles
from exp2_inputs import ROOT, adapters, source_state, synthetic_module, validate_population
from exp2_models import (boosted_curve, fit_corrections, group_curve_fit, huber_ridge,
                         local_curve_corrections, neural_objective, ridge, screened_taxa,
                         specifications)
from exp2_report import display
from exp2_score import house_intervals, summarize, validate_fold
from exp2_store import (load_unit, locked, publish, read_json, read_summary, register,
                        save_unit, unit_path)


def quick():
    rng = np.random.default_rng(881)
    main = rng.normal(size=(120, 8))
    main = np.column_stack([np.ones(len(main)), main, main[:, :2] * 2])
    p = Projection.fit(main[:90], main[90:], 1e-6)
    t = rng.normal(size=(90, 5))
    beta = p.coefficients(t)
    assert p.q.shape[1] == 9
    assert_allclose(p.q.T @ p.residual(t), 0, atol=1e-12)
    assert_allclose(p.transform(t, beta, pos=0), p.residual(t), atol=1e-12)
    # A direction almost indistinguishable in training can move in held data.
    # It must not inflate a well-scaled product by an inverse tiny singular value.
    a = rng.normal(size=90)
    b = rng.normal(size=90)
    ill = np.column_stack([np.ones(90), a, a + 1e-6 * b])
    held_ill = ill[:30].copy()
    held_ill[:, 2] += 0.2
    stable = Projection.fit(ill, held_ill, 0.005)
    assert stable.q.shape[1] == 2 and stable.condition < 200
    assert stable.q_target.shape[1] == 3
    assert_allclose(stable.target_residual(ill @ np.array([5.0, 2.0, 0.2])), 0, atol=1e-12)
    assert np.max(np.abs(stable.transform(np.zeros(30), stable.coefficients(b)))) < 1.0
    x = rng.normal(size=(90, 5))
    d, y = rng.normal(size=(2, 90))
    par, direction = rng.normal(size=(2, 18))
    f = lambda a: neural_objective(a, x, d, p.residual(y), p, 3, 0.08)
    loss, grad = f(par)
    step = 1e-5
    assert np.isfinite(loss)
    assert_allclose(grad @ direction, (f(par + step * direction)[0] - f(par - step * direction)[0]) / (2 * step),
                    atol=1e-8)
    for fun in (fourier, nystrom):
        a, b = fun(x, x[:10], 123)
        a2, b2 = fun(x, np.vstack([1000 * x[10:], x[:10]]), 123)
        assert_allclose(a, a2, atol=1e-12)
        assert_allclose(b, b2[-10:], atol=1e-12)
    assert earliest_best([1, 1 - 1e-9, 2], 1e-8) == 0
    try:
        earliest_best([1, -np.inf], 1e-8)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative infinite loss selected")
    pred = np.column_stack([y, -y, np.zeros_like(y)])
    weights, fallback = fit_blend(pred, -y, 1e-8)
    assert not fallback
    assert_allclose(weights, [0, 1, 0], atol=0.002)
    weights, _ = fit_blend(pred, np.zeros_like(y), 1e-8)
    assert_array_equal(weights, np.zeros(3))
    with patch("exp2_engine.optimize.minimize", return_value=SimpleNamespace(success=False)):
        weights, fallback = fit_blend(pred, -y, 1e-8)
    assert fallback and weights.sum() <= 1.0
    assert_allclose(weights, [0, 1, 0], atol=0.002)
    assert np.mean((-y - pred @ weights) ** 2) < 0.01 * np.mean(y ** 2)
    hh = np.arange(90) // 2
    prediction = np.column_stack([np.zeros_like(y), y, -y])
    ci, rci = house_intervals(y, prediction, hh, (0, 1), 200, 55)
    assert_array_equal(ci[0], np.zeros(2))
    assert_array_equal(rci[1], np.zeros(2))
    assert ci[1, 0] >= 1 and ci[2, 1] < 0
    specs = specifications(read_json(ROOT / "config.json"))
    assert len(specs) == len({s["name"] for s in specs}) == 101
    assert [sum(s["edition"] == k for s in specs) for k in (1, 2, 3, 4, 5, 6)] == [26, 20, 10, 9, 18, 18]
    for key, count in (("select_ridge", 12), ("select_other", 53), ("select_all", 65),
                       ("select_new", 18), ("select_expanded", 71),
                       ("select_added", 18), ("select_full", 89)):
        assert sum(SELECTORS[key](s) for s in specs) == count
    raw = np.array([[1, 0, 2, 0], [1, 3, 0, 0], [1, 5, 0, 0]])
    clr = np.arange(6).reshape(3, 2)
    split = SimpleNamespace(preprocessing={"CLR": {"oral": {
        "reference_indices": [3, 1, 2], "output_indices": [2, 1],
    }}}, ot=clr, ov=clr[:2])
    take, replay = prevalent_clr(split, "oral", raw, 1)
    assert_array_equal(take, clr[:, 1:2])
    assert_array_equal(replay, clr[:2, 1:2])
    # Force an actual convergence warning in a single model option.
    dose = (d[:, None], d[:30, None])
    role = SimpleNamespace(diet=dose, lo=dose[1], hi=dose[1], projection=p,
                           split=SimpleNamespace(rows=np.arange(30)),
                           features={("taxa_sparse", "oral"): (x, x[:30])})
    sparse = next(s for s in specs if s["name"] == "taxa_sparse_oral")
    with patch("exp2_models.ElasticNet.fit", side_effect=ConvergenceWarning("fixture")):
        fits = fit_corrections(role, sparse, [0, 1], p.residual(y))
    assert fits[1][2] == "failed_taxa_sparse_convergence"
    assert_array_equal(fits[1][0], fits[0][0])
    assert all(s["settings"][0] is None and len(s["settings"]) == 4 for s in specs)
    v3_methods()
    release_v2_methods(specs)
    expansion_methods(specs)
    followup_methods(specs)
    print("EXP2_SYNTHETIC_QUICK_OK projection/replay, training-only bases, numeric fits, "
          "latest six families, paired household guard, frozen grid and null controls",
          flush=True)


def v3_methods():
    rng = np.random.default_rng(301)
    linear = np.tile([-1., 1.], 100)[:, None]
    contaminated = 2 * linear[:, 0] + rng.normal(0, 0.1, len(linear))
    contaminated[1] += 300
    robust = huber_ridge(linear, contaminated, 0.003)
    ordinary = ridge(linear, contaminated, 0.003)
    assert abs(robust[0] - 2) < 0.15 and abs(ordinary[0] - 2) > 1.0

    dose = rng.uniform(-1.5, 1.5, 730)
    marker = rng.uniform(-1, 1, len(dose))
    n = 600
    diet = np.column_stack([dose, dose ** 2])
    design = np.column_stack([np.ones(len(dose)), diet, marker])
    projection = Projection.fit(design[:n], design[n:], 0.005)
    low = np.tile([-1.0, 1.0], (len(dose) - n, 1))
    high = np.tile([0.0, 0.0], (len(dose) - n, 1))
    local_diet = anchored_curve((diet[:n], diet[n:], low, high), 0.0)
    assert_allclose(local_diet[0][:, 0] @ local_diet[0][:, 1], 0.0, atol=1e-12)
    assert_array_equal(local_diet[3], high)
    raw = 5 * marker * dose ** 2
    role = SimpleNamespace(local_curve_diet=local_diet, projection=projection)
    valid, _, _ = next(local_curve_corrections(
        role, marker[:n, None], marker[n:, None], projection.target_residual(raw[:n]), [128], False,
    ))
    truth = projection.transform(raw[n:], projection.coefficients(raw[:n]))
    assert np.corrcoef(valid, truth)[0, 1] > 0.90


def release_v2_methods(specs):
    rng = np.random.default_rng(870)
    x = rng.normal(size=(700, 6))
    group = np.tile(np.arange(3), 2)
    truth = 2 * x[:, 1] + 3 * x[:, 4]
    fitted = group_curve_fit(x, truth, group, 0.12)
    assert np.all(fitted[group != 1] == 0)
    assert np.corrcoef(truth, x @ fitted)[0, 1] > 0.99

    n, m = 500, 200
    taxa = rng.normal(size=(n + m, 60))
    dose = rng.uniform(-1.5, 1.5, n + m)
    marker = taxa[:, 47]
    design = np.column_stack([np.ones(n + m), dose])
    x_lo = np.tile([1.0, 0.0], (m, 1))
    x_hi = np.tile([1.0, 1.0], (m, 1))
    projection = Projection.fit(design[:n], design[n:], 0.005)
    raw = 3 * marker + 4 * dose * marker
    d = np.column_stack([dose, dose ** 2])
    lo, hi = np.zeros((m, 2)), np.tile([1.0, 1.0], (m, 1))
    role = SimpleNamespace(
        screen_keep=4, project_rcond=0.005, projection=projection,
        x=(design[:n], design[n:]), x_lo=x_lo, x_hi=x_hi,
        diet=(d[:n], d[n:]), lo=lo, hi=hi,
        split=SimpleNamespace(rows=np.arange(m), yt=raw[:n]),
        features={("taxa_screen", "oral"): (taxa[:n], taxa[n:])},
        contrast=lambda low, high, c: projection.transform(high - low, c, x=x_hi - x_lo),
    )
    residual = projection.target_residual(raw[:n])
    selected, _ = screened_taxa(role, (taxa[:n], taxa[n:]), residual)
    assert_array_equal(selected[:, 0], marker[:n])
    target_route = next(s for s in specs if s["name"] == "taxa_screen_oral")
    prediction, delta, status = fit_corrections(role, target_route, [1], residual, contrast=True)[1]
    assert status == "fit"
    assert np.corrcoef(delta, marker[n:])[0, 1] > 0.97
    assert np.corrcoef(prediction, dose[n:] * marker[n:])[0, 1] > 0.95

    # A nonlinear dose component with a split in microbial coordinates.
    cov = np.column_stack([design, dose ** 2, marker])
    p = Projection.fit(cov[:n], cov[n:], 0.005)
    curved = anchored_curve((d[:n], d[n:], lo, hi), 0.0)
    q = SimpleNamespace(projection=p, local_curve_diet=curved)
    generated = 5 * dose ** 2 * np.where(marker > 0, 1, -1)
    val, _, status = boosted_curve(
        q, marker[:n, None], marker[n:, None], p.target_residual(generated[:n]),
        {"depth": 2, "steps": 32}, 812, False,
    )
    expected = p.transform(generated[n:], p.coefficients(generated[:n]))
    assert status == "fit" and np.corrcoef(val, expected)[0, 1] > 0.9


def expansion_methods(specs):
    rng = np.random.default_rng(241)
    raw = rng.binomial(1, [0.50, 0.02, 0.7, 1.0], size=(180, 4)).astype(float)
    a, b = detection_features(raw[:150], raw[150:], 2)
    a2, b2 = detection_features(raw[:150], np.vstack([np.zeros_like(raw), raw[150:]]), 2)
    assert_array_equal(a, a2)
    assert_array_equal(b, b2[-30:])
    assert set(map(tuple, a.T)) == {tuple(raw[:150, 0]), tuple(raw[:150, 2])}
    train = rng.normal(size=(120, 4))
    test = rng.normal(size=(30, 4))
    a, b = soft_profiles(train, test, 90)
    a2, b2 = soft_profiles(train, np.vstack([test * 100, test]), 90)
    assert_allclose(a.sum(1), 1, atol=1e-14)
    assert_allclose(a, a2, atol=1e-13)
    assert_allclose(b, b2[-30:], atol=1e-13)
    assert soft_profiles(np.ones((120, 4)), test, 90)[0].shape == (120, 0)

    x = np.linalg.qr(rng.normal(size=(120, 12)))[0] * np.sqrt(120)
    planted = np.outer([1.5, -2.0], [0.0, 0.0, 1.0, -0.5, 0.0, 0.0])
    y = x @ planted.ravel()
    coef = rank_curve_fit(x, y, 0.04).reshape(planted.shape)
    assert np.linalg.matrix_rank(coef, tol=1e-6) == 1
    assert_allclose(coef, 0.96 / 1.005 * planted, atol=1e-6)
    path = component_path(x, y, [1, 15, 80])
    loss = [np.mean((x @ beta - y) ** 2) for beta in path]
    assert loss[2] < loss[1] < loss[0] and loss[2] < 0.08 * np.mean(y ** 2)
    assert_array_equal(component_path(x, np.zeros(len(x)), [1, 15])[1], np.zeros(x.shape[1]))

    # Fully crossed dose and two profile axes separate main effects from products.
    states = np.asarray([[d, a, b] for d in (-1., 0., 1.) for a in (-1., 1.) for b in (-1., 1.)])
    design = np.tile(states, (20, 1))
    d, marker, noise = design.T
    train, valid = np.arange(180), np.arange(180, 240)
    base = np.column_stack([np.ones(len(d)), d])
    dose = d[:, None]
    diets = tuple(np.column_stack([dose[i, 0], dose[i, 0] ** 2]) for i in (train, valid))
    lo = np.tile([-1.0, 1.0], (len(valid), 1))
    hi = np.tile([1.0, 1.0], (len(valid), 1))
    main = (base[train], base[valid])
    projector = Projection.fit(*main, 0.005)
    truth = 2 * d * marker
    train_y = 10 + 3 * d + truth
    # Detection and cluster fits must remove the large standalone profile term.
    with_main_y = train_y + 20 * marker
    a, b = marker[:, None], noise[:, None]
    mf = {}
    for key, values in (("elastic", np.column_stack([a, b])), ("fourier", a), ("hellinger", b),
                        ("ridge_pc10", a), ("taxa_sparse", a), ("soft_cluster", a),
                        ("detection_slope", a)):
        mf[(key, "oral")] = (values[train], values[valid])
    role = SimpleNamespace(diet=diets, lo=lo, hi=hi, x=main,
                           x_lo=np.tile([1., -1.], (len(valid), 1)),
                           x_hi=np.tile([1., 1.], (len(valid), 1)),
                           project_rcond=0.005, projection=projector,
                           split=SimpleNamespace(rows=valid, yt=train_y[train]), features=mf)
    role.contrast = lambda low, high, beta: projector.transform(
        high - low, beta, x=role.x_hi - role.x_lo)
    families = {"adaptive_sparse", "kernel_mix_curve", "rank_curve", "component_curve",
                "soft_cluster", "detection_slope"}
    for spec in (s for s in specs if s["family"] in families and s["site"] == "oral"):
        protected = spec["family"] in ("soft_cluster", "detection_slope")
        role.split.yt = (with_main_y if protected else train_y)[train]
        results = fit_corrections(role, spec, [0, 1, 2, 3],
                                  projector.target_residual(role.split.yt), contrast=True)
        value, contrast, status = results[3 if spec["family"] == "component_curve" else 1]
        assert status == "fit", (spec["name"], status)
        assert np.corrcoef(value, truth[valid])[0, 1] > 0.98
        assert np.corrcoef(contrast, marker[valid])[0, 1] > 0.98
        assert np.mean((value - truth[valid]) ** 2) < 0.08 * np.mean(truth[valid] ** 2)
        if protected:
            role.split.yt = (with_main_y - truth)[train]
            null, _, _ = fit_corrections(role, spec, [1], projector.target_residual(role.split.yt))[1]
            assert_allclose(null, 0.0, atol=1e-12)

    groups = np.repeat(np.arange(6), 2)
    correction = np.ones(len(groups))
    strong = household_loss_guard(2 * correction, correction, groups, 1e-8)
    weak_diff = np.repeat([3., -2., 3., -2., 3., -2.], 2)
    weak = household_loss_guard((weak_diff + 1) / 2, correction, groups, 1e-8)
    assert strong == dict(gain=3.0, se=0.0, passed=True)
    assert 0 < weak["gain"] < weak["se"] and not weak["passed"]
    assert not household_loss_guard(2 * correction, correction, np.zeros(len(groups)), 1e-8)["passed"]


def followup_methods(specs):
    rng = np.random.default_rng(3101)
    raw = rng.lognormal(size=(150, 8))
    raw[:, 1:] *= rng.random((150, 7)) > 0.25
    for build in (balance_profiles, two_part_profiles):
        train, held = build(raw[:100], raw[100:])
        changed = np.vstack([raw[:25] * 15, raw[100:] * np.arange(1, 51)[:, None]])
        a, b = build(raw[:100], changed)
        assert_allclose(train, a, atol=1e-12)
        assert_allclose(held, b[-50:], atol=1e-12)
        assert np.isfinite(held).all()
    _, profile = two_part_profiles(raw[:100], raw[100:])
    indicators, magnitude = np.split(profile, 2, axis=1)
    assert np.any(indicators == 0) and np.any(indicators == 1)
    assert_allclose(magnitude[indicators == 0], 0, atol=0)

    rows = np.asarray([[d, m, n] for d in (-2., -1.3, -0.6, 0., 0.6, 1.3, 2.)
                       for m in (-1., 1.) for n in (-1., 1.)])
    dose, marker, noise = np.tile(rows, (12, 1)).T
    tr, va = np.arange(280), np.arange(280, 336)
    diet = np.column_stack([dose, dose ** 2])
    dt, dv = diet[tr], diet[va]
    lo = np.tile([-1.3, 1.3 ** 2], (len(va), 1))
    hi = np.tile([1.3, 1.3 ** 2], (len(va), 1))

    def base(d):
        return np.column_stack([np.ones(len(d)), d])

    main = (base(dt), base(dv))
    projection = Projection.fit(*main, 0.005)
    names = ("balance_curve", "two_part_curve", "ridge_pc10", "ridge_pc30")
    profiles = np.column_stack([marker, noise])
    features = {(key, "oral"): (profiles[tr], profiles[va]) for key in names}
    nuisance = 3 * dose + 14
    truth = 2 * dose * marker
    role = SimpleNamespace(diet=(dt, dv), lo=lo, hi=hi, x=main,
                           x_lo=base(lo), x_hi=base(hi), project_rcond=0.005, projection=projection,
                           dose_bins=local_dose_features((dt, dv, lo, hi)),
                           split=SimpleNamespace(rows=va, yt=nuisance[tr]), features=features)
    role.contrast = lambda low, high, coef: projection.transform(
        high - low, coef, x=role.x_hi - role.x_lo)
    for spec in (s for s in specs if s["site"] == "oral" and s["family"] in FOLLOWUP_FAMILIES):
        family = spec["family"]
        own_main = nuisance + (18 * marker if family in ("balance_curve", "two_part_curve") else 0)
        if family == "local_dose_curve":
            own_main = own_main.copy()
            own_main[tr] += 5 * role.dose_bins[0][:, 0]
        role.split.yt = (own_main + truth)[tr]
        fitted = fit_corrections(role, spec, [1], projection.target_residual(role.split.yt), True)
        value, contrast, status = fitted[1]
        assert status in ("fit", "ard_iteration_cap"), (family, status)
        assert np.corrcoef(value, truth[va])[0, 1] > 0.97, family
        assert np.corrcoef(contrast, marker[va])[0, 1] > 0.97, family
        assert np.mean((value - truth[va]) ** 2) < 0.15 * np.mean(truth[va] ** 2), family
        role.split.yt = own_main[tr]
        value, _, _ = fit_corrections(role, spec, [1], projection.target_residual(role.split.yt))[1]
        assert_allclose(value, 0.0, atol=1e-11, err_msg=family)

    dose = np.r_[np.linspace(-2, 2, 100), 16.0]
    design = dose[:, None]
    y = 2 * dose
    y[-1] += 120
    weights = np.sqrt(dose_weights(dose))
    weighted = ridge(design * weights[:, None], y * weights, 0.01)[0]
    unweighted = ridge(design, y, 0.01)[0]
    assert abs(weighted - 2) < abs(unweighted - 2) * 0.2
    flat = np.zeros((4, 2))
    assert_allclose(local_dose_features((flat, flat))[1], 0.0, atol=0)


def positive_fixture():
    cfg = read_json(ROOT / "config.json")
    _, features = adapters()
    ds = synthetic_module().Dataset(n=960, seed=113, taxa_count=18)
    cfg["expected_people"] = cfg["expected_discovery_rows"] = len(ds.y)
    cfg["expected_households"] = ds.index.split_unit_id.nunique()
    # Plant a dose slope that reverses near 12 and varies with a known
    # compositional factor. A constant microbial slope cannot express it.
    rng = np.random.default_rng(444)
    latent = rng.normal(size=len(ds.y))
    oral = ds.raw["oral"]
    oral[:, :5] *= np.exp(latent[:, None])
    oral[:, 5:10] *= np.exp(-latent[:, None])
    oral /= oral.sum(1, keepdims=True)
    dose = (ds.index.fiber_density.to_numpy() - 12) / 5
    dose_shape = np.abs(dose)
    ds.y += 12 * (dose_shape - dose_shape.mean()) * latent
    ds.reference["y"][:] = ds.y
    return cfg, features, ds


def full():
    cfg, features, ds = positive_fixture()
    validate_population(ds, cfg)
    engine = Engine(cfg, features, lambda m: print(m, flush=True))
    assignment = ds.reference["outer"][0]
    role = RoleDesign(ds, assignment != 0, assignment == 0, cfg, features, cfg["model_seed"])
    low, high = role.at_validation_dose(), role.at_validation_dose(high=True)
    target = role.projection.target_residual(role.split.yt)
    # Exercise each family independently of which ones are selected on this fixture.
    for spec in (s for s in engine.specs if s["site"] == "oral" or s["family"].startswith("oral_gut_")):
        (_, delta, flag), (lo, _, lf), (hi, _, hf) = [
            fit_corrections(r, spec, [1], target, contrast=c)[1]
            for r, c in ((role, True), (low, False), (high, False))]
        assert flag == lf == hf, spec["name"]
        assert_allclose(hi - lo, delta, atol=1e-9, rtol=1e-9, err_msg=spec["name"])
    del role, low, high, target
    print("SYNTHETIC_FIXED_DOSE_REPLAY_OK", flush=True)
    schedule = [(r, f) for r in range(2) for f in range(5)]
    registration = dict(schedule=[list(k) for k in schedule], fixture="synthetic")
    with tempfile.TemporaryDirectory(prefix="hpp_exp2_synthetic_") as temp:
        run = Path(temp)
        from exp2_run import verify_sources
        tracked = run / "fixture_source"
        tracked.write_bytes(b"a")
        state = source_state([tracked])
        replaced = run / "fixture_replacement"
        replaced.write_bytes(b"b")
        replaced.replace(tracked)
        try:
            verify_sources([tracked], state)
        except ValueError as exc:
            assert "inputs changed" in str(exc)
        else:
            raise AssertionError("Source replacement was accepted")
        with locked(run):
            try:
                with locked(run):
                    raise AssertionError("Nested worker should never acquire lock")
            except FileExistsError:
                pass
            digest = register(run, registration, False)
            arrays, meta = engine.run_unit(ds, 0, 0)
            save_unit(run, 0, 0, digest, arrays, meta)
        held = arrays["rows"]
        previous = ds.y[held[0]]
        ds.y[held[0]] = np.nan
        try:
            validate_population(ds, cfg)
        except ValueError as exc:
            assert "nonfinite raw outcome" in str(exc)
        else:
            raise AssertionError("An eligible row with missing raw outcome was accepted")
        finally:
            ds.y[held[0]] = previous
        try:
            validate_fold(ds, engine, 0, 1, arrays)
        except ValueError as exc:
            assert "checkpoint shape" in str(exc) or "Checkpoint rows" in str(exc)
        else:
            raise AssertionError("A valid checkpoint was accepted for the wrong fold")
        old_y = ds.y.copy()
        ds.y[held] += np.linspace(-100, 100, len(held))
        ds.reference["y"][:] = ds.y
        new, altered = engine.run_unit(ds, 0, 0)
        ds.y[:] = old_y
        ds.reference["y"][:] = ds.y
        for key in ("predictions", "contrast", "anchor"):
            assert_allclose(arrays[key], new[key], rtol=0, atol=1e-9)
        for key in ("choices", "selected", "weights", "main_penalty", "guard", "full_guard"):
            assert meta[key] == altered[key]
        assert not np.array_equal(arrays["y"], new["y"])
        print("SYNTHETIC_HELD_OUT_MUTATION_OK", flush=True)
        with locked(run):
            assert register(run, registration, True) == digest
            restored = load_unit(run, 0, 0, digest)
            assert_array_equal(restored[0]["predictions"], arrays["predictions"])
            assert restored[1] == meta
            units = {(0, 0): restored}
            for r, f in schedule[1:]:
                a, m = engine.run_unit(ds, r, f)
                save_unit(run, r, f, digest, a, m)
                units[(r, f)] = load_unit(run, r, f, digest)
                print("SYNTHETIC_OUTER_COMPLETE r%d f%d seconds=%.1f" % (r, f, m["seconds"]), flush=True)
            spline = engine.names.index("ridge_spline_oral")
            max_spline_correction = max(np.max(np.abs(a["predictions"][:, spline] - a["predictions"][:, 0]))
                                        for a, _ in units.values())
            assert max_spline_correction < 60.0, max_spline_correction
            print("SYNTHETIC_STABLE_SPLINE_REPLAY_OK max_correction=%.3f" % max_spline_correction, flush=True)
            summary = summarize(ds, engine, units)
            # The positive control is specific to the new dose-shape family.
            # Print both pooled gains if a later change weakens recovery.
            for route in ("dose_hinge_oral", "select_other", "select_expanded", "select_full"):
                gains = [s[route]["delta"] for s in summary["scores"]]
                assert min(gains) > 0.03, (route, gains)
            publish(run, summary, schedule, digest)
            assert read_summary(run) == summary
            payload = unit_path(run, 1, 4).with_suffix(".npz")
            with payload.open("ab") as stream:
                stream.write(b"corruption")
            try:
                read_summary(run)
            except ValueError as exc:
                assert "SHA256 mismatch" in str(exc)
            else:
                raise AssertionError("Corrupted fold was accepted")
        assert len(summary["core_routes"]) == 26 and len(summary["added_routes"]) == 20
        assert len(summary["v3_routes"]) == 10 and len(summary["release_added_routes"]) == 9
        assert len(summary["expansion_routes"]) == 18
        assert len(summary["followup_routes"]) == 18
        assert all(s["select_full"]["vs_previous"] ==
                   s["select_full"]["r2"] - s["select_expanded"]["r2"] for s in summary["scores"])
        for scores in summary["scores"]:
            model = scores["select_full"]
            assert_allclose(model["dose_replay"]["full_vs_fixed"],
                            model["delta"] - model["dose_replay"]["fixed_only"], atol=1e-12)
        assert all(s["select_expanded"]["vs_legacy"] ==
                   s["select_expanded"]["r2"] - s["select_other"]["r2"] for s in summary["scores"])
        assert all(sum(x["people"] for x in q["support_strata"].values()) == len(ds.y)
                   for q in summary["quality"])
        assert len(summary["joint_comparison"]) == 2
        for i in range(1, 11):
            display(summary, i, synthetic=True)
        # Exact additive signal in the shared main space: no route should
        # manufacture useful correction from machine-precision residuals.
        ds.y[:] = 20 + 0.75 * ds.index.age_calendar_years.to_numpy()
        ds.reference["y"][:] = ds.y
        additive, fitted = engine.run_unit(ds, 0, 0)
        assert all(j == 0 for j in fitted["choices"].values())
        assert all(route is None for route in fitted["selected"].values())
        assert_array_equal(additive["predictions"],
                           np.repeat(additive["predictions"][:, :1], len(engine.names), axis=1))
        assert_array_equal(additive["contrast"], np.zeros_like(additive["contrast"]))
        assert_array_equal(additive["anchor"], np.zeros_like(additive["anchor"]))
        print("SYNTHETIC_EXACT_ADDITIVE_NULL_OK", flush=True)
        print("EXP2_FULL_SYNTHETIC_OK 10/10 folds; held-out invariance, checkpoint/resume, "
              "corruption rejection, additive null and known interaction recovery", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        quick()
        if args.full:
            full()
