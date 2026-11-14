"""Household-nested matching and stacking of explicit interaction predictions."""

from collections import Counter
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from interactions import FAMILIES, available_routes, fit_corrections
from moderation_models import BASES, SEED, fit_base, prepare, residuals, scale_columns, scores


BASE = BASES[0]
ZERO = "zero"


def select_candidates(routes, fold_losses):
    candidates = tuple(dict.fromkeys(key for keys in routes.values() for key in keys))
    loss = {key: float(np.sum(fold_losses[key])) for key in candidates}
    choices = {route: min(keys, key=lambda key: round(loss[key], 9)) for route, keys in routes.items()}
    choices["selector"] = min(candidates, key=lambda key: round(loss[key], 9))

    null = np.asarray(fold_losses[ZERO])
    threshold = len(null) // 2 + 1
    # Require an improvement in a strict majority of inner household folds.
    eligible = {ZERO, *(key for key in candidates if loss[key] < loss[ZERO] - 1e-9
                        and np.count_nonzero(np.asarray(fold_losses[key]) < null - 1e-9) >= threshold)}
    stable = {route: min((key for key in keys if key in eligible), key=lambda key: round(loss[key], 9))
              for route, keys in routes.items()}
    choices["selector_stable"] = min((key for key in candidates if key in eligible),
                                     key=lambda key: round(loss[key], 9))
    return choices, stable, loss[ZERO]


def with_panel(role, split, panel):
    columns = (panel[split.train_rows, None], panel[split.rows, None])
    if not all(np.isfinite(x).all() for x in columns):
        raise ValueError("NONFINITE_PANEL_IN_FOLD")
    scaled = scale_columns(*columns, allow_constant=True)
    if np.std(columns[0]) <= 1e-10:
        scaled = tuple(np.zeros_like(x) for x in columns)
    return replace(
        role, design=tuple(np.column_stack([x, z]) for x, z in zip(role.design, scaled, strict=True)),
        markers={**role.markers, "fermentation": scaled},
    )


def base_predict(role):
    return fit_base(role.design[0], role.y[0], role.width, BASE,
                    shared_columns=role.shared_columns).predict(role.design[1])


def stack_weights(matrix, target):
    """Fit nonnegative weights summing to at most one using inner predictions."""
    active = np.any(np.abs(matrix) > 1e-12, axis=0)
    weights = np.zeros(matrix.shape[1])
    if not active.any():
        return weights
    x = matrix[:, active]
    n = len(target)
    gram = x.T @ x / n
    # Shrink noisy combinations toward zero; the scale is prediction variance.
    gram += np.diag(.05 * np.maximum(np.diag(gram), 1e-12))
    linear = x.T @ target / n
    scale = max(np.max(np.diag(gram)), np.max(np.abs(linear)), 1e-12)
    gram, linear = gram / scale, linear / scale
    result = minimize(
        lambda w: (.5 * w @ gram @ w - linear @ w, gram @ w - linear),
        np.zeros(x.shape[1]), jac=True, method="SLSQP", bounds=[(0, 1)] * x.shape[1],
        constraints={"type": "ineq", "fun": lambda w: 1.0 - w.sum(),
                     "jac": lambda w: -np.ones(len(w))},
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all() or result.x.sum() > 1.000001:
        raise ValueError("INVALID_INNER_STACK_OPTIMIZATION")
    weights[active] = np.clip(result.x, 0, 1)
    return weights / max(1.0, weights.sum())


def run_background(inner_roles, outer_role, *, name):
    routes = available_routes(outer_role)
    if not routes or any(available_routes(role) != routes for role in inner_roles):
        raise ValueError("INCONSISTENT_FOLD_ROUTES")
    candidates = tuple(dict.fromkeys(key for keys in routes.values() for key in keys))
    if ZERO not in candidates or "gut_shannon" not in routes:
        raise ValueError("MISSING_REFERENCE_ROUTE")
    target = []
    held = {key: [] for key in candidates}
    fold_losses = {key: [] for key in candidates}
    for role in inner_roles:
        error = role.y[1] - base_predict(role)
        target.append(error)
        corrections = fit_corrections(role, residuals(role, BASE), candidates)
        for key in candidates:
            added = corrections[key][0]
            held[key].append(added)
            fold_losses[key].append(float(np.sum((error - added) ** 2)))
    target = np.concatenate(target)
    inner_added = {key: np.concatenate(values) for key, values in held.items()}
    chosen, stable, null_loss = select_candidates(routes, fold_losses)

    # Use distinct nonzero family winners so a candidate cannot get extra votes.
    def nonzero(keys):
        return tuple(dict.fromkeys(key for key in keys if key != ZERO))

    def stack_matrix(keys):
        return np.column_stack([inner_added[key] for key in keys]) if keys else np.empty((len(target), 0))

    keys = nonzero(chosen[route] for route in routes)
    stacked = stack_matrix(keys)
    equal = np.full(len(keys), 1 / len(keys)) if keys else np.empty(0)
    if keys and np.sum((target - stacked @ equal) ** 2) >= null_loss:
        equal[:] = 0
    ensembles = {f"stack_{name}": (keys, stack_weights(stacked, target)), "stack_equal": (keys, equal)}
    stable_keys = nonzero(stable.values())
    ensembles["stack_stable"] = (stable_keys, stack_weights(stack_matrix(stable_keys), target))
    if name == "panel":
        core_keys = nonzero(chosen[route] for route in routes if "fermentation" not in FAMILIES[route].markers)
        ensembles["stack_core"] = (core_keys, stack_weights(stack_matrix(core_keys), target))

    base = base_predict(outer_role)
    required = tuple(dict.fromkeys([ZERO, *chosen.values(), *stable_keys]))
    outer_residuals = residuals(outer_role, BASE) if len(required) > 1 else None
    final = fit_corrections(outer_role, outer_residuals, required)
    predictions = {route: (base + final[key][0], final[key][1]) for route, key in chosen.items()}
    for route, (stack_keys, weights) in ensembles.items():
        if stack_keys:
            delta, contrast = (np.column_stack([final[key][j] for key in stack_keys]) @ weights for j in (0, 1))
        else:
            delta, contrast = final[ZERO]
        predictions[route] = (base + delta, contrast)
        chosen[route] = "active" if np.any(weights > 1e-10) else ZERO
    if not all(np.isfinite(x).all() for pair in predictions.values() for x in pair):
        raise ValueError("NONFINITE_HELD_PREDICTION")
    return base, predictions, chosen


def run_fold(source, repeat, fold, panel):
    inner, outer = source.roles(repeat, fold)
    held_groups = source.groups[outer.rows]
    if any(np.intersect1d(source.groups[np.r_[split.train_rows, split.rows]], held_groups).size
           for split in inner):
        raise ValueError("INNER_OUTER_HOUSEHOLD_OVERLAP")
    inner_roles = [prepare(source, split, shared_host=True) for split in inner]
    outer_role = prepare(source, outer, association=True, shared_host=True)
    output = {"core": run_background(inner_roles, outer_role, name="core")}
    if panel is not None:
        panel_roles = [with_panel(role, split, panel) for role, split in zip(inner_roles, inner, strict=True)]
        output["panel"] = run_background(panel_roles, with_panel(outer_role, outer, panel), name="panel")
    return outer_role, output


def summarize_prediction(parts, y, groups, order, *, name, repeat):
    names = tuple(parts[0][1])
    if any(tuple(part[1]) != names for part in parts):
        raise ValueError("INCONSISTENT_OUTER_ROUTE_ORDER")
    baseline = np.concatenate([part[0] for part in parts])[order]
    full = np.column_stack([np.concatenate([part[1][key][0] for part in parts])[order] for key in names])
    contrasts = np.column_stack([np.concatenate([part[1][key][1] for part in parts])[order] for key in names])
    base_r2, full_r2, delta, ci = scores(y, baseline, full, groups, seed=SEED + repeat)
    routes = {}
    for j, key in enumerate(names):
        choices = Counter(part[2][key] for part in parts)
        routes[key] = {"full_r2": float(full_r2[j]), "delta_r2": float(delta[j]), "ci95": ci[j].tolist(),
                       "active_folds": len(parts) - choices.get(ZERO, 0), "choices": dict(choices),
                       "contrast_sd_mg_dl": float(contrasts[:, j].std()),
                       "contrast_q25_q75_mg_dl": np.quantile(contrasts[:, j], [0.25, 0.75]).tolist()}
    planned = [(f"stack_{name}", "selector"), (f"stack_{name}", "gut_shannon"),
               ("selector", "gut_shannon"), ("stack_equal", "gut_shannon"),
               ("shannon_fermentation_joint", "gut_shannon"), ("fermentation", "gut_shannon"),
               ("stack_panel", "stack_core"), ("selector_stable", "selector"),
               ("stack_stable", f"stack_{name}"), ("stack_stable", "gut_shannon")]
    comparisons = {}
    for candidate, reference in planned:
        if candidate in routes and reference in routes:
            _, _, gain, interval = scores(y, full[:, names.index(reference)],
                                          full[:, [names.index(candidate)]], groups, seed=SEED + repeat)
            comparisons[f"{candidate}_minus_{reference}"] = {
                "delta_r2": float(gain[0]), "ci95": interval[0].tolist(),
            }
    return {"baseline_r2": float(base_r2), "routes": routes, "comparisons": comparisons}, dict(zip(names, contrasts.T, strict=True))


def model_selftest():
    rng = np.random.default_rng(41)
    x = np.linalg.qr(rng.normal(size=(100, 3)))[0] * 10
    np.testing.assert_allclose(stack_weights(x, -x[:, 0]), [0, 0, 0], atol=1e-8)
    got = stack_weights(x, .55 * x[:, 0] + .35 * x[:, 1])
    np.testing.assert_allclose(got, [.55, .35, 0], atol=.04)
    assert got.sum() <= 1 and (got >= 0).all()
    np.testing.assert_equal(stack_weights(np.zeros_like(x), x[:, 0]), np.zeros(3))
    options = {"a": (ZERO, "a1", "a2"), "b": (ZERO, "b1")}
    chosen, stable, null = select_candidates(options, {ZERO: [10, 10, 10], "a1": [1, 12, 12],
                                                       "a2": [9, 9, 10], "b1": [10, 10, 10]})
    assert chosen["selector"] == "a1" and chosen["a"] == "a1" and chosen["b"] == ZERO
    assert chosen["selector_stable"] == "a2" and stable == {"a": "a2", "b": ZERO} and null == 30
