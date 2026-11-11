"""Nested household selection; outer outcomes are used only for scoring."""

from collections import Counter
from time import monotonic
import warnings

import numpy as np
from scipy import optimize

from exp2_design import RoleDesign
from exp2_models import CORE_OTHER, RIDGE, fit_corrections, ridge, specifications


LEGACY_WORKFLOWS = ("select_ridge", "select_other", "select_all", "blend",
                    "select_core_other", "select_v2_other", "select_v1_release_other")
EXPANSION_WORKFLOWS = ("select_new", "select_expanded", "select_guarded")
FOLLOWUP_WORKFLOWS = ("select_added", "select_full", "select_full_guarded")
WORKFLOWS = LEGACY_WORKFLOWS + EXPANSION_WORKFLOWS + FOLLOWUP_WORKFLOWS
ANCHOR_WORKFLOWS = ("select_ridge", "select_expanded", "select_full", "select_full_guarded")
SELECTORS = {
    "select_ridge": lambda spec: spec["family"] in RIDGE,
    "select_other": lambda spec: spec["edition"] <= 4 and spec["family"] not in RIDGE,
    "select_all": lambda spec: spec["edition"] <= 4,
    "select_core_other": lambda spec: spec["family"] in CORE_OTHER,
    "select_v2_other": lambda spec: spec["edition"] <= 2 and spec["family"] not in RIDGE,
    "select_v1_release_other": lambda spec: spec["edition"] <= 3 and spec["family"] not in RIDGE,
    "select_new": lambda spec: spec["edition"] == 5,
    "select_expanded": lambda spec: spec["edition"] <= 5 and spec["family"] not in RIDGE,
    "select_added": lambda spec: spec["edition"] == 6,
    "select_full": lambda spec: spec["family"] not in RIDGE,
}
BASE = "common_main"


def earliest_best(values, tolerance):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).any()
            or np.isnan(values).any() or np.isneginf(values).any()):
        raise ValueError("Invalid selection losses")
    return int(np.flatnonzero(values <= np.min(values) + tolerance)[0])


def fit_blend(corrections, target, tolerance):
    """Inner-only convex shrinkage toward the no-interaction correction."""
    n, k = corrections.shape
    if not k or not np.any(corrections):
        return np.zeros(k), False
    gram = corrections.T @ corrections / n
    gram.flat[::k + 1] += 0.0001
    cross = corrections.T @ target / n

    def objective(w):
        gradient = gram @ w - cross
        return 0.5 * w @ (gradient - cross), gradient

    opt = optimize.minimize(objective, np.zeros(k), jac=True, method="SLSQP",
                            bounds=[(0., 1.)] * k,
                            constraints=[dict(type="ineq", fun=lambda w: 1 - w.sum(),
                                              jac=lambda w: -np.ones(len(w)))],
                            options=dict(ftol=1e-11, maxiter=500))
    if (opt.success and np.isfinite(opt.x).all()
            and np.min(opt.x) >= -1e-6 and opt.x.sum() <= 1.000001):
        w = np.clip(opt.x, 0, 1)
        w /= max(1, w.sum())
        return (w if objective(w)[0] < -tolerance else np.zeros(k)), False

    # The same objective has a closed-form optimum on each single-route axis.
    # These candidates, including zero, satisfy the original blend constraint.
    diagonal = gram.diagonal()
    feasible = np.clip(cross / diagonal, 0, 1)
    loss = 0.5 * diagonal * feasible ** 2 - cross * feasible
    pick = earliest_best(loss, tolerance)
    w = np.zeros(k)
    if loss[pick] < -tolerance:
        w[pick] = feasible[pick]
    return w, True


def household_loss_guard(error, correction, groups, tolerance):
    """Heuristic one-SE threshold on paired inner errors, clustered by household."""
    difference = correction * (2 * error - correction)
    gain = float(difference.mean())
    unique, group = np.unique(groups, return_inverse=True)
    h = len(unique)
    if h < 2:
        return dict(gain=gain, se=None, passed=False)
    centered = np.bincount(group, weights=difference - gain, minlength=h)
    se = float(np.linalg.norm(centered) * np.sqrt(h / (h - 1)) / len(difference))
    return dict(gain=gain, se=se, passed=gain > max(se, tolerance))


def guard_choice(route, error, inner_best, groups, tolerance):
    if route is None:
        return None, dict(gain=0.0, se=None, passed=False)
    check = household_loss_guard(error, inner_best[route], groups, tolerance)
    return (route if check["passed"] else None), check


class Engine:
    def __init__(self, cfg, features, log=print):
        self.cfg, self.features, self.log = cfg, features, log
        self.specs = specifications(cfg)
        self.route_names = [s["name"] for s in self.specs]
        self.legacy_routes = [s["name"] for s in self.specs if s["edition"] <= 4]
        self.names = [BASE] + list(WORKFLOWS) + self.route_names

    def run_unit(self, dataset, repeat, fold):
        start = monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            arrays, meta = self._run_unit(dataset, repeat, fold)
        meta["warnings"] = sorted({str(w.message) for w in caught})
        meta["seconds"] = round(monotonic() - start, 3)
        return arrays, meta

    def _run_unit(self, dataset, repeat, fold):
        cfg, tolerance = self.cfg, self.cfg["selection_tolerance"]
        group = dataset.index.split_unit_id.astype(str).to_numpy()
        assignment = dataset.reference["outer"][repeat]
        train, valid = assignment != fold, assignment == fold
        dataset.api.separated(group, train, valid)
        inner = np.full(len(group), -1)
        seed = cfg["model_seed"] + 100 * repeat + 10 * fold
        inner[train] = dataset.api.folds(group[train], cfg["inner_folds"],
                                        cfg["outer_seed"] + 100 * repeat + fold + 1)
        if set(inner[train]) != set(range(cfg["inner_folds"])):
            raise ValueError("Missing inner fold")
        label = "r%d f%d" % (repeat, fold)
        self.log("START %s training-role features" % label)
        roles = []
        for k in range(cfg["inner_folds"]):
            tr, va = train & (inner != k), train & (inner == k)
            dataset.api.separated(group, tr, va)
            roles.append(RoleDesign(dataset, tr, va, cfg, self.features, seed + k))
        inner_y = np.concatenate([r.split.yv for r in roles])
        inner_groups = np.concatenate([group[r.split.rows] for r in roles])
        main_coefs = [[ridge(r.x[0], r.split.yt, p, free_intercept=True)
                       for p in cfg["main_penalties"]] for r in roles]
        base_matrices = [r.x[1] @ np.column_stack(c) for r, c in zip(roles, main_coefs)]
        main_losses = np.mean((inner_y[:, None] - np.concatenate(base_matrices)) ** 2, axis=0)
        main_pick = earliest_best(main_losses, tolerance)
        main_penalty = cfg["main_penalties"][main_pick]
        inner_error = inner_y - np.concatenate([a[:, main_pick] for a in base_matrices])
        residuals = [r.projection.target_residual(r.split.yt) for r in roles]
        choices, losses, inner_best, status = {}, {}, {}, Counter()
        failures = {}
        self.log("SEARCH %s %d routes; common penalty=%g" % (label, len(self.specs), main_penalty))
        for spec in self.specs:
            key = spec["name"]
            options = list(range(len(spec["settings"])))
            estimates = []
            failed = set()
            for role_index, (r, e) in enumerate(zip(roles, residuals)):
                evaluated = fit_corrections(r, spec, options, e)
                status.update(x[2] for x in evaluated.values() if x[2] not in ("fit", "null"))
                for j, (_, _, flag) in evaluated.items():
                    if flag.startswith("failed_"):
                        failed.add(j)
                        failures.setdefault(key, []).append([role_index, j, flag])
                estimates.append(np.column_stack([evaluated[j][0] for j in options]))
            pooled = np.concatenate(estimates)
            loss = np.mean((inner_error[:, None] - pooled) ** 2, axis=0)
            if not np.isfinite(loss).all():
                raise ValueError("Nonfinite inner loss for " + key)
            loss[list(failed)] = np.inf
            pick = earliest_best(loss, tolerance)
            choices[key] = pick
            losses[key] = [None if j in failed else float(value) for j, value in enumerate(loss)]
            inner_best[key] = pooled[:, pick]
        del roles, main_coefs, base_matrices, residuals, estimates, pooled

        baseline_loss = float(np.mean(inner_error ** 2))
        selected = {}
        for name, includes in SELECTORS.items():
            candidates = [None] + [s["name"] for s in self.specs if includes(s)]
            selected[name] = candidates[earliest_best(
                [baseline_loss if x is None else losses[x][choices[x]] for x in candidates], tolerance)]
        selected["select_guarded"], guard = guard_choice(
            selected["select_expanded"], inner_error, inner_best, inner_groups, tolerance)
        selected["select_full_guarded"], full_guard = guard_choice(
            selected["select_full"], inner_error, inner_best, inner_groups, tolerance)
        weights, blend_fallback = fit_blend(
            np.column_stack([inner_best[key] for key in self.legacy_routes]), inner_error, tolerance)
        if blend_fallback:
            status["blend_fallback"] += 1

        self.log("FIT %s held-out models; settings frozen" % label)
        r = RoleDesign(dataset, train, valid, cfg, self.features, seed + 9)
        s = r.split
        expected = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(s.rows, expected) or not np.array_equal(s.yv, dataset.reference["y"][repeat, expected]):
            raise ValueError("Outer rows or transformed outcomes changed from the frozen reference")
        base_coef = ridge(r.x[0], s.yt, main_penalty, free_intercept=True)
        main = r.x[1] @ base_coef
        e = r.projection.target_residual(s.yt)
        correction, delta = {None: np.zeros(len(s.rows))}, {None: np.zeros(len(s.rows))}
        fixed = {None: np.zeros(len(s.rows))}
        replay = r.at_validation_dose()
        replay_routes = {selected[key] for key in ANCHOR_WORKFLOWS}
        for spec in self.specs:
            key = spec["name"]
            result = fit_corrections(r, spec, [choices[key]], e, contrast=True)
            correction[key], delta[key], flag = result[choices[key]]
            if flag not in ("fit", "null"):
                status["outer_" + flag] += 1
            if flag.startswith("failed_"):
                failures.setdefault(key, []).append(["outer", choices[key], flag])
            if key in replay_routes:
                fixed[key], _, replay_flag = fit_corrections(replay, spec, [choices[key]], e)[choices[key]]
                if replay_flag != flag:
                    raise ValueError("Training status changed during fixed-dose replay: " + key)
        for key, selected_route in selected.items():
            correction[key], delta[key] = correction[selected_route], delta[selected_route]
        correction["blend"] = np.column_stack([correction[k] for k in self.legacy_routes]) @ weights
        delta["blend"] = np.column_stack([delta[k] for k in self.legacy_routes]) @ weights
        correction[BASE], delta[BASE] = correction[None], delta[None]
        support = r.design.support()
        arrays = dict(rows=s.rows, y=s.yv, raw_y=dataset.y[s.rows], support=support,
                      predictions=main[:, None] + np.column_stack([correction[k] for k in self.names]),
                      contrast=np.column_stack([delta[k] for k in self.names]),
                      anchor=np.column_stack([fixed[selected[k]] for k in ANCHOR_WORKFLOWS]))
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite held-out value")
        applied = [key for key in self.names if np.any(np.abs(correction[key]) > 1e-10)]
        meta = dict(applied=applied, main_penalty=main_penalty, main_losses=main_losses.tolist(),
                    inner_losses=losses, choices=choices, selected=selected,
                    weights=weights.tolist(), guard=guard, full_guard=full_guard,
                    status=dict(status), failures=failures,
                    main_columns=int(r.x[0].shape[1]), projected_rank=int(r.projection.q.shape[1]),
                    projected_condition=r.projection.condition)
        return arrays, meta
