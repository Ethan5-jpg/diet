"""Frozen person-level fits for 10-to-15 fiber contrasts and matched prediction."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


SEED = 20261112
MARKERS = ("oral_shannon", "oral_dominance", "gut_shannon", "gut_dominance")
SITES = ("oral", "gut")
BASES = (("shallow", 100, 7, 30), ("medium", 160, 15, 28))
ZERO = ("zero", 0, 0, 0.0)
FIXED = {name: (ZERO, *((name, 1, 1, penalty) for penalty in (.1, 1.0)))
         for name in MARKERS}
LEARNED = {f"learned_{site}": (ZERO, *((f"learned_{site}", dose, keep, penalty)
           for dose in (1, 2) for keep in (5, 12) for penalty in (.1, 1.0)))
           for site in (*SITES, "both")}
OPTIONS = {**FIXED, **LEARNED}
OPTIONS["selector"] = tuple(dict.fromkeys(key for keys in OPTIONS.values() for key in keys))
CANDIDATES = OPTIONS["selector"]


@dataclass(frozen=True)
class Role:
    rows: np.ndarray
    y: tuple
    design: tuple
    width: int
    groups: np.ndarray
    dose: tuple
    delta: np.ndarray
    markers: dict
    raw_eco: tuple
    association_main: np.ndarray
    shared_columns: tuple = ()


def ecology(x):
    x = np.asarray(x, dtype=float)
    total = x.sum(1, keepdims=True)
    if not np.isfinite(x).all() or (x < 0).any() or (total <= 0).any():
        raise ValueError("INVALID_ECOLOGY_PROFILE")
    p = x / total
    return np.column_stack((-(p * np.log(np.maximum(p, 1e-300))).sum(1), p.max(1)))


def scale_columns(train, valid, allow_constant=False):
    center, scale = train.mean(0), train.std(0)
    use = scale > 1e-10
    if allow_constant:
        scale = np.where(use, scale, 1)
        return (train - center) / scale, (valid - center) / scale
    return (train[:, use] - center[use]) / scale[use], (valid[:, use] - center[use]) / scale[use]


def context(extra, train, valid):
    if extra is None:
        return np.empty((len(train), 0)), np.empty((len(valid), 0))
    alcohol = np.asarray(extra["alcohol"], dtype=object)
    levels = sorted({v for v in alcohol[train] if isinstance(v, str) and v.strip()})
    return tuple(np.column_stack([
        extra["blocks"]["full"][rows], extra["blocks"]["distribution"][rows],
        *((alcohol[rows] == level).astype(float) for level in levels),
    ]) for rows in (train, valid))


def finite_inference_background(host, ctx, pc, eco):
    """Frozen nuisance features. Held-out outcomes never choose columns."""
    finite = np.isfinite(ctx[0])
    medians = np.array([np.median(c[ok]) if ok.any() else 0.0 for c, ok in zip(ctx[0].T, finite.T, strict=True)])
    missing = ~np.isfinite(ctx[1])
    return np.column_stack([
        np.ones(len(host)), host, np.where(missing, medians, ctx[1]), missing,
        *(block[1][:, :6] for block in pc), eco[1], eco[1] ** 2,
    ])


def _shared_host_columns(split):
    """Find only the retained V4 age and sex main-effect columns."""
    try:
        labels = split.preprocessing["host"]["labels"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("MISSING_SHARED_HOST_LABELS") from exc
    if (not isinstance(labels, (list, tuple))
            or len(labels) != split.ht.shape[1] or len(labels) != split.hv.shape[1]
            or not all(isinstance(label, str) for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError("INVALID_SHARED_HOST_LABELS")
    return tuple(j for j, label in enumerate(labels) if label in ("age", "sex"))


def prepare(source, split, association=False, shared_host=False):
    train, valid = split.train_rows, split.rows
    groups = source.groups[train]
    if np.intersect1d(groups, source.groups[valid]).size:
        raise ValueError("HOUSEHOLD_ROLE_OVERLAP")
    raw_sites, pc_sites = [], []
    for name, a, b in (("oral", split.ot, split.ov), ("gut", split.gt, split.gv)):
        raw_sites.append(tuple(ecology(source.raw[name][rows]) for rows in (train, valid)))
        k = min(12, a.shape[1], len(a) - 1)
        if k < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("INVALID_MICROBIOME_DESIGN")
        pca = PCA(n_components=k, svd_solver="randomized", random_state=SEED).fit(a)
        pc_sites.append(scale_columns(pca.transform(a), pca.transform(b)))
    raw = tuple(np.column_stack([site[pos] for site in raw_sites]) for pos in (0, 1))
    eco = scale_columns(*raw, allow_constant=True)
    markers = {name: tuple(z[:, j:j + 1] for z in eco) for j, name in enumerate(MARKERS)}
    for j, site in enumerate(SITES):
        markers[f"learned_{site}"] = tuple(
            np.column_stack([pc_sites[j][pos], eco[pos][:, 2*j:2*j + 2]]) for pos in (0, 1)
        )
    markers["learned_both"] = tuple(np.column_stack([markers[f"learned_{s}"][pos] for s in SITES]) for pos in (0, 1))
    ctx = context(getattr(source, "extra", None), train, valid)
    host = (split.ht, split.hv)
    left = tuple(np.column_stack([host[pos], ctx[pos]]) for pos in (0, 1))
    design = tuple(np.column_stack([left[pos], *(pc[pos] for pc in pc_sites), eco[pos]]) for pos in (0, 1))
    if any(np.isinf(x).any() for x in design) or not all(np.isfinite(y).all() for y in (split.yt, split.yv)):
        raise ValueError("NONFINITE_MATCHED_DESIGN")
    center = split.dt.mean(0)
    main = finite_inference_background(split.hv, ctx, pc_sites, eco) if association else np.empty((len(valid), 0))
    shared_columns = _shared_host_columns(split) if shared_host else ()
    return Role(valid, (split.yt, split.yv), design, left[0].shape[1], groups,
                (split.dt - center, split.dv - center), split.delta, markers, raw, main, shared_columns)


def fit_base(x, y, width, config, shared_columns=()):
    _, steps, leaves, min_leaf = config
    return HistGradientBoostingRegressor(
        max_iter=steps, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        learning_rate=.07, l2_regularization=8.0, max_bins=63, early_stopping=False,
        interaction_cst=[tuple(range(width)), (*shared_columns, *range(width, x.shape[1]))],
        random_state=SEED,
    ).fit(x, y)


def residuals(role, config):
    x, y = role.design[0], role.y[0]
    pred = np.full(len(y), np.nan)
    # Refit the tree learner using only subfolds of the current training role.
    for t, v in GroupKFold(3).split(x, groups=role.groups):
        pred[v] = fit_base(x[t], y[t], role.width, config, shared_columns=role.shared_columns).predict(x[v])
    if not np.isfinite(pred).all():
        raise ValueError("INCOMPLETE_TRAINING_RESIDUALS")
    return y - pred


def correction(role, residual, name, n_dose, k, strength):
    if name == "zero":
        return np.zeros(len(role.rows)), np.zeros(len(role.rows))
    train, valid = role.markers[name]
    d0, d1 = (d[:, :n_dose] for d in role.dose)
    # Only the training role's residuals rank candidate product columns.
    train_cube = d0[:, :, None] * train[:, None, :]
    norms = np.sqrt(np.sum(train_cube ** 2, axis=0))
    signal = np.einsum("ndp,n->dp", train_cube, residual)
    value = np.divide(signal, norms, out=np.zeros_like(signal), where=norms > 1e-10)
    keep = np.argsort(-np.linalg.norm(value, axis=0), kind="stable")[:k]
    a = train_cube[:, :, keep].reshape(len(train), -1)
    b = (d1[:, :, None] * valid[:, None, keep]).reshape(len(valid), -1)
    c = (role.delta[None, :n_dose, None] * valid[:, None, keep]).reshape(len(valid), -1)
    scale = np.sqrt(np.mean(a ** 2, axis=0))
    use = scale > 1e-10
    a = a[:, use] / scale[use]
    if not a.shape[1]:
        return np.zeros(len(valid)), np.zeros(len(valid))
    gram = a.T @ a / len(a)
    coef = solve(gram + strength * np.eye(len(gram)), a.T @ residual / len(a), assume_a="pos")
    weights = coef / scale[use]
    return b[:, use] @ weights, c[:, use] @ weights


def run_fold(source, repeat, fold, bases=BASES, options=OPTIONS, shared_host=False):
    inner, outer = source.roles(repeat, fold)
    groups = source.groups[outer.rows]
    if any(np.intersect1d(source.groups[np.r_[s.train_rows, s.rows]], groups).size for s in inner):
        raise ValueError("INNER_OUTER_HOUSEHOLD_OVERLAP")
    prepared = [prepare(source, split, shared_host=shared_host) for split in inner]
    candidates = tuple(dict.fromkeys(key for keys in options.values() for key in keys))
    fits = []
    for base in bases:
        predictions = [
            fit_base(r.design[0], r.y[0], r.width, base, shared_columns=r.shared_columns).predict(r.design[1])
            for r in prepared
        ]
        loss = sum(np.sum((r.y[1] - p) ** 2) for r, p in zip(prepared, predictions, strict=True))
        fits.append((loss, base, predictions))
    _, base, base_predictions = min(fits, key=lambda item: item[0])
    scores = dict.fromkeys(candidates, 0.0)
    for role, fitted in zip(prepared, base_predictions, strict=True):
        residual = residuals(role, base)
        for key in candidates:
            delta, _ = correction(role, residual, *key)
            scores[key] += np.sum((role.y[1] - fitted - delta) ** 2)
    choices = {name: min(keys, key=lambda key: round(scores[key], 9)) for name, keys in options.items()}
    role = prepare(source, outer, association=True, shared_host=shared_host)
    prediction = fit_base(role.design[0], role.y[0], role.width, base,
                          shared_columns=role.shared_columns).predict(role.design[1])
    residual = residuals(role, base) if any(key != ZERO for key in choices.values()) else None
    cache = {key: correction(role, residual, *key) for key in set(choices.values())}
    full = np.column_stack([prediction + cache[key][0] for key in choices.values()])
    contrasts = np.column_stack([cache[key][1] for key in choices.values()])
    if not np.isfinite(full).all() or not np.isfinite(contrasts).all():
        raise ValueError("INVALID_HELD_OUT_PREDICTION")
    return role, prediction, full, contrasts, choices, base[0]


def scores(y, baseline, full, groups, seed=SEED, draws=1999):
    y0 = y - y.mean()
    sst = y0 @ y0
    if not np.isfinite(sst) or sst <= 0:
        raise ValueError("INVALID_POOLED_SST")
    err0, err1 = (y - baseline) ** 2, (y[:, None] - full) ** 2
    gain = err0[:, None] - err1
    _, g = np.unique(groups, return_inverse=True)
    stats = np.column_stack([np.bincount(g), *(np.bincount(g, weights=x) for x in (y0, y0 ** 2, *gain.T))])
    h = len(stats)
    rng, samples = np.random.default_rng(seed), []
    for start in range(0, draws, 48):
        pick = rng.integers(0, h, (min(48, draws - start), h))
        total = np.vstack([np.bincount(row, minlength=h) for row in pick]) @ stats
        denominator = total[:, 2] - total[:, 1] ** 2 / total[:, 0]
        if (denominator <= 0).any():
            raise ValueError("DEGENERATE_HOUSEHOLD_BOOTSTRAP")
        samples.append(total[:, 3:] / denominator[:, None])
    return (1 - err0.sum() / sst, 1 - err1.sum(0) / sst, gain.sum(0) / sst,
            np.quantile(np.concatenate(samples), [.025, .975], axis=0).T)
