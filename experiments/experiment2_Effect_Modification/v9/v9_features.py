"""Fold-fitted microbial coordinates. All learned parameters use training households."""

import numpy as np
from scipy.special import entr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from scipy.linalg import solve

from adapters.fermentation_score import MASS_UNIT, resolve_species
import v8_models as v8
import v9_grid as grid


SEED = 914279
SPECIES = ("Ruminococcus bromii", "Bifidobacterium adolescentis", "Eubacterium rectale")


def empirical_rank(train, values):
    ordered = np.sort(train)
    return (np.searchsorted(ordered, values, "left") + np.searchsorted(ordered, values, "right")) / (
        2 * len(train)
    )


def columns(bank, names):
    return tuple(np.column_stack([bank[name][side] for name in names]) for side in (0, 1))


def indicators(train, valid, width):
    """Drop the largest training category, then pad unused coordinates with zero."""
    ids, counts = np.unique(train, return_counts=True)
    selected = np.delete(ids, np.argmax(counts))
    if len(selected) > width or not np.isin(valid, ids).all():
        raise ValueError("Invalid partition support")

    def encode(x):
        out = np.zeros((len(x), width))
        out[:, : len(selected)] = x[:, None] == selected
        return out

    return v8.centered(encode(train), encode(valid))


def add_partitions(bank, bases, y, u):
    def register(names, pair):
        if any(mat.shape[1] != len(names) for mat in pair):
            raise ValueError("Unexpected partition width")
        for j, name in enumerate(names):
            bank[name] = pair[0][:, j], pair[1][:, j]

    for key, (k, features) in grid.REGION_VIEWS.items():
        a, b = columns(bank, features)
        scale = a.std(axis=0)
        scale = np.where(scale > 1e-10, scale, 1.0)
        a, b = a / scale, b / scale
        count = min(k, len(np.unique(a, axis=0)))
        km = KMeans(n_clusters=count, n_init=4, random_state=SEED, algorithm="lloyd").fit(a)
        # Validation can be closest to a numerically empty center. Restrict the
        # assignment to occupied training centers in that degenerate case.
        used = np.unique(km.labels_)
        bv = used[np.argmin(km.transform(b)[:, used], axis=1)]
        register(grid.region_labels(key), indicators(km.labels_, bv, k - 1))

    # A shallow partition can uncover nonlinear slope patterns. Only the training
    # outcome enters the pseudo-response. Every leaf has its own main effect in
    # the paired fit and the exact same partition is used in both arms.
    x = np.column_stack((bases[0][:, 1:], columns(bank, grid.ECO)[0]))
    x -= x.mean(axis=0)
    sd = x.std(axis=0)
    x /= np.where(sd > 1e-10, sd, 1.0)
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += 0.1 * len(x)
    centered_y = y - np.mean(y)
    residual = centered_y - x @ solve(gram, x.T @ centered_y, assume_a="pos")
    pseudo = (u - u.mean()) * residual
    for key, (depth, site) in grid.TREE_VIEWS.items():
        features = grid.BASE + (grid.v7.pc_list("o", 5) if site == "both" else ())
        a, b = columns(bank, features)
        model = ExtraTreesRegressor(
            n_estimators=grid.TREE_COUNT,
            max_depth=depth,
            min_samples_leaf=max(6, int(np.ceil(0.04 * len(y)))),
            max_features=1.0,
            random_state=SEED,
            n_jobs=1,
        ).fit(a, pseudo)
        ta, vb = model.apply(a), model.apply(b)
        blocks = [indicators(ta[:, j], vb[:, j], 2**depth - 1) for j in range(grid.TREE_COUNT)]
        register(grid.tree_labels(key), tuple(np.column_stack(side) for side in zip(*blocks, strict=True)))


def build(source, split):
    bank, bases, old_dose = v8.background_and_features(source, split)
    for name in grid.ECO:
        a, b = bank[name]
        med = np.median(a)
        iqr = np.subtract(*np.quantile(a, [0.75, 0.25]))
        bank["rank_" + name] = v8.centered(*(empirical_rank(a, x) for x in (a, b)))
        bank["bounded_" + name] = v8.centered(
            *(np.tanh((x - med) / (iqr if iqr > 1e-10 else 1.0)) for x in (a, b))
        )
    for name in (*grid.ECO, "rank_shannon", "rank_fermentation", "bounded_shannon", "bounded_fermentation"):
        for j in range(1, 11 if name in grid.ECO else 5):
            key = f"v7_gate_{name}_g{j}"
            if key not in bank:
                pc = bank[f"g{j}"]
                center = pc[0].mean()
                bank[key] = v8.centered(*((x - center) * y for x, y in zip(pc, bank[name], strict=True)))
    for g in ("g1", "g2"):
        for o in ("o1", "o2"):
            a, b = bank[g], bank[o]
            bank[g + "_x_" + o] = v8.centered(
                *((x - a[0].mean()) * (y - b[0].mean()) for x, y in zip(a, b, strict=True))
            )

    matched = resolve_species(source.taxa["gut"])
    indices = [matched[s][0] for s in SPECIES]
    metrics = []
    for rows in (split.train_rows, split.rows):
        gut = np.asarray(source.raw["gut"][rows], dtype=np.float64)
        oral = np.asarray(source.raw["oral"][rows], dtype=np.float64)
        if (
            not np.isfinite(oral).all()
            or not np.isfinite(gut).all()
            or np.any(oral < 0)
            or np.any(gut < 0)
            or not (oral.sum(axis=1) > 0).all()
            or not (gut.sum(axis=1) > 0).all()
        ):
            raise ValueError("Invalid microbial abundance on eligible role")
        logs = np.log1p(gut[:, indices] / MASS_UNIT)
        gut = gut / gut.sum(axis=1, keepdims=True)
        oral = oral / oral.sum(axis=1, keepdims=True)
        metrics.append(
            (
                *logs.T,
                gut.max(axis=1),
                1.0 - np.einsum("ij,ij->i", gut, gut),
                entr(oral).sum(axis=1),  # The entropy contribution of a zero abundance is zero.
                oral.max(axis=1),
            )
        )
    labels = (
        "log_bromii",
        "log_adolescentis",
        "log_rectale",
        "gut_dominance",
        "gut_simpson",
        "oral_shannon",
        "oral_dominance",
    )
    for name, pair in zip(labels, zip(*metrics, strict=True), strict=True):
        bank[name] = v8.centered(*pair)
    dose = (split.dt, split.dv)
    low, high = np.quantile(dose[0][:, 0], [0.25, 0.75])
    bank["dose_3"] = v8.centered(*(np.maximum(low - x[:, 0], 0.0) for x in dose))
    bank["dose_4"] = v8.centered(*(np.maximum(x[:, 0] - high, 0.0) for x in dose))
    doses = [(*old_dose[i], dose[i][:, 1], bank["dose_3"][i], bank["dose_4"][i]) for i in (0, 1)]
    add_partitions(bank, bases, split.yt, dose[0][:, 0])
    return bank, bases, doses
