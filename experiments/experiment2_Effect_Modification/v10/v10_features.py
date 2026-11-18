"""Training-fitted coordinates for fiber and the three-species gut panel."""

import numpy as np
from adapters.fermentation_score import MASS_UNIT, resolve_species
from v8_models import background_and_features, centered
import v10_grid as grid


def scaled(pair):
    low, center, high = np.quantile(pair[0], [0.25, 0.5, 0.75])
    width = high - low
    if width < 1e-10:
        width = np.std(pair[0])
    if width < 1e-10:
        return tuple(np.zeros_like(x, dtype=float) for x in pair)
    return tuple((x - center) / width for x in pair)


def right_hinge(pair, percentile):
    knot = np.percentile(pair[0], percentile)
    return centered(*(np.maximum(x - knot, 0) for x in pair))


def soft_hinge(z, percentile, width):
    knot = np.percentile(z[0], percentile)
    return centered(*(width * np.logaddexp(0.0, (x - knot) / width) for x in z))


def gauss(z, percentile):
    center = np.percentile(z[0], percentile)
    return centered(*(np.exp(-0.5 * ((x - center) / 0.55) ** 2) for x in z))


def rc_basis(z, percentiles):
    """Natural cubic basis with linear tails, using interpolation past the final knot."""
    knots = np.percentile(z[0], percentiles)
    upper, last = knots[-2:]
    # Fixed dimensional output for ties; a collapsed knot carries no spurious degree of freedom.
    columns = []
    for knot in knots[:-2]:
        if min(upper - knot, last - upper) <= 1e-9:
            columns.append(tuple(np.zeros_like(x) for x in z))
            continue

        def values(x):
            # Evaluate the cubic difference only inside the boundary. Beyond it the
            # natural spline is a line with known slope, avoiding cubic cancellation.
            t = np.minimum(x, last)
            coef = (last - knot) / (last - upper)
            within = np.maximum(t - knot, 0) ** 3 - coef * np.maximum(t - upper, 0) ** 3
            slope = 3 * (last - knot) * (upper - knot)
            return (within + slope * np.maximum(x - last, 0)) / (last - knots[0]) ** 2

        columns.append(centered(*(values(x) for x in z)))
    return columns


def orthogonal_to_linear(pair, linear):
    design = [np.column_stack((np.ones(len(x)), x)) for x in linear]
    coef = np.linalg.lstsq(design[0], pair[0], rcond=None)[0]
    return tuple(v - x @ coef for v, x in zip(pair, design, strict=True))


def add_scores(bank):
    f = bank[grid.F]
    z = scaled(f)
    for q in grid.KNOTS:
        if q != 50:
            bank[grid.hinge(q)] = right_hinge(f, q)
    for q in (35, 50, 65):
        for w in (10, 25, 50):
            bank[f"f_soft{q}_{w}"] = soft_hinge(z, q, w / 100)
    for q in (25, 50, 75):
        bank[f"f_rbf{q}"] = gauss(z, q)
    bank["f_sat"] = centered(*(np.tanh(x) for x in z))
    ordered = np.sort(f[0])

    def midrank(x):
        return (np.searchsorted(ordered, x, "left") + np.searchsorted(ordered, x, "right")) / (
            2 * len(ordered)
        )

    bank["f_rank"] = centered(*(midrank(x) for x in f))
    caps = np.quantile(f[0], [0.05, 0.95])
    bank["f_winsor"] = centered(*(np.clip(x, *caps) for x in f))
    for name in ("f_rank", "f_winsor"):
        bank[name + "_h"] = right_hinge(bank[name], 50)
    for k, qs in ((3, [10, 50, 90]), (4, [10, 35, 65, 90])):
        for j, pair in enumerate(rc_basis(z, qs)):
            bank[f"f_rc{k}_{j}"] = pair


def add_doses(bank, original):
    linear, curve = zip(*original, strict=True)
    z = scaled(linear)
    extra = {f"d_h{q}": right_hinge(linear, q) for q in (25, 35, 50, 65, 75)}
    extra["d_soft50"] = soft_hinge(z, 50, 0.25)
    for q in (30, 70):
        extra[f"d_rbf{q}"] = gauss(z, q)
    for k, qs in ((3, [10, 50, 90]), (4, [10, 35, 65, 90])):
        for j, pair in enumerate(rc_basis(z, qs)):
            extra[f"d_rc{k}_{j}"] = pair
    extra = {n: orthogonal_to_linear(pair, linear) for n, pair in extra.items()}
    bank.update(extra)
    return {grid.U: linear, grid.C: curve, **extra}


def add_composition(source, split, bank):
    match = resolve_species(source.taxa["gut"])
    indices = [
        match[n][0] for n in ("Ruminococcus bromii", "Bifidobacterium adolescentis", "Eubacterium rectale")
    ]
    parts = []
    for rows in (split.train_rows, split.rows):
        panel = np.asarray(source.raw["gut"][np.ix_(rows, indices)], dtype=float)
        if not np.isfinite(panel).all() or np.any(panel < 0):
            raise ValueError("Invalid three-species abundance")
        total = panel.sum(axis=1)
        logs = np.log1p(panel / MASS_UNIT)
        shares = np.divide(panel, total[:, None], out=np.zeros_like(panel), where=total[:, None] > 0)
        side = {
            "mean_logs": logs.mean(axis=1),
            "max_log": logs.max(axis=1),
            "root_mean": np.log1p(np.sqrt(panel / MASS_UNIT).mean(axis=1) ** 2),
            "detect_any": (total > 0).astype(float),
            "panel_dominance": shares.max(axis=1),
            "panel_evenness": -(shares * np.log(np.maximum(shares, 1e-300))).sum(axis=1) / np.log(3),
        }
        for j, s in enumerate(grid.MEMBERS):
            side[s], side["present_" + s] = logs[:, j], (panel[:, j] > 0).astype(float)
            side["omit_" + s] = np.log1p(panel[:, [i for i in range(3) if i != j]].sum(axis=1) / MASS_UNIT)
        parts.append(side)
    for name in parts[0]:
        bank[name] = centered(*(p[name] for p in parts))
    for n in (*grid.PANEL_VIEWS, *grid.MEMBERS):
        bank[n + "_h"] = right_hinge(bank[n], 50)
    for j in (1, 2):
        pc = bank[f"g{j}"]
        bank[f"f_gate_g{j}"] = centered(
            *((x - pc[0].mean()) * f for x, f in zip(pc, bank[grid.F], strict=True))
        )


def design_matrices(source, split):
    if np.intersect1d(source.groups[split.train_rows], source.groups[split.rows]).size:
        raise ValueError("A household crosses the fit and evaluation roles")
    bank, bases, old = background_and_features(source, split)
    add_scores(bank)
    add_composition(source, split, bank)
    doses = add_doses(bank, old)
    return bank, bases, doses
