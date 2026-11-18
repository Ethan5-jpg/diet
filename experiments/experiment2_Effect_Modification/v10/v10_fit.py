"""Fit all registered pairs from one training-only profile, retaining an off fit per family."""

import numpy as np
from v7_models import _Profile, MAIN_PENALTY
from v9_fit import InteractionBlock
import v10_grid as grid
from v10_features import design_matrices

MAIN = tuple(dict.fromkeys(m for r in grid.FITS.values() for m in r.mains))
TERMS = tuple(dict.fromkeys(t for r in grid.FITS.values() for t in r.terms))
MI = {m: i for i, m in enumerate(MAIN)}
TI = {t: len(MAIN) + i for i, t in enumerate(TERMS)}


def audit_matched_off(off, main_only_predict):
    """Solve each distinct registered additive model without passing a single product.

    Reconstruct the defined off arm of every estimator, shrinkage and blend by name.
    This runs on each outer fold; no validation outcomes are required.
    """
    cache, expected = {}, {}
    for n, recipe in grid.FITS.items():
        if recipe.mains not in cache:
            cache[recipe.mains] = main_only_predict(recipe.mains)
        expected[n] = cache[recipe.mains]
    for n, (_, parent) in grid.ALT.items():
        expected[n] = expected[parent]
    for n, (parent, _) in grid.FRACTIONS.items():
        expected[n] = expected[parent]
    for n, parents in grid.BLENDS.items():
        expected[n] = sum(w * expected[p] for p, w in parents)
    for n in grid.FAMILIES:
        if not np.allclose(off[:, grid.FAMILY_INDEX[n]], expected[n], rtol=0.0, atol=2e-8):
            raise ValueError("Off prediction differs from the main-only fit for " + n)


def evaluate(source, split, *, check_off=False):
    bank, bases, doses = design_matrices(source, split)
    columns = [
        np.column_stack([*(bank[m][s] for m in MAIN), *(doses[d][s] * bank[m][s] for d, m in TERMS)])
        for s in (0, 1)
    ]
    prof = _Profile(
        *bases,
        np.column_stack((split.ot, split.gt)),
        np.column_stack((split.ov, split.gv)),
        split.yt,
        *columns,
    )
    n = len(split.rows)
    off = np.full((n, len(grid.FAMILIES)), np.nan)
    on = np.full((n, len(grid.KEYS)), np.nan)
    written = set()

    def record(name, base, values):
        indices = [grid.INDEX[grid.key(name, p)] for p in grid.SETTINGS[name]]
        if name in written or np.shape(base) != (n,) or np.shape(values) != (n, len(indices)):
            raise ArithmeticError("Unexpected prediction dimensions for " + name)
        written.add(name)
        off[:, grid.FAMILY_INDEX[name]] = base
        on[:, indices] = values

    for name, recipe in grid.FITS.items():
        block = InteractionBlock(
            prof, [MI[m] for m in recipe.mains], [TI[t] for t in recipe.terms], recipe.weights
        )
        record(name, block.off, block.ridge(grid.SETTINGS[name]))
        for method in grid.ALT_OPTIONS:
            other = method + "__" + name
            if other in grid.ALT:
                record(other, block.off, getattr(block, method)(grid.SETTINGS[other]))

    def parent_on(name, options):
        return on[:, [grid.INDEX[grid.key(name, p)] for p in options]]

    for name, (parent, weight) in grid.FRACTIONS.items():
        base = off[:, grid.FAMILY_INDEX[parent]]
        record(
            name, base, base[:, None] + weight * (parent_on(parent, grid.SETTINGS[name]) - base[:, None])
        )
    for name, parents in grid.BLENDS.items():
        base = sum(w * off[:, grid.FAMILY_INDEX[p]] for p, w in parents)
        enhanced = sum(w * parent_on(p, grid.SETTINGS[name]) for p, w in parents)
        record(name, base, enhanced)
    if written != set(grid.FAMILIES) or not np.isfinite(off).all() or not np.isfinite(on).all():
        raise ArithmeticError("Incomplete or nonfinite paired predictions")
    if check_off:
        audit_matched_off(off, lambda mains: prof.prediction([MI[m] for m in mains], MAIN_PENALTY))
    return off, on
