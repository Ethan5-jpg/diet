"""Disentangle three interaction blocks at one penalty and one identical additive fit."""

from itertools import permutations

import numpy as np

import v8_scores as scores
import v9_cv as cv
import v9_grid as grid


def order_average(cells, block_count):
    """Average the added R² over every order in which the blocks could be added."""
    if cells.shape[-1] != 2**block_count:
        raise ValueError("Incomplete paired factorial")
    orders = tuple(permutations(range(block_count)))
    weights = np.zeros((cells.shape[-1], block_count))
    for order in orders:
        mask = 0
        for block in order:
            weights[mask, block] -= 1.0 / len(orders)
            mask |= 1 << block
            weights[mask, block] += 1.0 / len(orders)
    values = (cells - cells[..., :1]) @ weights
    if not np.allclose(values.sum(axis=-1), cells[..., -1] - cells[..., 0], atol=2e-10, rtol=1e-10):
        raise ArithmeticError("Contributions do not sum to the full-minus-core prediction gain")
    return values


def summarize(panels, draws, seed):
    keys = [grid.v7.setting_key(n, grid.MATCHED_PENALTY) for n in grid.MATCHED_CELLS]
    on_indices = [cv.INDEX[key] for key in keys]
    off_indices = cv.OFF_INDEX[on_indices]
    paired = []
    for p in panels:
        off = p["setting_off"][:, off_indices]
        if not np.all(off == off[:, :1]):
            raise ValueError("Shared-baseline ablation has nonidentical off predictions")
        paired.append(
            dict(rows=p["rows"], groups=p["groups"], y=p["y"], off=off, on=p["setting_on"][:, on_indices])
        )
    point, boot = scores.paired_bootstrap(paired, draws, seed)
    reliable = len(boot) >= np.ceil(0.95 * draws)
    # Units throughout this decomposition are percentage points of R².
    gain = 100.0 * point[:, 2, :]
    sampled_gain = 100.0 * boot[:, :, 2, :]
    block_count = len(grid.MATCHED_BLOCKS)
    contribution = order_average(gain, block_count)
    sampled_contribution = order_average(sampled_gain, block_count)
    radius = None
    if reliable:
        # One two-sided band for these three fixed contributions in both repeats.
        max_error = np.max(np.abs(sampled_contribution - contribution), axis=(1, 2))
        radius = float(np.quantile(max_error, 0.95))
    full_mask = len(keys) - 1
    repeats = []
    for r in (0, 1):
        blocks = {}
        for j, name in enumerate(grid.MATCHED_BLOCKS):
            without = full_mask ^ (1 << j)
            estimate = float(contribution[r, j])
            blocks[name] = dict(
                order_average_r2_pp=estimate,
                order_average_simultaneous_ci95_pp=(
                    None if radius is None else [estimate - radius, estimate + radius]
                ),
                full_minus_without_r2_pp=float(gain[r, -1] - gain[r, without]),
                full_minus_without_paired_ci95_pp=(
                    scores.interval(sampled_gain[:, r, -1] - sampled_gain[:, r, without])
                    if reliable
                    else None
                ),
            )
        repeats.append(
            dict(
                repeat=r,
                shared_off_r2=float(point[r, 0, 0]),
                core_on_r2=float(point[r, 1, 0]),
                full_on_r2=float(point[r, 1, -1]),
                core_delta_r2_pp=float(gain[r, 0]),
                full_delta_r2_pp=float(gain[r, -1]),
                full_minus_core_r2_pp=float(gain[r, -1] - gain[r, 0]),
                full_minus_core_paired_ci95_pp=(
                    scores.interval(sampled_gain[:, r, -1] - sampled_gain[:, r, 0]) if reliable else None
                ),
                blocks=blocks,
            )
        )
    return dict(
        fixed_penalty=grid.MATCHED_PENALTY,
        settings_by_mask=keys,
        block_order=grid.MATCHED_BLOCKS,
        off_equal_all_rows=True,
        valid_household_draws=len(boot),
        repeats=repeats,
    )
