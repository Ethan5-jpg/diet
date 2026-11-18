"""Two household repartitions inside each outer training role for adaptive comparators."""

import numpy as np
from v7_eval import _check_roles, FLOOR
from v8_selection import second_inner
from v10_fit import evaluate
import v10_grid as grid

BASE = np.asarray(grid.OFF_INDEX)
REF = grid.INDEX[grid.REFERENCE]


# Allow strong and modest interaction signals; a guarded recipe must retain comparable main-only R².
def validation_ids(parts):
    ids = np.concatenate([np.asarray(x) for x in parts])
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Repeated or invalid inner validation row")
    return ids


def inner_forecasts(source, splits):
    for split in splits:
        off, on = evaluate(source, split)
        yield split.rows, split.yv, off, on


def training_scores(pieces, common):
    # Pair the populations *between* inner partitions as well as between arms.
    # Each inner fold retains its own training-fitted transformation of the target.
    if len(common) < 2 or not np.array_equal(common, np.unique(common)):
        raise ValueError("Insufficient or invalid common inner membership")
    rows, targets = [], []
    off_sse = np.zeros(len(grid.FAMILIES))
    on_sse = np.zeros(len(grid.KEYS))
    for ids, y, off, on in pieces:
        keep = np.isin(ids, common)
        used = y[keep]
        targets.append(used)
        rows.append(ids[keep])
        off_sse += np.sum((used[:, None] - off[keep]) ** 2, axis=0)
        on_sse += np.sum((used[:, None] - on[keep]) ** 2, axis=0)
    if not np.array_equal(np.sort(validation_ids(rows)), common):
        raise ValueError("Common inner membership not scored exactly once")
    y = np.concatenate(targets)
    sst = np.sum((y - y.mean()) ** 2)
    if not np.isfinite(sst) or sst <= 0:
        raise ValueError("Insufficient within-role outcome variance")
    base, full = 1 - off_sse[BASE] / sst, 1 - on_sse / sst
    ratio = np.divide(
        100 * (full - base), full, out=np.full_like(full, -np.inf), where=(base > FLOOR) & (full > FLOOR)
    )
    return ratio, base, len(y)


def choose(first, second):
    """Ties use the frozen grid order. A null inner result turns off the anchor."""
    score = np.minimum(first[0], second[0])
    guarded = np.logical_and.reduce(
        [x[1] >= x[1][REF] - grid.SELECTION_OFF_TOLERANCE for x in (first, second)]
    )
    order = np.argsort(-score, kind="stable")
    positive = [int(i) for i in order if score[i] > 1e-10 and np.isfinite(score[i])]
    passing = [i for i in positive if guarded[i]]
    diverse, groups = [], set()
    for i in passing:
        group = grid.GROUPS[grid.FAMILIES[BASE[i]]]
        if group not in groups:
            diverse.append(i)
            groups.add(group)
        if len(diverse) == 3:
            break
    return {
        n: {
            "keys": [grid.KEYS[i] for i in indices] if indices else [grid.REFERENCE],
            "enabled": bool(indices),
        }
        for n, indices in zip(grid.ADAPTIVE, (positive[:1], passing[:1], diverse), strict=True)
    }


def adaptive_pairs(off, on, choices):
    if set(choices) != set(grid.ADAPTIVE):
        raise ValueError("Incomplete adaptive choices")
    a, b = [], []
    for n in grid.ADAPTIVE:
        info = choices[n]
        keys = info["keys"]
        if (
            type(info["enabled"]) is not bool
            or not keys
            or len(set(keys)) != len(keys)
            or len(keys) > (3 if n == "inner_diverse_blend" else 1)
            or any(k not in grid.INDEX for k in keys)
            or not info["enabled"]
            and keys != [grid.REFERENCE]
        ):
            raise ValueError("Invalid frozen inner choice")
        i = [grid.INDEX[k] for k in keys]
        base = off[:, BASE[i]].mean(axis=1)
        a.append(base)
        b.append(on[:, i].mean(axis=1) if info["enabled"] else base)
    return np.column_stack(a), np.column_stack(b)


def fold_predictions(source, repeat, fold):
    groups, assignment = np.asarray(source.groups), source.reference["outer"][repeat]
    original, outer = source.roles(repeat, fold)
    _check_roles(groups, assignment, fold, original, outer)
    partitions = {frozenset(groups[s.rows]) for s in original}
    first_rows = validation_ids([s.rows for s in original])
    first_forecasts = list(inner_forecasts(source, original))
    del original
    alternate = second_inner(source, repeat, fold)
    _check_roles(groups, assignment, fold, alternate, outer)
    if partitions == {frozenset(groups[s.rows]) for s in alternate}:
        raise ValueError("Both inner partitions are identical")
    second_rows = validation_ids([s.rows for s in alternate])
    common = np.intersect1d(first_rows, second_rows, assume_unique=True)
    first = training_scores(first_forecasts, common)
    del first_forecasts
    second = training_scores(inner_forecasts(source, alternate), common)
    del alternate
    choices = choose(first, second)
    off, on = evaluate(source, outer, check_off=True)
    ao, an = adaptive_pairs(off, on, choices)
    arrays = dict(
        rows=outer.rows,
        groups=groups[outer.rows].astype(str),
        y=outer.yv,
        setting_off=off,
        setting_on=on,
        adaptive_off=ao,
        adaptive_on=an,
    )
    return arrays, dict(
        repeat=repeat,
        fold=fold,
        people=len(outer.rows),
        choices=choices,
        paired_column_sha256=grid.COLUMN_SHA,
        additive_models_checked=len(grid.FAMILIES),
        inner_people=[len(first_rows), len(second_rows)],
        inner_common_people=len(common),
    )
