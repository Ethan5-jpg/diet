"""Nested household comparison of fixed structures and predeclared selection rules."""

from dataclasses import dataclass

import numpy as np

import v7_eval as old
import v7_models
import v8_selection
import v9_fit
import v9_grid as grid


PRIMARY = "stack_from_pc4"
WORKFLOWS = (PRIMARY, "stack_unrestricted", "selector_maximin", "selector_guarded", "blend_diverse_top3")
FIXED = "historical_pc10_fixed_0p1"
NAMES = (*grid.FAMILIES, FIXED, *WORKFLOWS)
INDEX = {k: i for i, k in enumerate(grid.KEYS)}
INDICES = {n: tuple(INDEX[v7_models.setting_key(n, p)] for p in grid.SETTINGS[n]) for n in grid.FAMILIES}
OFF_INDEX = np.repeat(np.arange(len(grid.FAMILIES)), [len(grid.SETTINGS[n]) for n in grid.FAMILIES])
ANCHOR_I = INDICES[grid.ANCHOR][0]
MIX_WEIGHTS = (0.25, 0.5, 0.75)
POOL_PER_GROUP = 3
MAX_COMPONENTS = 4


@dataclass(frozen=True)
class Choice:
    indices: tuple
    weights: tuple
    active: tuple

    def predict(self, off, on, *, off_index=None):
        idx = list(self.indices)
        weights = np.asarray(self.weights)
        x = off[:, idx if off_index is None else off_index[idx]]
        z = np.where(self.active, on[:, idx], x)
        return x @ weights, z @ weights

    def describe(self):
        return dict(
            settings=[grid.KEYS[i] for i in self.indices],
            weights=list(self.weights),
            active=list(self.active),
            enabled=any(self.active),
        )


def single(i, enabled):
    return Choice((i,), (1.0,), (enabled,))


def common_family_off(off):
    """Lossless compression is valid only if every setting has its family's off prediction."""
    representatives = off[:, [INDICES[n][0] for n in grid.FAMILIES]]
    for j, name in enumerate(grid.FAMILIES):
        if np.any(off[:, list(INDICES[name])] != representatives[:, j, None]):
            raise ValueError("Off predictions differ within model family " + name)
    return representatives


def verify_fold_choices(arrays, metadata):
    """Replay both saved arms from fixed settings, including disabled constituents."""
    for j, name in enumerate(NAMES):
        c = metadata["choices"][name]
        indices = tuple(INDEX[key] for key in c["settings"])
        weights, active = tuple(c["weights"]), tuple(c["active"])
        if (
            not indices
            or len(set(indices)) != len(indices)
            or len(indices) != len(weights)
            or len(indices) != len(active)
            or not all(type(v) is bool for v in active)
            or any(not np.isfinite(v) or v <= 0 for v in weights)
            or not np.isclose(sum(weights), 1.0, atol=1e-12, rtol=0)
            or c["enabled"] != any(active)
        ):
            raise ValueError("Invalid stored choice for " + name)
        arm_pair = Choice(indices, weights, active).predict(
            arrays["setting_off"], arrays["setting_on"], off_index=OFF_INDEX
        )
        for arm, prediction in zip(("off", "on"), arm_pair, strict=True):
            if not np.allclose(arrays[arm][:, j], prediction, atol=1e-10, rtol=0):
                raise ValueError("Stored named prediction differs from fixed settings: " + name + "/" + arm)


def best(values, candidates, fallback):
    winning, top = single(fallback, False), 0.0
    for i in candidates:
        if values[i] > top + old.TIE_PCT:
            top, winning = values[i], single(i, True)
    return winning


def inner_panel(source, splits):
    pieces = []
    for s in splits:
        a, b = old._pack_predictions(v9_fit.evaluate_split(source, s), grid.KEYS, len(s.rows))
        pieces.append((s.rows, s.yv, source.groups[s.rows], a, b))
    rows, y, groups, off, on = (np.concatenate(col) for col in zip(*pieces, strict=True))
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Duplicated inner validation row")
    return rows, (y, groups, off, on)


def stats(panel):
    # Limit temporary participant x setting matrices. Column zero is the common
    # reference off prediction used solely in the guarded sensitivity rule.
    y, group, off, on = panel
    ratios, guards = [], []
    for start in range(0, off.shape[1], 96):
        idx = [ANCHOR_I, *range(start, min(start + 96, off.shape[1]))]
        if v8_selection.LEGACY_INDICES[0] != 0:
            raise ValueError("Archived comparator index changed")
        r, g = v8_selection.inner_stats(y, group, off[:, idx], on[:, idx])
        ratios.extend(r[1:])
        guards.extend(g[1:])
    return np.array(ratios), np.array(guards)


def group_name(n):
    if n in grid.RECIPES:
        return grid.RECIPES[n].group
    if n in grid.ALTERNATIVES:
        return grid.ALTERNATIVES[n][0]
    return "blend"


def min_ratio(panels, predictions, ssts):
    values = []
    for (y, _, _, _), (off, on), sst in zip(panels, predictions, ssts, strict=True):
        e0, e1 = y - off, y - on
        b, f = 1.0 - (e0 @ e0) / sst, 1.0 - (e1 @ e1) / sst
        if min(b, f) <= old.FLOOR:
            return -np.inf
        values.append(100.0 * (f - b) / f)
    return min(values)


def greedy(panels, pool, start):
    """Train weights on paired inner out-of-fold predictions in both partitions."""
    current = start

    def predict(which):
        return [which.predict(off, on) for _, _, off, on in panels]

    combined = predict(current)
    predictions = [(c, predict(c)) for c in pool]
    ssts = [np.sum((p[0] - np.mean(p[0])) ** 2) for p in panels]
    current_score = min_ratio(panels, combined, ssts)
    while len(current.indices) < MAX_COMPONENTS:
        top, pick = current_score, None
        for candidate, pairs in predictions:
            if candidate.indices[0] in current.indices:
                continue
            for w in MIX_WEIGHTS:
                merged = [
                    tuple((1 - w) * a + w * b for a, b in zip(left, right, strict=True))
                    for left, right in zip(combined, pairs, strict=True)
                ]
                value = min_ratio(panels, merged, ssts)
                if value > top + old.TIE_PCT:
                    top, pick = value, (candidate, w, merged)
        if pick is None:
            break
        candidate, w, combined = pick
        current = Choice(
            current.indices + candidate.indices,
            tuple(v * (1 - w) for v in current.weights) + (w,),
            current.active + candidate.active,
        )
        current_score = top
    return current


def select(first, second):
    common, a, b = np.intersect1d(first[0], second[0], assume_unique=True, return_indices=True)
    if not len(common) or not np.array_equal(first[1][1][a], second[1][1][b]):
        raise ValueError("Training repeat identity mismatch")
    panels = [tuple(x[t] for x in p[1]) for p, t in ((first, a), (second, b))]
    r0, g0 = stats(panels[0])
    r1, g1 = stats(panels[1])
    values = np.minimum(r0, r1)
    choices = {n: best(values, inds, inds[0]) for n, inds in INDICES.items()}
    choices[FIXED] = single(INDEX[grid.FIXED_PC10], True)
    choices["selector_maximin"] = best(values, range(len(values)), ANCHOR_I)
    choices["selector_guarded"] = best(np.where(g0 & g1, values, -np.inf), range(len(values)), ANCHOR_I)
    # One nomination per individual family; omit averages whose parents already compete.
    groups = {}
    for name in (*grid.RECIPES, *grid.ALTERNATIVES):
        c = choices[name]
        if any(c.active):
            groups.setdefault(group_name(name), []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: -values[c.indices[0]])
    top = sorted((v[0] for v in groups.values()), key=lambda c: -values[c.indices[0]])[:3]
    choices["blend_diverse_top3"] = (
        Choice(tuple(c.indices[0] for c in top), (1 / len(top),) * len(top), (True,) * len(top))
        if top
        else single(ANCHOR_I, False)
    )
    pool = [c for group in groups.values() for c in group[:POOL_PER_GROUP]]
    choices[PRIMARY] = greedy(panels, pool, choices[grid.ANCHOR])
    choices["stack_unrestricted"] = greedy(panels, pool, choices["selector_maximin"])
    return choices, dict(
        first_rows=len(first[0]),
        second_rows=len(second[0]),
        common_rows=len(common),
        positive_in_both=int(np.sum(values > old.TIE_PCT)),
        stack_pool=len(pool),
    )


def fold_predictions(source, repeat, fold):
    groups, assignment = np.asarray(source.groups), source.reference["outer"][repeat]
    original, outer = source.roles(repeat, fold)
    old._check_roles(groups, assignment, fold, original, outer)
    roles = {frozenset(groups[s.rows]) for s in original}
    first = inner_panel(source, original)
    del original
    alternate = v8_selection.second_inner(source, repeat, fold)
    old._check_roles(groups, assignment, fold, alternate, outer)
    if roles == {frozenset(groups[s.rows]) for s in alternate}:
        raise ValueError("Identical inner household repartitions")
    second = inner_panel(source, alternate)
    del alternate
    choices, audit = select(first, second)
    del first, second
    off, on = old._pack_predictions(v9_fit.evaluate_split(source, outer), grid.KEYS, len(outer.rows))
    pairs = [choices[n].predict(off, on) for n in NAMES]
    arrays = dict(
        rows=outer.rows,
        groups=groups[outer.rows].astype(str),
        y=outer.yv,
        off=np.column_stack([p[0] for p in pairs]),
        on=np.column_stack([p[1] for p in pairs]),
        setting_off=common_family_off(off),
        setting_on=on,
    )
    meta = dict(
        repeat=repeat,
        fold=fold,
        people=len(outer.rows),
        names=list(NAMES),
        choices={n: choices[n].describe() for n in NAMES},
        inner=audit,
    )
    return arrays, meta
