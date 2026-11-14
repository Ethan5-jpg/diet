"""Select the locked fermentation correction using only discovery outcomes."""

import numpy as np

from interactions import ROUTES, ZERO, fit_corrections
from models import BASE, base_predict, with_panel
from moderation_models import MARKERS, prepare, residuals


REPEAT = 0
FOLDS = 5
ROUTE = "fermentation_smooth_curve"
CANDIDATES = (ZERO, f"{ROUTE}@0.1", f"{ROUTE}@1", f"{ROUTE}@10")


def _held_squared_errors(target, baseline, corrections):
    target, baseline = np.asarray(target, dtype=float), np.asarray(baseline, dtype=float)
    if target.ndim != 1 or not len(target) or baseline.shape != target.shape:
        raise ValueError("INVALID_DISCOVERY_TUNING_PREDICTION")
    error = target - baseline
    losses = {}
    for key in CANDIDATES:
        added = np.asarray(corrections[key][0], dtype=float)
        if added.shape != error.shape:
            raise ValueError("INVALID_DISCOVERY_TUNING_PREDICTION")
        losses[key] = float(np.sum((error - added) ** 2))
    if not all(np.isfinite(value) for value in losses.values()):
        raise ValueError("NONFINITE_DISCOVERY_TUNING_LOSS")
    return losses


def _choose(parts):
    losses = {key: float(np.sum([part[key] for part in parts])) for key in CANDIDATES}
    chosen = min(CANDIDATES, key=lambda key: round(losses[key], 9))
    return chosen, losses


def _quartiles(values):
    return {
        name: np.quantile(column, (0.25, 0.75)).tolist()
        for name, column in zip(("gut_shannon", "fermentation"), values.T, strict=True)
    }


def select(source, panel):
    """Use the original r0 household folds to tune one v4 interaction family.

    The returned SSE values are discovery selection criteria, not independent
    performance estimates. Outputs contain no participant-level information.
    """
    if ROUTES.get(ROUTE) != CANDIDATES:
        raise ValueError("V4_FERMENTATION_CANDIDATE_CONTRACT_CHANGED")
    groups = np.asarray(source.groups)
    panel = np.asarray(panel, dtype=float)
    reference = np.asarray(source.reference["y"][REPEAT], dtype=float)
    assignment = np.asarray(source.reference["outer"][REPEAT])
    if (groups.ndim != 1 or panel.shape != groups.shape or reference.shape != groups.shape
            or assignment.shape != groups.shape or not np.issubdtype(assignment.dtype, np.integer)
            or set(assignment) != set(range(FOLDS)) or np.isinf(reference).any()):
        raise ValueError("INVALID_DISCOVERY_TUNING_ALIGNMENT")
    eligible = np.isfinite(reference)
    expected = np.flatnonzero(eligible)
    if not len(expected) or not np.isfinite(panel[eligible]).all():
        raise ValueError("INCOMPLETE_DISCOVERY_TUNING_PANEL")

    source.verify_sources()
    parts, held_rows, held_target, held_markers = [], [], [], []
    for fold in range(FOLDS):
        split = source.outer_split(REPEAT, fold)
        valid = assignment == fold
        role = with_panel(prepare(source, split, shared_host=True), split, panel)
        expected_fold = np.flatnonzero(valid & eligible)
        if (not np.array_equal(role.rows, expected_fold)
                or not np.array_equal(role.y[1], reference[expected_fold])
                or np.any(valid[split.train_rows])
                or np.intersect1d(groups[split.train_rows], groups[valid]).size):
            raise ValueError("DISCOVERY_TUNING_FOLD_CHANGED")
        baseline = base_predict(role)
        corrections = fit_corrections(role, residuals(role, BASE), CANDIDATES)
        parts.append(_held_squared_errors(role.y[1], baseline, corrections))
        held_rows.append(role.rows)
        held_target.append(role.y[1])
        held_markers.append(np.column_stack((
            role.raw_eco[1][:, MARKERS.index("gut_shannon")], panel[role.rows],
        )))

    rows, target, markers = map(np.concatenate, (held_rows, held_target, held_markers))
    order = np.argsort(rows)
    if (not np.array_equal(rows[order], expected)
            or not np.array_equal(target[order], reference[expected])
            or markers.shape != (len(expected), 2) or not np.isfinite(markers).all()):
        raise ValueError("POOLED_DISCOVERY_TUNING_ROWS_OR_TARGETS_CHANGED")
    source.verify_sources()
    chosen, losses = _choose(parts)
    return {
        "repeat": REPEAT,
        "route": ROUTE,
        "candidate": chosen,
        "people": len(expected),
        "households": int(np.unique(groups[expected]).size),
        "folds": FOLDS,
        "baseline_sse": losses[ZERO],
        "candidate_sse": losses,
        "marker_quartiles": _quartiles(markers[order]),
    }


def selector_selftest():
    """Test exact pooled SSE, ties in frozen order and paired quartiles."""
    target = np.array([2.0, -1.0, 5.0, 4.0])
    baseline = np.array([1.0, 0.0, 3.0, 2.0])
    exact = target - baseline
    added = (np.zeros(len(exact)), exact * 0.5, exact, -exact)
    corrections = {key: (values, np.zeros_like(values))
                   for key, values in zip(CANDIDATES, added, strict=True)}
    parts = [
        _held_squared_errors(target[start:stop], baseline[start:stop],
                             {key: (pair[0][start:stop], None) for key, pair in corrections.items()})
        for start, stop in ((0, 1), (1, 4))
    ]
    winner, losses = _choose(parts)
    assert winner == CANDIDATES[2] and losses[winner] == 0.0
    assert losses[ZERO] == float(np.sum(exact ** 2)) and losses[CANDIDATES[-1]] == 4 * losses[ZERO]
    tie = dict.fromkeys(CANDIDATES, 12.0)
    tie[CANDIDATES[1]] -= 1e-10
    assert _choose([tie])[0] == ZERO
    tie[CANDIDATES[2]] -= 1e-8
    assert _choose([tie])[0] == CANDIDATES[2]
    assert _quartiles(np.column_stack((target, baseline))) == {
        "gut_shannon": [1.25, 4.25], "fermentation": [0.75, 2.25],
    }
