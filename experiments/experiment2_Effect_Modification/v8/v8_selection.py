"""Two training-only household repartitions for selecting a reproducible gain."""
from dataclasses import dataclass

import numpy as np

import v7_eval
import v7_models
from v8_models import CORE, FAMILIES, KEYS, MODELS, PENALTIES, evaluate_split


REFERENCE = "v7_ecology_replay"
PRIMARY = "nested_maximin"
GUARDED = "nested_maximin_guarded"
ENSEMBLE = "nested_top3_blend"
NAMES = (*FAMILIES, REFERENCE, PRIMARY, GUARDED, ENSEMBLE)
SECOND_INNER_SEED = 810271
KEY_INDEX = {key: i for i, key in enumerate(KEYS)}
FAMILY_INDICES = {name: [KEY_INDEX[v7_models.setting_key(name, p)] for p in PENALTIES]
                  for name in FAMILIES}
LEGACY_INDICES = [KEY_INDEX[v7_models.setting_key(CORE, p)] for p in v7_models.PENALTIES]


@dataclass(frozen=True)
class Choice:
    indices: tuple
    enabled: bool

    def describe(self):
        return dict(settings=[KEYS[i] for i in self.indices], enabled=self.enabled)


def second_inner(source, repeat, fold):
    """A second, reproducible 3-fold split of these outer training households only."""
    training = source.reference["outer"][repeat] != fold
    unique, inv = np.unique(source.groups[training], return_inverse=True)
    order = np.random.default_rng(SECOND_INNER_SEED + 100 * repeat + fold).permutation(len(unique))
    per_group = np.empty(len(unique), dtype=int)
    per_group[order] = np.arange(len(unique)) % 3
    assignment = np.full(len(training), -1, dtype=int)
    assignment[training] = per_group[inv]
    return tuple(source.split(training & (assignment != j), training & (assignment == j)) for j in range(3))


def inner_predictions(source, splits):
    pieces = []
    for split in splits:
        off, on = v7_eval._pack_predictions(evaluate_split(source, split), KEYS, len(split.rows))
        pieces.append((split.rows, split.yv, source.groups[split.rows], off, on))
    rows, y, groups, off, on = (np.concatenate(col) for col in zip(*pieces, strict=True))
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Repeated inner validation row")
    return rows, (y, groups, off, on)


def inner_stats(y, groups, off, on):
    """Point ratios and paired household-based eligibility for the guarded sensitivity."""
    stats = v7_eval._matrices(y, groups, off, on)
    k = off.shape[1]
    sst, b, f, delta = v7_eval._score_components(stats, k)
    ratio = np.divide(100 * delta, f, out=np.full_like(f, -np.inf),
                      where=(f > v7_eval.FLOOR) & (b > v7_eval.FLOOR))
    off_if = (stats[:, 2, None] * (1 - b) - stats[:, 3:3+k]) / sst
    on_if = (stats[:, 2, None] * (1 - f) - stats[:, 3+k:]) / sst
    anchor = LEGACY_INDICES[0]  # All penalties share one main-only fit.
    def se(influences):
        return np.sqrt(len(stats) / (len(stats) - 1) * np.sum(influences ** 2, axis=0))
    guarded = ((b - 1.96 * se(off_if) > v7_eval.FLOOR) & (f - 1.96 * se(on_if) > v7_eval.FLOOR)
               & (b >= b[anchor] - se(off_if - off_if[:, [anchor]]) - 1e-12))
    return ratio, guarded


def best_choice(score, indices, fallback):
    best = 0.
    selected = Choice((fallback,), False)
    for index in indices:
        if score[index] > best + v7_eval.TIE_PCT:
            best = score[index]
            selected = Choice((index,), True)
    return selected


def select(first, second):
    """Use the smaller of the two inner OOF ratios for each fixed on/off setting."""
    rows0, parts0 = first
    rows1, parts1 = second
    # The original V7 family used its full first inner split and five penalties.
    ratio0, guard0 = inner_stats(*parts0)
    legacy = best_choice(ratio0, LEGACY_INDICES, LEGACY_INDICES[0])
    common, at0, at1 = np.intersect1d(rows0, rows1, assume_unique=True, return_indices=True)
    if not len(common) or not np.array_equal(parts0[1][at0], parts1[1][at1]):
        raise ValueError("Inner repeat common households or row identity differ")
    if len(common) != len(rows0):
        ratio0, guard0 = inner_stats(*(a[at0] for a in parts0))
    aligned1 = parts1 if len(common) == len(rows1) else tuple(a[at1] for a in parts1)
    ratio1, guard1 = inner_stats(*aligned1)
    robust = np.minimum(ratio0, ratio1)
    choices = {family: best_choice(robust, indices, indices[0]) for family, indices in FAMILY_INDICES.items()}
    choices[REFERENCE] = legacy
    choices[PRIMARY] = best_choice(robust, range(len(KEYS)), LEGACY_INDICES[0])
    choices[GUARDED] = best_choice(np.where(guard0 & guard1, robust, -np.inf), range(len(KEYS)),
                                   LEGACY_INDICES[0])
    # One setting per single model. Fixed blends cannot vote again for their parents.
    winners = [choices[model.name].indices[0] for model in MODELS if choices[model.name].enabled]
    winners.sort(key=lambda index: -robust[index])
    choices[ENSEMBLE] = Choice(tuple(winners[:3]), True) if winners else Choice((LEGACY_INDICES[0],), False)
    audit = dict(first_inner_people=len(rows0), second_inner_people=len(rows1), common_inner_people=len(common),
                 eligible_positive_in_both=int(np.sum(robust > v7_eval.TIE_PCT)),
                 eligible_guarded_in_both=int(np.sum((robust > v7_eval.TIE_PCT) & guard0 & guard1)))
    return choices, audit


def fold_predictions(source, repeat, fold):
    groups = np.asarray(source.groups)
    assignment = source.reference["outer"][repeat]
    original, outer = source.roles(repeat, fold)
    v7_eval._check_roles(groups, assignment, fold, original, outer)
    original_groups = {frozenset(groups[split.rows]) for split in original}
    first = inner_predictions(source, original)
    del original
    other = second_inner(source, repeat, fold)
    v7_eval._check_roles(groups, assignment, fold, other, outer)
    if original_groups == {frozenset(groups[split.rows]) for split in other}:
        raise ValueError("Both inner household partitions are identical")
    second = inner_predictions(source, other)
    del other
    choices, audit = select(first, second)
    del first, second
    off, on = v7_eval._pack_predictions(evaluate_split(source, outer), KEYS, len(outer.rows))
    null_predictions, full_predictions = [], []
    for name in NAMES:
        choice = choices[name]
        idx = list(choice.indices)
        null_prediction = off[:, idx].mean(axis=1)
        null_predictions.append(null_prediction)
        full_predictions.append(on[:, idx].mean(axis=1) if choice.enabled else null_prediction)
    arrays = dict(rows=outer.rows, groups=groups[outer.rows].astype(str), y=outer.yv,
                  off=np.column_stack(null_predictions), on=np.column_stack(full_predictions))
    meta = dict(repeat=repeat, fold=fold, people=len(outer.rows), names=list(NAMES),
                choices={name: c.describe() for name, c in choices.items()}, inner=audit)
    return arrays, meta
