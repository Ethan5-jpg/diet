"""Training-only microbial compositions and a finite two-taxon balance bank.

These fixed-size views are inspired by log-ratio/balance methods such as selbal;
they do not implement selbal's supervised greedy balance search. All returned
microbial views must also enter the caller's matched main-effect baseline.
There is deliberately no outcome argument. The fixed floor is in relative
abundance fraction units, rather than a sequencing-count pseudocount.
"""

from numbers import Integral

import numpy as np


PREVALENCE_THRESHOLD = 0.1
RELATIVE_ABUNDANCE_FLOOR = 1e-6
CONSTANT_TOLERANCE = 1e-12


def _raw(values, label, columns=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2
            or (columns is not None and values.shape[1] != columns)):
        raise ValueError("INVALID_" + label + "_SHAPE")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("INVALID_" + label + "_ABUNDANCE")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-3):
        raise ValueError("EXPECTED_" + label + "_RELATIVE_ABUNDANCE_FRACTIONS")
    return values


def _standardize(train, valid):
    """Replay one training center/scale, retaining constant columns explicitly."""
    center, scale = train.mean(axis=0), train.std(axis=0)
    constant = scale <= CONSTANT_TOLERANCE
    safe_scale = np.where(constant, 1.0, scale)
    return ((train - center) / safe_scale, (valid - center) / safe_scale,
            np.flatnonzero(constant).tolist())


def prepare_views(raw_train, raw_valid, taxa, keep=16):
    """Return three exact train/validation views and aggregate provenance.

    Taxa are selected by decreasing *training* CLR log variance among those
    observed in at least 10 percent of training rows. Ties retain the original
    dictionary order. Fewer available taxa are retained, with at least two
    required. All pairwise two-part balances are built from these selected
    taxa. The same matrices may be reused as microbial main effects and as
    inputs to dose-product corrections.

    Views:
      ``taxa``: selected CLR coordinates, centered/scaled on training rows;
      ``balance``: (log(p_i)-log(p_j))/sqrt(2) for every selected i < j;
      ``two_part``: selected log abundances followed by presence indicators.

    Constants remain as explicit zero-centered training columns, with their
    indices recorded. No validation row can choose taxa or fit transformations.
    """
    if isinstance(keep, bool) or not isinstance(keep, Integral) or keep < 2:
        raise ValueError("INVALID_TAXON_KEEP")
    train = _raw(raw_train, "TRAIN")
    valid = _raw(raw_valid, "VALID", train.shape[1])
    if len(train) < 3:
        raise ValueError("INSUFFICIENT_TRAINING_PROFILES")
    if (not isinstance(taxa, (list, tuple)) or len(taxa) != train.shape[1]
            or any(not isinstance(name, str) or not name.strip() for name in taxa)
            or len(set(taxa)) != len(taxa)):
        raise ValueError("INVALID_TAXON_DICTIONARY_ALIGNMENT")

    logs = tuple(np.log(np.maximum(values, RELATIVE_ABUNDANCE_FLOOR))
                 for values in (train, valid))
    clr = tuple(values - values.mean(axis=1, keepdims=True) for values in logs)
    prevalence = np.mean(train > 0, axis=0)
    variance = np.var(clr[0], axis=0)
    # Treat numerical dust in exactly constant log coordinates as a tie.
    variance = np.where(variance <= CONSTANT_TOLERANCE ** 2, 0.0, variance)
    eligible = np.flatnonzero(prevalence >= PREVALENCE_THRESHOLD)
    if len(eligible) < 2:
        raise ValueError("FEWER_THAN_TWO_PREVALENT_TAXA")
    order = np.argsort(-variance[eligible], kind="stable")
    selected = eligible[order[:int(keep)]]
    pairs = [(i, j) for i in range(len(selected)) for j in range(i + 1, len(selected))]

    unscaled = {
        "taxa": tuple(values[:, selected] for values in clr),
        "balance": tuple(np.column_stack([
            (values[:, selected[i]] - values[:, selected[j]]) / np.sqrt(2.0)
            for i, j in pairs
        ]) for values in logs),
        "two_part": tuple(np.column_stack((values[:, selected], (raw[:, selected] > 0).astype(float)))
                          for values, raw in zip(logs, (train, valid), strict=True)),
    }
    output, constants = {}, {}
    for name, (a, b) in unscaled.items():
        a, b, constant = _standardize(a, b)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ArithmeticError("NONFINITE_COMPOSITIONAL_VIEW")
        output[name] = a, b
        constants[name] = constant

    metadata = dict(
        method="finite training-variance pairwise balance bank; selbal-inspired, not greedy selbal",
        requested_taxa=int(keep), selected_taxa_count=int(len(selected)),
        selected_indices=selected.tolist(), selected_taxa=[taxa[i] for i in selected],
        eligible_prevalent_taxa=int(len(eligible)), prevalence_threshold=PREVALENCE_THRESHOLD,
        relative_abundance_floor=RELATIVE_ABUNDANCE_FLOOR,
        zero_handling="fixed relative abundance floor before logarithm; zeros remain observed presences=0",
        column_counts={name: int(values[0].shape[1]) for name, values in output.items()},
        constant_training_columns=constants,
        balance_pairs=[dict(numerator=taxa[selected[i]], denominator=taxa[selected[j]]) for i, j in pairs],
        training_profiles=int(len(train)),
        selected_prevalence_q25_q50_q75=[float(v) for v in np.quantile(prevalence[selected], [.25, .5, .75])],
        selected_zero_fraction=float(np.mean(train[:, selected] == 0)),
        selected_positive_below_floor_fraction=float(np.mean(
            (train[:, selected] > 0) & (train[:, selected] < RELATIVE_ABUNDANCE_FLOOR))),
        maximum_balance_linear_rank=int(len(selected) - 1),
    )
    return output, metadata
