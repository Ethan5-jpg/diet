"""Training-only gut modifier views for the expanded, matched V7 grid."""

import numpy as np


# Sine and cosine supply 24 coordinates for each map. Broad and narrow
# bandwidths on the same PCs share random directions.
RFF_BANDS = {"broad": (10, 0.25), "narrow": (10, 1.0), "medium20": (20, 0.5)}
N_FREQUENCIES = 12
RFF_SEED = 20261118
KEEP_TAXA = 12
PREVALENCE = 0.10
LOG_FLOOR = 1e-6
COMPOSITION_WIDTHS = {
    "clr": KEEP_TAXA,
    "pivot": KEEP_TAXA - 1,
    "detect_intensity": 2 * KEEP_TAXA,
}


def labels(view, width):
    return tuple(f"v7_{view}_{i + 1}" for i in range(width))


def _center(train, valid):
    mean = train.mean(axis=0)
    return train - mean, valid - mean


def fit_rff(train, valid, band):
    """Approximate exp(-gamma * squared PC distance), gamma = factor / p."""
    p, gamma_factor = RFF_BANDS[band]
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != p or valid.shape[1] != p:
        raise ValueError("RFF_PC_WIDTH_CHANGED")
    directions = np.random.default_rng(RFF_SEED + p).standard_normal((p, N_FREQUENCIES))
    omega = directions * np.sqrt(2.0 * gamma_factor / p)
    origin = train.mean(axis=0)
    features = []
    for role in (train, valid):
        angle = (role - origin) @ omega
        features.append(np.column_stack((np.cos(angle), np.sin(angle))) / np.sqrt(N_FREQUENCIES))
    return _center(*features)


def _pad(matrix, width):
    # Padding only matters in a fold with fewer than KEEP_TAXA eligible taxa.
    # All missing coordinates have zero training and validation values.
    if matrix.shape[1] == width:
        return matrix
    return np.pad(matrix, ((0, 0), (0, width - matrix.shape[1])))


def fit_composition(train, valid):
    """Return three subject-level views and the indices selected in this role.

    Rank common taxa by training variance of their centered log-ratio across
    all original columns. The stable order breaks equal variances by archived
    column order. Each pivot compares a taxon to the geometric mean of the
    remaining selected taxa. Conditional intensity is floored log abundance
    minus its mean among positive training records. An absent taxon has
    intensity zero, with its detection separately represented.
    """
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("COMPOSITION_WIDTH_MISMATCH")
    for matrix in (train, valid):
        if (not np.isfinite(matrix).all() or np.any(matrix < 0) or
                not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-3, rtol=0)):
            raise ValueError("EXPECTED_FRACTIONAL_GUT_PROFILE")

    # Work on a new floating-point buffer; the archived arrays are never modified.
    train_clr = np.maximum(train, LOG_FLOOR, dtype=float)
    np.log(train_clr, out=train_clr)
    train_clr -= train_clr.mean(axis=1, keepdims=True)
    prevalence = (train > 0).mean(axis=0)
    eligible = np.flatnonzero(prevalence >= PREVALENCE)
    variance = train_clr.var(axis=0)
    selected = eligible[np.argsort(-variance[eligible], kind="stable")[:KEEP_TAXA]]
    count = len(selected)
    selected_train_clr = train_clr[:, selected]
    del train_clr

    valid_log = np.maximum(valid, LOG_FLOOR, dtype=float)
    np.log(valid_log, out=valid_log)
    selected_valid_clr = valid_log[:, selected] - valid_log.mean(axis=1, keepdims=True)
    del valid_log
    raw = train[:, selected], valid[:, selected]
    logs = tuple(np.log(np.maximum(x, LOG_FLOOR, dtype=float)) for x in raw)
    positive = raw[0] > 0
    positive_log = np.where(positive, logs[0], 0.0).sum(axis=0) / positive.sum(axis=0)

    pivots, two_part = [], []
    for abundance, log_abundance in zip(raw, logs):
        # The ILR basis is triangular and orthonormal on the selected subcomposition.
        contrasts = []
        for i in range(count - 1):
            k = count - i - 1
            reference = log_abundance[:, i + 1:].mean(axis=1)
            contrasts.append(np.sqrt(k / (k + 1)) * (log_abundance[:, i] - reference))
        pivots.append(np.column_stack(contrasts) if contrasts else np.empty((len(abundance), 0)))
        detected = abundance > 0
        intensity = np.where(detected, log_abundance - positive_log, 0.0)
        two_part.append(np.column_stack((_pad(detected, KEEP_TAXA), _pad(intensity, KEEP_TAXA))))

    result = {}
    for name, pair in (("clr", (selected_train_clr, selected_valid_clr)),
                       ("pivot", pivots), ("detect_intensity", two_part)):
        width = COMPOSITION_WIDTHS[name]
        result[name] = _center(*(_pad(matrix, width) for matrix in pair))
    return result, selected, prevalence[selected]


def add_views(bank, source, split, audit=None):
    """Append deterministic training-fit representations, each with shared marginals."""
    def register(view, pair):
        for j, name in enumerate(labels(view, pair[0].shape[1])):
            bank[name] = pair[0][:, j], pair[1][:, j]

    for band, (p, _) in RFF_BANDS.items():
        pcs = tuple(np.column_stack([bank[f"g{i}"][side] for i in range(1, p + 1)])
                    for side in (0, 1))
        register("rff_" + band, fit_rff(*pcs, band))

    raw = source.raw["gut"]
    composition, selected, prevalences = fit_composition(raw[split.train_rows], raw[split.rows])
    for view, pair in composition.items():
        register(view, pair)

    for marker in ("shannon", "fermentation"):
        for i in range(1, 5):
            key = f"v7_gate_{marker}_g{i}"
            pc = bank[f"g{i}"]
            mean = pc[0].mean()
            pair = tuple((a - mean) * b for a, b in zip(pc, bank[marker]))
            bank[key] = _center(*pair)
    if audit is not None:
        audit.update(selected_gut_taxa=[source.taxa["gut"][int(i)] for i in selected],
                     selected_gut_indices=selected.tolist(),
                     selected_training_prevalences=prevalences.tolist(),
                     training_rows=len(split.train_rows), requested_taxa=KEEP_TAXA,
                     min_prevalence=PREVALENCE, log_floor=LOG_FLOOR)
