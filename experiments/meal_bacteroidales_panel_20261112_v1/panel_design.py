"""Fixed three-species carbohydrate-metabolism candidate and matched scores."""

import numpy as np


SPECIES = ("s__Alistipes_indistinctus", "s__Alistipes_finegoldii", "s__Bacteroides_thetaiotaomicron")
PSEUDO_FRACTION = 1e-4
MODELS = ("panel", "pcs", "joint")
COMPARISONS = {
    "panel_vs_baseline": ("baseline", "panel"),
    "pcs_vs_baseline": ("baseline", "pcs"),
    "joint_vs_pcs": ("pcs", "joint"),
    "panel_vs_pcs": ("pcs", "panel"),
}


def species_columns(taxa):
    terminal = [path.rsplit("|", 1)[-1] for path in taxa]
    if any(terminal.count(name) != 1 for name in SPECIES):
        raise ValueError("Each named species must align exactly once")
    return [terminal.index(name) for name in SPECIES]


def panel_mass(raw, taxa):
    if (raw.ndim != 2 or raw.shape[1] != len(taxa) or not np.isfinite(raw).all()
            or (raw < 0).any() or not np.allclose(raw.sum(axis=1), 1., rtol=0, atol=.001)):
        raise ValueError("Unexpected archived gut abundance scale")
    values = raw[:, species_columns(taxa)]
    return np.log1p(values.sum(axis=1) / PSEUDO_FRACTION), (values > 0).sum(axis=0)


def score_losses(values, seed):
    baseline, panel, pcs, joint, tss = values
    active = np.flatnonzero(np.any(values != 0, axis=0))
    if np.count_nonzero(tss > 0) < 2 or not np.isfinite(values).all():
        raise ValueError("Insufficient paired household outcome variance")
    draws = np.random.default_rng(seed).choice(active, (1999, len(active)), replace=True)
    denominator = tss[draws].sum(axis=1)
    if not (denominator > 0).all():
        raise ValueError("Nonpositive paired household denominator")
    losses = dict(zip(("baseline", *MODELS), (baseline, panel, pcs, joint)))
    result = {"r2": {name: float(1 - value.sum() / tss.sum()) for name, value in losses.items()}}
    for name, (reference, alternative) in COMPARISONS.items():
        gain = losses[reference] - losses[alternative]
        result[name] = dict(delta_r2=float(gain.sum() / tss.sum()),
                            interval=np.quantile(gain[draws].sum(axis=1) / denominator, [.025, .975]).tolist())
    return result
