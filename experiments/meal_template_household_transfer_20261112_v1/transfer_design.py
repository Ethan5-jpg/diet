"""Compare fixed repeated-template models on identical held-out households."""

import numpy as np


def predict(x, interaction, background, projection, beta, reference_beta):
    baseline = x @ background
    residual = interaction - x @ projection
    return baseline, residual @ beta, residual @ reference_beta


def household_losses(y, weights, family, baseline, change, reference, n_households):
    loss = ((y - baseline) ** 2, (y - baseline - change) ** 2,
            (y - baseline - reference) ** 2, y ** 2)
    return np.stack([np.bincount(family, weights=weights * part, minlength=n_households) for part in loss])


def score_summary(losses, seed):
    null, template, reference, total = losses
    active = np.flatnonzero(np.any(losses != 0, axis=0))
    if np.count_nonzero(total > 0) < 2 or not np.isfinite(losses).all():
        raise ValueError("Insufficient held-out household outcome variance")
    draws = np.random.default_rng(seed).choice(active, (1999, len(active)), replace=True)
    denominator = total[draws].sum(axis=1)
    if not (denominator > 0).all():
        raise ValueError("Nonpositive paired bootstrap denominator")

    def comparison(gain):
        return dict(delta_r2=float(gain.sum() / total.sum()),
                    interval=np.quantile(gain[draws].sum(axis=1) / denominator, [.025, .975]).tolist())

    gain = null - template
    leave_one = (gain.sum() - gain[active]) / (total.sum() - total[active])
    return dict(
        baseline_r2=float(1 - null.sum() / total.sum()),
        template_r2=float(1 - template.sum() / total.sum()),
        reference_r2=float(1 - reference.sum() / total.sum()),
        template_vs_baseline=comparison(gain),
        template_vs_reference=comparison(reference - template),
        reference_vs_baseline=comparison(null - reference),
        leave_one_template_vs_baseline=[float(leave_one.min()), float(leave_one.max())],
    )
