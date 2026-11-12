"""Linear prediction and paired household summaries for synthetic outcomes."""

import numpy as np


AMPLITUDES = (0., .25, .5, 1., 2., 4.)
METHODS = ("pc_ridge", "pc_stronger_ridge", "known_direction")


def prediction_errors(train_x, valid_x, train_z, valid_z, train_truth, valid_truth, train_y, valid_y, weights):
    """Columns of y hold noise replicates followed by the unit-signal response."""
    sw = np.sqrt(weights)
    u, singular, vt = np.linalg.svd(train_x * sw[:, None], full_matrices=False)
    keep = singular > singular[0] * 1e-10
    u = u[:, keep]
    inverse = vt[keep].T / singular[keep]
    weighted_y = train_y * sw[:, None]
    coordinate = u.T @ weighted_y
    baseline = valid_x @ (inverse @ coordinate)
    target = weighted_y - u @ coordinate
    errors = {"baseline": valid_y - baseline, "target": valid_y}
    for name, z_train, z_valid, penalty in (
        ("pc_ridge", train_z, valid_z, .03),
        ("pc_stronger_ridge", train_z, valid_z, .3),
        ("known_direction", train_truth[:, None], valid_truth[:, None], .03),
    ):
        weighted = z_train * sw[:, None]
        rotation = u.T @ weighted
        residual = weighted - u @ rotation
        beta = np.linalg.solve(residual.T @ residual + penalty * weights.sum() * np.eye(z_train.shape[1]),
                               residual.T @ target)
        change = (z_valid - valid_x @ (inverse @ rotation)) @ beta
        errors[name] = valid_y - baseline - change
    return errors


def household_moments(errors, family, weights, count):
    """Store squared-error polynomials over the injected amplitude."""
    order = np.argsort(family, kind="stable")
    groups, starts = np.unique(family[order], return_index=True)
    moments = {}
    for name, error in errors.items():
        noise, unit = error[order, :-1], error[order, -1:]
        w = weights[order, None]
        parts = np.zeros((3, count, noise.shape[1]))
        parts[0, groups] = np.add.reduceat(w * noise ** 2, starts, axis=0)
        parts[1, groups] = np.add.reduceat(2 * w * noise * unit, starts, axis=0)
        parts[2, groups] = np.add.reduceat(w * unit ** 2, starts, axis=0)
        moments[name] = parts
    return moments


def recoverability(moments, seed, n_boot):
    target = moments["target"]
    active = np.logical_or.reduce([np.any(values != 0, axis=(0, 2)) for values in moments.values()])
    total = target[:, active].sum(axis=1)
    households = int(active.sum())
    if households < 2:
        raise ValueError("Insufficient paired households for simulation")
    rng = np.random.default_rng(seed)
    draw = rng.integers(households, size=(n_boot, households))
    counts = np.zeros((n_boot, households))
    np.add.at(counts, (np.arange(n_boot)[:, None], draw), 1)

    def boot(values):
        return np.stack([counts @ part[active] for part in values])

    def curve(values, a):
        return values[0] + a * (values[1] + a * values[2])

    target_boot = boot(target)
    results = {}
    for name in METHODS:
        gain = moments["baseline"] - moments[name]
        gain_boot, gain_total = boot(gain), gain[:, active].sum(axis=1)
        rows = []
        for amplitude in AMPLITUDES:
            denominator = curve(target_boot, amplitude)
            if not (denominator > 0).all():
                raise ValueError("Nonpositive simulation bootstrap denominator")
            point = curve(gain_total, amplitude) / curve(total, amplitude)
            lower = np.quantile(curve(gain_boot, amplitude) / denominator, .025, axis=0)
            rows.append(np.stack([point, lower]))
        results[name] = np.stack(rows)
    return results


def aggregate(arrays):
    report = {}
    for name, values in arrays.items():
        rows = []
        for amplitude, (point, lower) in zip(AMPLITUDES, values):
            probability = float(np.mean(lower > 0))
            n = len(lower)
            z = 1.959963984540054
            denom = 1 + z * z / n
            middle = (probability + z * z / (2 * n)) / denom
            width = z * np.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n)) / denom
            rows.append(dict(amplitude=amplitude, positive_interval_rate=probability,
                             monte_carlo_se=float(np.sqrt(probability * (1 - probability) / n)),
                             monte_carlo_interval=[float(max(0, middle - width)), float(min(1, middle + width))],
                             median_delta_r2=float(np.median(point))))
        threshold = next((row["amplitude"] for i, row in enumerate(rows)
                          if all(x["positive_interval_rate"] >= .8 for x in rows[i:])), None)
        report[name] = dict(grid=rows, first_persistent_80_percent=threshold)
    return report
