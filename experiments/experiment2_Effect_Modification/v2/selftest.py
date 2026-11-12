"""Generated-data checks for household isolation and the interaction contrast."""

from types import SimpleNamespace

import numpy as np

from dose_sensitivity import dose_selftest
from marker_effects import effect_selftest
from moderation_models import OPTIONS, ZERO, ecology, fit_base, run_fold, scores


def selftest():
    effect_selftest()
    dose_selftest()
    rng = np.random.default_rng(476)
    n = 220
    groups = np.arange(n) // 2
    raw = {
        "oral": rng.dirichlet(np.full(14, 1.1), n),
        "gut": rng.gamma(np.geomspace(.2, 5, n)[:, None], 1, size=(n, 16)),
    }
    raw["gut"] /= raw["gut"].sum(1, keepdims=True)
    log = {name: np.log(profile) for name, profile in raw.items()}
    clr = {name: values - values.mean(1, keepdims=True) for name, values in log.items()}
    z = ecology(raw["gut"])[:, 0]
    z = (z - z.mean()) / z.std()
    u, host = rng.normal(size=(2, n))
    x = np.c_[u, u ** 2, host]
    dose = x[:, :2]
    y = 100 + u + host + 3 * u * z + rng.normal(0, .1, n)
    outer = groups % 5
    inner = (groups // 5) % 3

    class Generated:
        def __init__(self, outcome):
            self.raw, self.groups, self.y, self.extra = raw, groups, outcome, None

        def split(self, train, valid):
            t, v = np.flatnonzero(train), np.flatnonzero(valid)
            return SimpleNamespace(
                train_rows=t, rows=v, yt=self.y[t], yv=self.y[v], ht=x[t], hv=x[v],
                dt=dose[t], dv=dose[v], delta=np.array([1.0, 1.0]),
                ot=clr["oral"][t], ov=clr["oral"][v], gt=clr["gut"][t], gv=clr["gut"][v],
                preprocessing={"host": {"labels": ["diet_linear", "diet_RCS", "age"]}},
            )

        def roles(self, repeat, fold):
            t = outer != fold
            return tuple(self.split(t & (inner != k), t & (inner == k)) for k in range(3)), self.split(t, ~t)

    config = ("smoke", 40, 7, 8)
    choices = {name: OPTIONS[name] for name in ("gut_shannon", "learned_gut", "selector")}
    result = run_fold(Generated(y), 0, 0, bases=(config,), options=choices)
    changed = run_fold(Generated(y + (outer == 0) * 1000), 0, 0, bases=(config,), options=choices)
    assert result[4] == changed[4]
    for a, b in zip(result[1:4], changed[1:4], strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    role, base, full, contrast, selected, _ = result
    br, fr, gain, _ = scores(role.y[1], base, full, groups[role.rows], draws=127)
    np.testing.assert_allclose(gain, fr - br, atol=1e-12)
    assert gain[0] > .25 and selected["gut_shannon"] != ZERO and contrast[:, 0].std() > .2
    null = run_fold(Generated(np.full(n, 100.0)), 0, 0, bases=(config,), options=choices)
    assert all(key == ZERO for key in null[4].values())

    main = fit_base(np.c_[x, z], y, x.shape[1], config)
    a, b = rng.normal(size=(2, 20, x.shape[1]))
    c, d = rng.normal(size=(2, 20))
    double_swap = main.predict(np.c_[a, c]) + main.predict(np.c_[b, d]) - main.predict(np.c_[a, d]) - main.predict(np.c_[b, c])
    np.testing.assert_allclose(double_swap, 0.0, atol=1e-10)

    shared_y = y + 2 * host * z
    host_choices = {name: OPTIONS[name] for name in ("gut_shannon", "learned_both")}
    shared = run_fold(Generated(shared_y), 0, 0, bases=(config,), options=host_choices, shared_host=True)
    changed_shared = run_fold(Generated(shared_y + (outer == 0) * 1000), 0, 0,
                              bases=(config,), options=host_choices, shared_host=True)
    assert shared[0].shared_columns == (2,)
    assert shared[4] == changed_shared[4]
    for a0, b0 in zip(shared[1:4], changed_shared[1:4], strict=True):
        np.testing.assert_allclose(a0, b0, atol=1e-12, rtol=1e-12)

    main = fit_base(np.c_[x, z], 6 * host * z, x.shape[1], config, shared_columns=(2,))
    a[:, 2] = b[:, 2]
    diet_swap = main.predict(np.c_[a, c]) + main.predict(np.c_[b, d]) - main.predict(np.c_[a, d]) - main.predict(np.c_[b, c])
    np.testing.assert_allclose(diet_swap, 0.0, atol=1e-10)
    b[:, :2] = a[:, :2]
    a[:, 2], b[:, 2] = -.8, .8
    age_swap = main.predict(np.c_[a, c]) + main.predict(np.c_[b, d]) - main.predict(np.c_[a, d]) - main.predict(np.c_[b, c])
    assert np.max(np.abs(age_swap)) > .1
