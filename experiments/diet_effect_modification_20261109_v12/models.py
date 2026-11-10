"""Matched block-penalised ridge models, including an exact no-interaction option."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg


def specifications():
    return [dict(name="%s_%s_%s" % (module, form, site), module=module, form=form, site=site)
            for module in ("ecology", "composition") for form in ("linear", "spline")
            for site in ("oral", "gut", "both")]


def ridge_fit(x, y, main_penalty, z=None, interaction_penalty=None):
    a = x if z is None else np.column_stack([x, z])
    n = len(y)
    penalties = np.full(a.shape[1], float(main_penalty))
    penalties[0] = 0.0
    if z is not None:
        if interaction_penalty is None or interaction_penalty <= 0:
            raise ValueError("Positive interaction penalty required")
        penalties[x.shape[1]:] = interaction_penalty
    gram = a.T @ a
    gram.flat[::len(gram) + 1] += n * penalties
    return linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=False), a.T @ y, check_finite=False)


@dataclass
class Fitted:
    design: object
    coefficient: object
    spec: object = None
    interaction_map: object = None

    def predict(self, training=False, diet=None):
        main = self.design.main(training, diet)
        if self.spec is None:
            return main @ self.coefficient
        z = self.interaction_map.transform(self.design, self.spec, training, diet)
        return np.column_stack([main, z]) @ self.coefficient

    def contrasts(self):
        lo, hi = self.design.config["contrast"]
        total = self.predict(diet=hi) - self.predict(diet=lo)
        if self.spec is None:
            component = np.zeros(len(total))
        else:
            product_delta = self.interaction_map.product_difference(self.design, self.spec, lo, hi)
            component = product_delta @ self.coefficient[self.design.xt.shape[1]:]
        return total, component


def fit_main(design, penalty):
    coefficient = ridge_fit(design.xt, design.role.split.yt, penalty)
    return Fitted(design, coefficient)


def fit_augmented(design, spec, main_penalty, interaction_penalty, base=None):
    if interaction_penalty is None:
        return base if base is not None else fit_main(design, main_penalty)
    zt, _, state = design.interactions(spec)
    if not zt.shape[1]:
        return base if base is not None else fit_main(design, main_penalty)
    coefficient = ridge_fit(design.xt, design.role.split.yt, main_penalty, zt, interaction_penalty)
    return Fitted(design, coefficient, spec, state)
