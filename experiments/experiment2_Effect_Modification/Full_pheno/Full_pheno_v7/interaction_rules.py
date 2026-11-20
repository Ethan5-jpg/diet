"""Fixed modifier recipes and a bounded upgrade learned from inner predictions."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Rule(NamedTuple):
    name: str
    diet: str
    pc5: float
    pc2: float = 0.
    oral: float = 0.
    pooling: float = 0.
    robust: bool = False


REFERENCE = Rule("pc5_half", "legacy", .5)
RULES = (
    REFERENCE,
    Rule("legacy_pc5_080", "legacy", .8),
    Rule("host_pc5_half", "host", .5),
    Rule("host_pc5_065", "host", .65),
    Rule("host_pc5_080", "host", .8),
    Rule("curve_pc5_065", "curve", .65),
    Rule("linear_pc5_065", "linear", .65),
    Rule("host_pc2_mix", "host", .45, .35),
    Rule("curve_pc2_mix", "curve", .45, .35),
    Rule("linear_pc2_mix", "linear", .45, .35),
    Rule("host_oral_add", "host", .65, oral=.20),
    Rule("legacy_oral_add", "legacy", .65, oral=.20),
    Rule("host_domain_pc5", "host", .8, pooling=.65),
    Rule("host_domain_pc2_mix", "host", .45, .35, pooling=.65),
    Rule("curve_domain_oral_add", "curve", .65, oral=.20, pooling=.65),
    Rule("host_robust_pc5", "host", .8, robust=True),
    Rule("host_robust_oral_add", "host", .65, oral=.20, robust=True),
)
MENU = ("host_pc5_080", "host_pc2_mix", "host_oral_add", "host_domain_pc2_mix", "host_robust_pc5")
METHODS = (*(r.name for r in RULES), "inner_bounded_upgrade")


@dataclass(frozen=True)
class Upgrade:
    index: int = 0
    weight: float = 0.


def choose_upgrade(parts):
    """Change at most half the reference prediction in the original units."""
    y = np.concatenate([part.outcome for part in parts])
    sst = float(np.sum((y - y.mean())**2))
    if not np.isfinite(sst) or sst <= 0.:
        raise ValueError("V7_INVALID_INNER_OUTCOME_VARIANCE")
    best = Upgrade()
    best_score = (0., 0.)
    for name in MENU:
        j = METHODS.index(name)
        for weight in (.25, .5):
            on, off, raw_interaction = [], [], []
            for part in parts:
                b0, c0 = part.off[:, 0], part.on[:, 0]
                residual_on, residual_off = part.outcome - c0, part.outcome - b0
                dc = weight * (part.on[:, j]-c0)
                db = weight * (part.off[:, j]-b0)
                on.append(float(dc @ (2.*residual_on - dc)) / sst)
                off.append(float(db @ (2.*residual_off - db)) / sst)
                raw_interaction.append(float(residual_off @ residual_off - residual_on @ residual_on) / sst)
            on, off = np.array(on), np.array(off)
            interaction = on - off
            if (on.sum() < .0002 or interaction.sum() < .0002 or np.count_nonzero(on > 0.) < 3
                    or np.count_nonzero(interaction > 0.) < 3 or np.sum(raw_interaction + interaction) <= 0.):
                continue
            score = (float(interaction.sum()), float(on.sum()))
            if score > best_score:
                best, best_score = Upgrade(j, weight), score
    return best


def domain_prior(profiles):
    if not profiles:
        return None
    gram = np.mean([profile.gram for profile in profiles], axis=0)
    rhs = np.mean([profile.rhs for profile in profiles], axis=0)
    return np.linalg.solve(gram + np.eye(len(rhs)), rhs)


def prediction(profile, penalty, pooling, prior):
    if prior is None or pooling == 0.:
        return profile.fit(penalty)
    return profile.fit(penalty + pooling, prior=pooling*prior/(penalty + pooling))
