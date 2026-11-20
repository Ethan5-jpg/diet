"""Read fixed paired household moments and estimate precision for future evaluation."""

from math import ceil, sqrt
from statistics import NormalDist


def readout(moments):
    from scipy.stats import t
    from run_v2 import require

    n,h,y,sy2,a,b,c,ss,dn,n2 = (float(x) for x in moments)
    require(n > 0 and h > 1, "V24_INVALID_EVALUATION_SUPPORT")
    denominator = sy2 - y*y/n
    require(denominator > 1e-10*abs(sy2), "V24_INVALID_EVALUATION_VARIANCE")
    factor = 100./denominator
    gain = b-c
    center = gain/n
    se = sqrt(max(0.,(ss - 2.*center*dn + center*center*n2)*h/(h-1.)))
    require(se > 0., "V24_DEGENERATE_HOUSEHOLD_ERRORS")
    return dict(people=int(n), families=int(h), diet_main_pp=(a-b)*factor, interaction_pp=gain*factor,
                diet_total_pp=(a-c)*factor, p_one_sided=float(t.sf(gain/se,h-1.)), se_pp=se*factor)


def sample_plan(replays, practical=.10, alpha=.05/3., power=.80):
    sd = max(r["se_pp"]*sqrt(r["families"]) for r in replays)
    normal = NormalDist()
    critical = normal.inv_cdf(1.-alpha)
    planned = normal.inv_cdf(power)
    estimates = []
    for effect in (.10,.15,.20):
        significance = ceil(((critical+planned)*sd/effect)**2)
        floor = ceil((planned*sd/(effect-practical))**2) if effect > practical else None
        estimates.append(dict(effect_pp=effect, significance_only=significance,
                              significance_and_practical=max(significance,floor) if floor is not None else None))
    return dict(sigma_per_sqrt_family=sd, assumptions=dict(alpha=alpha,power=power,minimum_pp=practical),
                family_scenarios=estimates)
