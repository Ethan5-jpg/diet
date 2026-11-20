"""Describe assay families and redundancy of discovery-stage lipid candidates."""

from collections import Counter, defaultdict
from itertools import combinations
import re


def family(endpoint):
    source, _, raw = endpoint.partition(":")
    x = raw.lower()
    if source == "nightingale":
        match = re.match(r"^(?:[a-z]{1,3}_)?(hdl|idl|ldl|vldl)(?:_|$)", x)
        if match:
            return match.group(1).upper()
        if re.match(r"^(?:dha|ala|la|fa|omega_[369]|mufa|pufa|sfa|unsaturation)(?:_|$)", x):
            return "fatty_acids"
        if x.startswith(("sphingomyelin", "ceramide")):
            return "sphingolipids"
        if x.startswith(("phospholipid", "phosphatidyl")):
            return "phospholipids"
        if x.startswith(("total_", "remnant_", "apoa", "apob", "clinical_")):
            return "overall_lipids"
    if source == "serum_lipidomics":
        prefix = re.split(r"[_ :]", x, 1)[0]
        if prefix in {"cer", "sm", "hexcer", "laccer", "dhcer", "sph", "hc", "dc"}:
            return "sphingolipids"
        if prefix in {"pc", "pe", "pi", "ps", "pg", "lpc", "lpe", "lpi", "lysopc", "lysope", "plasmalogen"}:
            return "phospholipids"
        if prefix in {"tg", "tag", "dag", "dg", "mg", "mag"}:
            return "glycerides"
        if prefix in {"ce", "fc", "cho"}:
            return "sterols"
        if prefix in {"fa", "ffa"}:
            return "fatty_acids"
    return "unclassified"


def cluster(endpoints, values, minimum=120, threshold=.85):
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.stats import rankdata

    n = len(endpoints)
    if n < 2:
        return ([endpoints] if n else []), dict(comparable_pairs=0, correlated_pairs=0)
    distance, comparable, correlated = [], 0, 0
    for i, j in combinations(range(n), 2):
        first, second = values[:, i], values[:, j]
        mask = np.isfinite(first) & np.isfinite(second)
        r = 0.
        if mask.sum() >= minimum:
            rank = rankdata(np.column_stack((first[mask], second[mask])), axis=0)
            if (rank.std(axis=0) > 0.).all():
                r = min(1., abs(float(np.corrcoef(rank.T)[0, 1])))
                comparable += 1
                correlated += int(r >= threshold)
        distance.append(1. - r)
    labels = fcluster(linkage(np.asarray(distance), method="complete"), t=1. - threshold, criterion="distance")
    groups = defaultdict(list)
    for e, k in zip(endpoints, labels, strict=True):
        groups[int(k)].append(e)
    return sorted(groups.values(), key=lambda x: (-len(x), x)), dict(comparable_pairs=comparable, correlated_pairs=correlated)


def describe(prior, current, domains, values, endpoints):
    from taxonomy import GROUPS

    previous = set().union(*(set(prior[g]) for g in GROUPS))
    recent = set().union(*(set(current[g]) for g in GROUPS))
    index = {e: j for j, e in enumerate(endpoints)}
    lipid = "circulating_lipids_metabolites"
    output = {}
    for label, candidate in (("v16", previous), ("v17", recent), ("replicated_across_rounds", previous & recent)):
        selected = sorted(e for e in candidate if domains[e] == lipid)
        groups, pairs = cluster(selected, values[:, [index[e] for e in selected]])
        output[label] = dict(all=len(candidate), lipid=len(selected), families=dict(Counter(map(family, selected))),
                             sources=dict(Counter(e.partition(":")[0] for e in selected)),
                             clusters=groups, **pairs)
    output["same_category_replication"] = {g: sorted(set(prior[g]) & set(current[g])) for g in GROUPS}
    return output
