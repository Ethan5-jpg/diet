"""Frozen input ladders, new learner families and matched interaction backgrounds."""

REFERENCES = ("v8_ridge_wide", "v8_ridge_joint", "v9_convex_wide")
LADDER = ("none", "macro", "sodium", "life", "full")
VARIANTS = (
    ("hgb_oral", "hgb", "oral"),
    ("cat_balanced", "cat", "balanced"),
    ("block_balanced", "block", "balanced"),
    ("hgb_host", "hgb", "host"),
    ("cat_host", "cat", "host"),
)
INTERACTIONS = dict(
    fiber_gut_linear="block_pc_balanced_full",
    fiber_oral_linear="block_pc_balanced_full",
    fiber_both_rcs="block_pc_balanced_full",
    fiber_gut_tree="cat_balanced_full",
    fiber_macro="block_pc_balanced_full",
    fiber_gut_kernel="kernel_balanced_full",
)
SOURCES = dict(
    fiber_gut_linear="gut",
    fiber_oral_linear="oral",
    fiber_both_rcs="balanced",
    fiber_gut_tree="gut",
    fiber_macro="macro",
    fiber_gut_kernel="gut",
)
POOL_NAMES = ("base", "macro", "full", "diverse", "wide")
META = tuple(kind + "_" + pool for pool in POOL_NAMES for kind in ("convex", "ridge"))


def specifications(config):
    out = []

    def add(name, learner, site, context, representation="clr"):
        grid, seen = [], set()
        for case in config["families"][learner]["grid"]:
            effective = case.copy()
            if site == "host":
                for unused in ("count", "components"):
                    effective.pop(unused, None)
            if learner == "block" and site in ("oral", "gut"):
                ratio = effective.pop("gut_weight", 1)
                if site == "gut":
                    effective["micro_weight"] *= ratio
            key = tuple(sorted(effective.items()))
            if key not in seen:
                grid.append(case)
                seen.add(key)
        out.append(
            dict(
                name=name,
                learner=learner,
                site=site,
                context=context,
                representation=representation,
                grid=grid,
            )
        )

    for context in LADDER:
        for prefix, learner, site in VARIANTS:
            add(prefix + "_" + context, learner, site, context)
    for learner, site in (("hgb", "balanced"), ("cat", "oral"), ("block", "oral"), ("block", "gut")):
        add(learner + "_" + site + "_full", learner, site, "full")
    for context in ("macro_life", "joint", "fiber_days", "distribution", "ratios"):
        for prefix, learner, site in VARIANTS[:2]:
            add(prefix + "_" + context, learner, site, context)
    for learner in ("spline", "svr", "extra", "kernel"):
        for site in ("oral", "balanced"):
            add(learner + "_" + site + "_full", learner, site, "full", "pc")
    for learner, site in (("hgb", "oral"), ("hgb", "balanced"), ("cat", "balanced")):
        add("hurdle_" + learner + "_" + site + "_full", learner, site, "full", "hurdle")
    add("eco_cat_balanced_full", "cat", "balanced", "full", "ecohybrid")
    add("block_pc_balanced_full", "block", "balanced", "full", "pc")
    for name, base in INTERACTIONS.items():
        grid = [dict(weight=0.0)]
        for k, p in config["interaction_grid"]:
            case = dict(weight=1.0, components=3 if SOURCES[name] == "macro" else k, penalty=p)
            if case not in grid:
                grid.append(case)
        out.append(
            dict(
                name=name,
                null=base,
                learner="correction",
                source=SOURCES[name],
                nonlinear=name == "fiber_both_rcs",
                grid=grid,
            )
        )
    return out


def names(config):
    return [s["name"] for s in specifications(config)] + list(META)


def pools(config):
    out = {
        label: [v[0] + "_" + context for v in VARIANTS]
        for label, context in (("base", "none"), ("macro", "macro"), ("full", "full"))
    }
    out["diverse"] = [
        s["name"] for s in specifications(config) if s.get("context") == "full" or s["name"] in INTERACTIONS
    ]
    out["wide"] = [s["name"] for s in specifications(config)]
    return out


def additions():
    out = {}
    for prefix, _, _ in VARIANTS:
        for context, base in (("macro", "none"), ("sodium", "macro"), ("life", "none"), ("full", "sodium")):
            out[prefix + "_" + context] = prefix + "_" + base
    for context in ("macro_life", "joint", "fiber_days", "distribution", "ratios"):
        for prefix, _, _ in VARIANTS[:2]:
            out[prefix + "_" + context] = prefix + "_full"
    for kind in ("convex", "ridge"):
        out[kind + "_macro"] = kind + "_base"
        out[kind + "_full"] = kind + "_macro"
        out[kind + "_diverse"] = kind + "_full"
    return out
