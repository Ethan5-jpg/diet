"""Frozen new inputs, complementary learners, corrections and matched pools."""

REFERENCES = ("v8_ridge_wide", "v10_ridge_diverse", "v10_replay")
CONTEXTS = (
    "full",
    "joint",
    "joint_upf",
    "shrink_global",
    "shrink_local",
    "shrink_uncertainty",
    "quality",
    "covariance",
    "rff",
    "prototype",
    "sliced",
)
CORRECTIONS = dict(
    resid_host_hgb="hgb_host_joint",
    resid_oral_hgb="hgb_host_joint",
    resid_balanced_hgb="hgb_host_joint",
    resid_oral_cat="hgb_host_joint",
    resid_oral_bayes="hgb_host_joint",
    resid_ridge_hgb="block_pc_oral_joint",
    resid_oralbase_oral="hgb_oral_joint",
    resid_oralbase_balanced="hgb_oral_joint",
)
INTERACTIONS = dict(
    cfiber_gut="hgb_balanced_joint",
    cfiber_oral="block_pc_oral_joint",
    cfiber_both="block_pc_balanced_joint",
    cfiber_macro="block_pc_balanced_joint",
)
POOLS = ("oral", "balanced", "new", "distribution", "combined")
META = tuple(kind + "_" + pool for pool in POOLS for kind in ("convex", "ridge", "positive")) + (
    "sparse_new",
    "sparse_combined",
)


def specifications(config):
    out = []

    def add(name, learner, site, context="joint", representation="clr", anchor=False):
        grid = config["families"][learner]["grid"]
        if site == "host":
            unique, seen = [], set()
            for case in grid:
                key = tuple(sorted((k, v) for k, v in case.items() if k not in ("count", "components")))
                if key not in seen:
                    seen.add(key)
                    unique.append(case)
            grid = unique
        out.append(
            dict(
                name=name,
                learner=learner,
                site=site,
                context=context,
                representation=representation,
                anchor=anchor,
                grid=grid,
                uses_gut=site == "balanced",
            )
        )

    for context in CONTEXTS:
        add("hgb_oral_" + context, "hgb", "oral", context, anchor=True)
        add("cat_balanced_" + context, "cat", "balanced", context, anchor=True)
    for family in ("hgb", "cat"):
        add(family + "_host_joint", family, "host", anchor=True)
    for learner, representation in (
        ("linear_tree", "pc"),
        ("dart", "clr"),
        ("posterior_cat", "clr"),
        ("huber_cat", "clr"),
        ("bayes", "spc"),
        ("ard", "spc"),
        ("adaptive_enet", "clr"),
    ):
        for site in ("oral", "balanced"):
            add(learner + "_" + site + "_joint", learner, site, representation=representation)
    for prefix, learner, representation in (
        ("spc_hgb", "hgb", "spc"),
        ("spc_cat", "cat", "spc"),
        ("quadratic_hgb", "hgb", "quadratic"),
        ("quadratic_ridge", "ridge", "quadratic"),
    ):
        for site in ("oral", "balanced"):
            add(
                prefix + "_" + site + "_joint",
                learner,
                site,
                representation=representation,
                anchor=learner in ("hgb", "cat"),
            )
    add("hgb_balanced_joint", "hgb", "balanced", anchor=True)
    add("cat_oral_joint", "cat", "oral", anchor=True)
    for site in ("oral", "balanced"):
        add("block_pc_" + site + "_joint", "block", site, representation="pc")
    stage_defs = [
        ("resid_host_hgb", "hgb", "host", "clr"),
        ("resid_oral_hgb", "hgb", "oral", "clr"),
        ("resid_balanced_hgb", "hgb", "balanced", "clr"),
        ("resid_oral_cat", "cat", "oral", "clr"),
        ("resid_oral_bayes", "bayes", "oral", "spc"),
        ("resid_ridge_hgb", "hgb", "oral", "clr"),
        ("resid_oralbase_oral", "hgb", "oral", "clr"),
        ("resid_oralbase_balanced", "hgb", "balanced", "clr"),
    ]
    by_name = {s["name"]: s for s in out}
    for name, family, site, rep in stage_defs:
        grid = [dict(weight=0.0)] + [
            dict(case, weight=w)
            for case in config["families"][family]["grid"][:2]
            for w in config["residual_weights"]
        ]
        out.append(
            dict(
                name=name,
                learner=family,
                site=site,
                representation=rep,
                context="joint",
                null=CORRECTIONS[name],
                grid=grid,
                uses_gut=site == "balanced" or by_name[CORRECTIONS[name]]["uses_gut"],
            )
        )
    for name, base in INTERACTIONS.items():
        source = dict(cfiber_gut="gut", cfiber_oral="oral", cfiber_both="balanced", cfiber_macro="macro")[
            name
        ]
        grid = [dict(weight=0.0)] + [
            dict(weight=1.0, components=3 if source == "macro" else k, penalty=p)
            for k, p in config["interaction_grid"]
        ]
        out.append(
            dict(
                name=name,
                learner="fiber",
                null=base,
                source=source,
                nonlinear=name == "cfiber_both",
                grid=grid,
                uses_gut=by_name[base]["uses_gut"] or source in ("gut", "balanced"),
            )
        )
    return out


def names(config):
    return [s["name"] for s in specifications(config)] + list(META)


def pools(config):
    matched = (
        "hgb",
        "cat",
        "linear_tree",
        "posterior_cat",
        "bayes",
        "ard",
        "adaptive_enet",
        "spc_cat",
        "block_pc",
    )
    out = {site: [prefix + "_" + site + "_joint" for prefix in matched] for site in ("oral", "balanced")}
    specs = specifications(config)
    out["new"] = [s["name"] for s in specs]
    out["distribution"] = [
        s["name"]
        for s in specs
        if s.get("context") not in ("full", "joint_upf")
        and s["name"].startswith(("hgb_oral_", "cat_balanced_"))
    ]
    out["combined"] = ["old__" + n for n in config["legacy_members"]] + out["new"]
    return out


def additions():
    pairs = {}
    for prefix in ("hgb_oral", "cat_balanced"):
        for context in CONTEXTS:
            if context != "joint":
                pairs[prefix + "_" + context] = prefix + "_joint"
    pairs.update(
        resid_oral_hgb="resid_host_hgb",
        resid_balanced_hgb="resid_oral_hgb",
        resid_oralbase_balanced="resid_oralbase_oral",
    )
    for kind in ("convex", "ridge", "positive"):
        pairs[kind + "_balanced"] = kind + "_oral"
        pairs[kind + "_combined"] = kind + "_new"
    return pairs
