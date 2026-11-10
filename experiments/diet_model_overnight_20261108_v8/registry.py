"""Frozen method breadth, matched nulls and separate original/joint-diet pools."""

REFERENCES = ("v4_hgb_oral", "v4_hgb_oral_gut", "v6_fusion", "v7_ridge_stack")
INTERACTIONS = {
    "sparse_diet_oral": "linear_main_oral",
    "sparse_diet_balanced": "linear_main_balanced",
    "nmf_diet_oral": "additive_nmf_oral",
    "nmf_diet_balanced": "additive_nmf_balanced",
}
META = (
    "convex_core",
    "convex_with_fm",
    "ridge_core",
    "convex_wide",
    "ridge_wide",
    "convex_joint",
    "ridge_joint",
    "inner_selector_core",
    "inner_selector_joint",
)


def specifications(config):
    out = []
    for family, settings in config["families"].items():
        for site in ("oral", "balanced"):
            out.append(dict(name=family + "_" + site, site=site, **settings))
    for name, null in INTERACTIONS.items():
        sparse = name.startswith("sparse")
        grid = (
            ([dict(weight=0.0)] + [dict(weight=1.0, fraction=p) for p in config["sparse_fractions"]])
            if sparse
            else ([dict(weight=0.0)] + [dict(weight=1.0, penalty=p) for p in config["interaction_penalties"]])
        )
        out.append(
            dict(
                name=name, null=null, learner="sparse_interaction" if sparse else "nmf_interaction", grid=grid
            )
        )
    return out


def names(config):
    return [v["name"] for v in specifications(config)] + list(META)


def pools(config):
    specs = specifications(config)
    # Wide pools exclude known fragile FM; the core pair isolates adding FM.
    original = [v["name"] for v in specs if not v.get("joint") and v["learner"] != "fm"]
    joint = original + [v["name"] for v in specs if v.get("joint")]
    core = config["core_members"]
    return dict(core=core, with_fm=core + ["fm_safe_balanced"], wide=original, joint=joint)
