"""V9 method definitions, matched nulls and original/daily prediction pools."""

REFERENCES = ("v6_fusion", "v7_ridge_stack", "v8_ridge_wide", "v8_ridge_joint")
INTERACTIONS = {
    "balance_diet_oral": "balance_ridge_oral",
    "balance_diet_balanced": "balance_ridge_balanced",
    "eco_diet_oral": "eco_ridge_oral",
    "eco_diet_balanced": "eco_ridge_balanced",
}
META = (
    "convex_core",
    "ridge_core",
    "convex_wide",
    "ridge_wide",
    "convex_daily",
    "ridge_daily",
    "gated_soft",
    "gated_medium",
    "gated_strong",
    "tree_meta",
    "polynomial_stack",
    "inner_selector",
)


def specifications(config):
    out = [
        dict(name=family + "_" + site, site=site, **settings)
        for family, settings in config["families"].items()
        for site in ("oral", "balanced")
    ]
    out.extend(
        dict(
            name=name,
            null=base,
            learner="weak_interaction",
            grid=[dict(weight=0.0)] + [dict(weight=1.0, penalty=p) for p in config["interaction_penalties"]],
        )
        for name, base in INTERACTIONS.items()
    )
    return out


def names(config):
    return [s["name"] for s in specifications(config)] + list(META)


def pools(config):
    original = [s["name"] for s in specifications(config) if not s.get("daily")]
    daily = [s["name"] for s in specifications(config)]
    return dict(core=config["core_members"], wide=original, daily=daily)
