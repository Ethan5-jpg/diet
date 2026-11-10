"""A frozen breadth-first grid: 30 new pipelines, 67 candidate settings."""

INTERACTIONS = {
    "ebm_diet_gut": "ebm_balanced",
    "additive_pc_interaction": "additive_boost",
    "additive_gate_interaction": "additive_boost",
}
REFERENCES = ("v4_hgb_oral", "v4_hgb_oral_gut", "v6_fusion")
META = ("inner_selector", "convex_blend", "ridge_stack")


def specifications(config):
    out = []
    for family, grid in config["grids"].items():
        sites = ("host", "oral", "gut", "balanced") if family == "lgbm" else ("oral", "balanced")
        for site in sites:
            out.append(dict(name=family + "_" + site, family=family, site=site, grid=grid))
    out.extend(
        [
            dict(
                name="gut_mixture", family="mixture", site="balanced", grid=config["mixture_grid"]
            ),
            dict(
                name="additive_boost",
                family="additive",
                site="balanced",
                grid=config["additive_grid"],
            ),
            dict(
                name="ebm_diet_gut",
                family="ebm_interaction",
                grid=[dict(weight=w) for w in (0.0, 0.5, 1.0)],
            ),
            dict(
                name="additive_pc_interaction",
                family="pc_interaction",
                grid=[dict(weight=0.0)]
                + [dict(weight=1.0, penalty=p) for p in config["interaction_penalties"]],
            ),
            dict(
                name="additive_gate_interaction",
                family="gate_interaction",
                grid=[dict(weight=0.0)]
                + [
                    dict(weight=1.0, clusters=k, penalty=p)
                    for k in (2, 3)
                    for p in config["interaction_penalties"][:2]
                ],
            ),
        ]
    )
    return out


def names(config):
    return [s["name"] for s in specifications(config)] + list(META)
