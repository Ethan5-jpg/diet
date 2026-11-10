"""Matched inverse-regression representations and complementary response objectives."""

REFERENCES = ("v10_ridge_diverse", "v11_positive_combined", "v11_sparse_combined", "v11_replay")
META = tuple(
    k + "_" + p
    for p in ("oral", "balanced", "non_tree", "new", "combined")
    for k in ("convex", "ridge", "positive")
) + ("sparse_new", "sparse_combined")


def specifications(config):
    out = []
    for rep in ("pc", "sir", "save"):
        for learner in ("hgb", "cat", "mars", "gp"):
            for site in ("oral", "balanced"):
                out.append(
                    dict(
                        name=learner + "_" + rep + "_" + site,
                        learner=learner,
                        representation=rep,
                        site=site,
                        context="joint",
                        grid=config["families"][learner]["grid"],
                    )
                )
    for rep in ("pc", "save"):
        for learner in ("bin_cat", "bin_hgb", "quantile_cat"):
            for site in ("oral", "balanced"):
                out.append(
                    dict(
                        name=learner + "_" + rep + "_" + site,
                        learner=learner,
                        representation=rep,
                        site=site,
                        context="joint",
                        grid=config["families"][learner]["grid"],
                    )
                )
    for site in ("oral", "balanced"):
        out.append(
            dict(
                name="graph_" + site,
                learner="graph",
                representation="graph",
                site=site,
                context="joint",
                grid=config["families"]["graph"]["grid"],
            )
        )
    return out


def names(config):
    return [s["name"] for s in specifications(config)] + list(META)


def pools(config):
    specs = specifications(config)
    out = {site: [s["name"] for s in specs if s["site"] == site] for site in ("oral", "balanced")}
    out["non_tree"] = [s["name"] for s in specs if s["learner"] in ("mars", "gp", "graph")]
    out["new"] = [s["name"] for s in specs]
    out["combined"] = ["bank__" + n for n in config["legacy_members"]] + out["new"]
    return out


def additions(config):
    out = {}
    for learner in ("hgb", "cat", "mars", "gp"):
        for site in ("oral", "balanced"):
            for rep in ("sir", "save"):
                out[learner + "_" + rep + "_" + site] = learner + "_pc_" + site
    for rep in ("pc", "save"):
        for site in ("oral", "balanced"):
            for learner, base in (("bin_cat", "cat"), ("quantile_cat", "cat"), ("bin_hgb", "hgb")):
                out[learner + "_" + rep + "_" + site] = base + "_" + rep + "_" + site
    for kind in ("convex", "ridge", "positive"):
        out[kind + "_balanced"] = kind + "_oral"
        out[kind + "_combined"] = kind + "_new"
    return out
