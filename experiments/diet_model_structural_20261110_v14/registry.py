"""Fixed representation contrasts, complementary regressors and prediction pools."""

REFERENCES = (
    "v11_positive_combined",
    "v11_sparse_combined",
    "v13_positive_combined",
    "v11_replay",
    "v13_replay",
)
POOLS = ("oral", "balanced", "non_tree", "new", "bank", "combined")
NONLINEAR = tuple(k + "_stack_" + p for p in ("bank", "combined") for k in ("local", "ntk", "rbf_ard"))
META = (
    tuple(k + "_" + p for p in POOLS for k in ("convex", "ridge", "positive"))
    + (
        "sparse_new",
        "sparse_combined",
    )
    + NONLINEAR
)


def specifications(config):
    out = []
    for rep in ("sir", "rff_sir", "pilot_gradient"):
        for learner in ("hgb", "cat"):
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
    for rep in ("sir", "rff_sir"):
        for learner in ("local", "forest_local", "ntk", "diffusion", "svr", "rbf_ard"):
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
    for rep in ("paired_pc", "cca"):
        for learner in ("hgb", "cat"):
            out.append(
                dict(
                    name=learner + "_" + rep + "_balanced",
                    learner=learner,
                    representation=rep,
                    site="balanced",
                    context="joint",
                    grid=config["families"][learner]["grid"],
                )
            )
    return out


def bank_names(config):
    return ["bank__" + n for n in config["legacy_members"]] + ["prior__" + n for n in config["prior_members"]]


def names(config):
    return [s["name"] for s in specifications(config)] + list(META)


def pools(config):
    specs = specifications(config)
    out = {site: [s["name"] for s in specs if s["site"] == site] for site in ("oral", "balanced")}
    out["non_tree"] = [
        s["name"] for s in specs if s["learner"] in ("local", "ntk", "diffusion", "svr", "rbf_ard")
    ]
    out["new"], out["bank"] = [s["name"] for s in specs], bank_names(config)
    out["combined"] = out["bank"] + out["new"]
    return out


def additions(config):
    out = {}
    for spec in specifications(config):
        name, rep = spec["name"], spec["representation"]
        if rep in ("rff_sir", "pilot_gradient"):
            out[name] = spec["learner"] + "_sir_" + spec["site"]
        elif rep == "cca":
            out[name] = spec["learner"] + "_paired_pc_balanced"
    for kind in ("convex", "ridge", "positive"):
        out[kind + "_balanced"] = kind + "_oral"
        out[kind + "_combined"] = kind + "_new"
    for pool in ("bank", "combined"):
        for kind in ("local", "ntk", "rbf_ard"):
            out[kind + "_stack_" + pool] = "positive_" + pool
    return out
