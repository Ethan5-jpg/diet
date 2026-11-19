"""Fixed candidate library and training rules for the protected anchor."""

from design import BASE, NAMES as V2_NAMES, SHARED_MODES, VIEWS as V2_VIEWS


VERSION = "hpp_full_pheno_v3"
V2_MANIFEST_SHA = "9a248b92228760fe6d42e4fcb9a3c8e5738fc9b715afb811ec2e73adda715bf7"
ANCHOR = ("fiber_density", "D_Glu")
ANCHOR_RECIPE = "v1_fusion"
PC5_REFERENCE = "gut_pc5/linear/ridge_1"
WORKFLOWS = ("fixed_pc5", "select", "select_guarded", "mean_top3", "blend",
             "blend_guarded", "domain_blend", "domain_partial")
PRIMARY = "blend_guarded"
GUARD_WIN_FOLDS = 3
GUARD_GAIN = 0.0005
TOP_K = 8
BLEND_PENALTY = .10
DOMAIN_WEIGHT = .50
MIN_GAIN_PP = .10
LARGE_GAIN_PP = .25
EPS_PP = 1e-9

NEW_VIEWS = ("oral_pc2", "gut_pc2", "both_pc2", "oral_shannon", "gut_ecology", "gut_shannon_pc2")
MORE_V2_PENALTIES = (.03, .3, 3., 10.)
NARROW_PENALTIES = (.03, .1, 1., 3.)
DOMAIN_VIEWS = ("both_pc5", "gut_pc5", "panel_shannon")
DOMAIN_RIDGE = (.03, .3)
DOMAIN_ADAPT = (None, .1, 1.)


def ridge_name(view, dose, penalty):
    return f"{view}/{dose}/ridge_{penalty:g}"


def domain_name(view, dose, penalty, adapt):
    suffix = "pooled" if adapt is None else f"partial_{adapt:g}"
    return f"domain_{view}/{dose}/prior_{penalty:g}/{suffix}"


RIDGES = tuple((v, s, p) for views, penalties in ((V2_VIEWS, MORE_V2_PENALTIES),
                                                  (NEW_VIEWS, NARROW_PENALTIES))
               for v in views for s in SHARED_MODES for p in penalties)
DOMAIN_SETTINGS = tuple((v, s, p, a) for v in DOMAIN_VIEWS for s in SHARED_MODES
                        for p in DOMAIN_RIDGE for a in DOMAIN_ADAPT)
DOMAIN_NAMES = tuple(domain_name(*setting) for setting in DOMAIN_SETTINGS)
LIBRARY = (*V2_NAMES, *(ridge_name(*setting) for setting in RIDGES), *DOMAIN_NAMES)
LIBRARY_INDEX = {name: col for col, name in enumerate(LIBRARY)}


def representation(recipe):
    """Group nearly duplicate settings before averaging or weighting."""
    if recipe == "v1_panel":
        return "panel"
    if recipe == "v1_fusion":
        return "panel_shannon"
    return recipe.split("/", 1)[0]
