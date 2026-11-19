"""Frozen candidate settings and workflow choices for Full Pheno v4."""

import v3_design as previous


VERSION = "hpp_full_pheno_v4"
V3_MANIFEST_SHA = "967ff132c7f8a80022262775fcb0ae809f6f1c88552c2c48b555da9dad72c88d"
ANCHOR = previous.ANCHOR
ANCHOR_RECIPE = previous.ANCHOR_RECIPE
PC5 = previous.PC5_REFERENCE
ORAL = "oral_shannon/curve/ridge_0.03"
REFERENCE = "v3_fixed_pc5"
PRIMARY = "pc5_capped_mix"
WORKFLOWS = (
    REFERENCE, "v3_guarded", "uniform_pc5", "uniform_oral", "pc5_half", "pc5_quarter",
    "fixed_two_site", "core_calibrated", "new_block_guarded", "new_block_maximin",
    PRIMARY, "peer_direction", "peer_checked",
)
UNIFORM = ("uniform_pc5", "uniform_oral")
CORE = (
    PC5, "gut_pc5/linear/ridge_0.3", "gut_pc2/linear/ridge_1", "both_pc2/linear/ridge_1",
    ORAL, "panel_shannon/linear/ridge_1", "gut_ecology/linear/ridge_1",
    "v1_panel", "oral_pc5/linear/ridge_1", "gut_shannon_pc2/linear/ridge_1",
)
VIEWS = ("oral_shannon", "shannon", "panel_shannon", "oral_pc2",
         "gut_pc2", "gut_pc5", "both_pc5", "gut_ecology")
CURVES = tuple((view, penalty, ratio) for view in VIEWS for penalty in (.03, .15, .6, 2.)
               for ratio in (4., 16.))
ROBUST = tuple((view, dose, penalty) for view in VIEWS for dose in ("linear", "curve")
               for penalty in (.03, .3, 1.))


def curve_name(view, penalty, ratio):
    return f"curve_block/{view}/ridge_{penalty:g}/curve_x{ratio:g}"


def robust_name(view, dose, penalty):
    return f"robust/{view}/{dose}/ridge_{penalty:g}"


NEW = (*(curve_name(*setting) for setting in CURVES), *(robust_name(*setting) for setting in ROBUST))
LIBRARY = (*previous.LIBRARY, *NEW)
INDEX = {name: i for i, name in enumerate(LIBRARY)}
MIX_CORE = (*CORE, *(robust_name(view, "linear", .3) for view in ("oral_shannon", "gut_pc5", "panel_shannon")))
MIX_FRACTIONS = (0., .15, .35)
SHRINK_PENALTY = .10
MIN_INNER_GAIN = .0003
INNER_WINS = 3
RATIO_DIET_FLOOR = .001
MIN_GAIN_PP = .10
LARGER_GAIN_PP = .25
EPS_PP = 1e-9
