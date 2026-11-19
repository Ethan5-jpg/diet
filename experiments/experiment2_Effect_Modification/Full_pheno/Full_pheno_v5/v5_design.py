"""Prespecified v5 grid and workflow rules."""

import v4_design as v4


VERSION = "hpp_full_pheno_v5"
V4_MANIFEST_SHA = "470d8782caeafcfd59b9e02ffa39e0c8cbb81f82c6b0adec105a54b5ad955099"
ANCHOR = v4.ANCHOR
ANCHOR_RECIPE = v4.ANCHOR_RECIPE
PC5 = v4.PC5
REFERENCE = "v4_pc5_half"
PRIMARY = "add_core"
ORAL_CURVE = v4.curve_name("oral_shannon", .03, 4.)
STRONG_PC5 = "gut_pc5/linear/ridge_3"
WORKFLOWS = (
    REFERENCE, "v4_fixed_pc5", "v4_maximin", "pc5_035", "pc5_065", "pc5_080",
    "fixed_oral_mix", "fixed_tensor_mix", PRIMARY, "add_tensor", "add_all_four",
    "diet_shared", "domain_shared", "domain_veto",
)
REPLAY = {REFERENCE: "pc5_half", "v4_fixed_pc5": "v3_fixed_pc5", "v4_maximin": "new_block_maximin"}
VIEWS = ("oral_shannon", "shannon", "panel", "gut_pc2", "gut_pc5", "both_pc2")
SHAPES = ("dose", "marker", "joint")
CENTERS = ("off", "half")
SETTINGS = (("ridge", .12), ("ridge", .6), ("robust", .3))


def name(view, shape, center, fit, penalty):
    return f"tensor/{view}/{shape}/{center}/{fit}_{penalty:g}"


NEW = tuple(name(v, s, c, f, p) for v in VIEWS for s in SHAPES for c in CENTERS for f, p in SETTINGS)
LIBRARY = (*v4.LIBRARY, *NEW)
INDEX = {r: j for j, r in enumerate(LIBRARY)}
FIXED_TENSOR = name("gut_pc2", "dose", "half", "ridge", .6)
CORE = (*v4.CORE, STRONG_PC5, ORAL_CURVE,
        *(name(view, shape, "half", "ridge", .6) for view in VIEWS for shape in ("dose", "marker")))
CAP = .50
MIN_RELATIVE = .0002
EXTRA_PENALTY = 4.
MIN_GAIN_PP = v4.MIN_GAIN_PP
EPS_PP = v4.EPS_PP
