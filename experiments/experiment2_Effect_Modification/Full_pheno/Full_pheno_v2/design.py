"""Frozen discovery design. Same recipes and penalties across all 18 tasks."""
from dataclasses import dataclass

VERSION = "hpp_full_pheno_v2"
V1_MANIFEST_SHA = "178baea01bf0a6a13fc5f166e06a4e332038e075d8eead8285a28ddf269c01e4"
MIN_GAIN = .001     # 0.10 R2 percentage points
LARGER_GAIN = .0025
SHARED_PENALTY = .03
SHARED_ADAPT = .10
SHARED_MODES = ("linear", "curve")
BASE = "v1_off"
REFS = ("v1_panel", "v1_fusion")
VIEWS = ("panel", "shannon", "panel_shannon", "ecology", "oral_pc5", "gut_pc5",
         "both_pc5", "panel_shannon_both_pc5")
PENALTIES = (.01, .10, 1.)
REPLAY_ATOL = 2e-8
DOMAIN = {"D_Glu": "glucose", "D_TotalFat": "adiposity", "D_VAT": "adiposity"}


@dataclass(frozen=True)
class Recipe:
    view: str
    dose: str
    penalty: float

    @property
    def name(self):
        return f"{self.view}/{self.dose}/ridge_{self.penalty:g}"


RECIPES = tuple(Recipe(v, d, penalty) for v in VIEWS for d in SHARED_MODES for penalty in PENALTIES)
SHARED = tuple(f"joint_both_pc5/{d}/{name}" for d in SHARED_MODES for name in ("pooled", "partial"))
NAMES = (*REFS, *(r.name for r in RECIPES), *SHARED)


def domain(endpoint):
    if endpoint == "D_Glu" or endpoint in ("D_TotalFat", "D_VAT"):
        return DOMAIN[endpoint]
    if endpoint.startswith("D_Liver_"):
        return "liver"
    raise ValueError("UNKNOWN_PHENOTYPE_DOMAIN")
