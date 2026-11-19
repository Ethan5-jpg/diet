"""The prespecified full phenotype v1 design."""

from pathlib import Path


VERSION = "hpp_full_pheno_v1"
HERE = Path(__file__).resolve().parent
HANDOFF = "collaborator_handoff_20261102_v1"
INPUTS = "inputs/discovery_v2"
CACHE = "preprocessing/folds"
MANIFEST = "manifests/payload_sha256_final.json"
INDEX = f"{INPUTS}/discovery_covariates_food_TRE_ONLY.parquet"
ROLES = f"{INPUTS}/roles_v2_full_TRE_ONLY.parquet"
FLOW = "manifests/preprocessed_flow.json"
KEY = ("participant_id", "cohort", "research_stage")
FOLDS = tuple(range(5))
ROLE_NAMES = ("discovery", "confirmation_reserved")
EXPECTED_ROLE_COUNTS = ((4427, 3989), (1878, 1710))
EXPECTED_GLU_FOLDS = (661, 677, 668, 665, 672)
EXPECTED_GLU_HOUSEHOLDS = 3059
PINS = {"numpy": "1.26.4", "scipy": "1.11.4", "pyarrow": "15.0.2"}

ENDPOINTS = {
    "D_Glu": "Mean CGM glucose",
    "D_TotalFat": "Total fat",
    "D_VAT": "Visceral adipose tissue",
    "D_Liver_elasticity": "Liver elasticity",
    "D_Liver_velocity": "Liver shear wave velocity",
    "D_Liver_viscosity": "Liver viscosity",
    "D_Liver_attenuation": "Liver attenuation",
    "D_Liver_dispersion": "Liver dispersion",
    "D_Liver_speed_of_sound": "Liver speed of sound",
}
EXPOSURES = {
    "fiber_density": ("n_fiber_days", "mean_daily_energy_fiber", "fiber"),
    "UPF": ("n_energy_days", "mean_daily_energy_UPF", "UPF"),
}
INDEX_COLS = (*KEY, "input_row", "split_unit_id", "cv_fold", "final_role",
              "age_calendar_years", "sex_code", "smoking_status", "fiber_density",
              "n_fiber_days", "n_energy_days", "mean_daily_energy_fiber", "mean_daily_energy_UPF")
SPECIES = {
    "Ruminococcus bromii": ("ruminococcus_bromii",),
    "Bifidobacterium adolescentis": ("bifidobacterium_adolescentis",),
    "Eubacterium rectale": ("eubacterium_rectale", "agathobacter_rectalis"),
}
PANEL_UNIT = 1e-4
MARKERS = ("three_species", "gut_shannon")
PREDICTORS = ("off", "three_species", "three_species_plus_shannon")
FIBER_DOSE = (10.0, 15.0)
FIBER_KNOT = 12.5
UPF_DOSE = (0.20, 0.30)
PANEL_PENALTY = (0.01, 0.01, 0.03, 0.03)
SHANNON_PENALTY = (0.18, 0.54)
N_PC = 5
RANK_TOL = 1e-10
R2_EPS = 1e-8
BOOTSTRAP_DRAWS = 9999
BOOTSTRAP_SEED = 20261118
ALPHA = 0.05
# Two fixed contrasts across the nine outcomes, separately for each exposure.
FAMILY_TESTS = 2 * len(ENDPOINTS)
V10_SETTING = "total_plus_omit_bromii@0.01"
V10_DESIGN_SHA = "3d985cfd432756093d71994bb3b805a8ed26206a0beac35887fea91f4832d86d"
MODES = {
    "bridge": (("fiber_density", "D_Glu"),),
    "fiber": tuple(("fiber_density", e) for e in ENDPOINTS),
    "upf": tuple(("UPF", e) for e in ENDPOINTS),
    "all": tuple((x, e) for x in EXPOSURES for e in ENDPOINTS),
}


def outcome_file(endpoint):
    return f"{INPUTS}/{endpoint}_raw_Y_TRE_ONLY.parquet"


def fold_file(fold):
    return f"{CACHE}/{'full' if fold is None else f'fold{fold}'}/transformed_discovery_TRE_ONLY.npz"


def upf_metadata(fold):
    return f"{CACHE}/{'full' if fold is None else f'fold{fold}'}/diet_basis.json"
