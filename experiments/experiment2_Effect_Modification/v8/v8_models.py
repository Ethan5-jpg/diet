"""Matched fiber-interaction models around the V7 gut PC10 + ecology baseline."""
from dataclasses import asdict, dataclass

import numpy as np

import v7_models as legacy
from adapters.fermentation_score import MASS_UNIT, resolve_species


ECO = ("shannon", "fermentation")
CORE = "pc10_ecology"
PENALTIES = (100., 10., 3., 1., .3, .1, .03, .01)
BALANCES = ("panel_rb", "panel_rb_e")
HINGES = ("shannon_hinge", "fermentation_hinge")


@dataclass(frozen=True)
class Model:
    name: str
    rank: int = 10
    # Weights multiply the interaction penalty, never the common main penalty.
    eco_weights: tuple = (1., 1.)
    tail_after: int = 10
    tail_weight: float = 1.
    extra: tuple = ()
    extra_weight: float = 2.
    curved: tuple = ()
    curve_weight: float = 3.

    @property
    def modifiers(self):
        return legacy.pc_list("g", self.rank) + ECO + self.extra

    @property
    def products(self):
        return (tuple((0, label) for label in self.modifiers)
                + tuple((1, label) for label in self.curved))

    def product_weights(self):
        linear = [1. if j <= self.tail_after else self.tail_weight for j in range(1, self.rank + 1)]
        return np.array(linear + list(self.eco_weights) + [self.extra_weight] * len(self.extra)
                        + [self.curve_weight] * len(self.curved))


def make_models():
    routes = [Model(CORE)]
    for name, weights in (("both_light", (.35, .35)), ("both_strong", (3., 3.)),
                          ("shannon_light", (.35, 2.)), ("fermentation_light", (2., .35))):
        routes.append(Model("ecology_" + name, eco_weights=weights))
    routes.extend((
        Model("pc_tail_soft", tail_after=5, tail_weight=2.),
        Model("pc_tail_strong", tail_after=5, tail_weight=5.),
        Model("pc_tail_ecology_light", tail_after=5, tail_weight=3., eco_weights=(.5, .5)),
        Model("pc12_ecology", rank=12, tail_weight=3.),
        Model("pc16_ecology", rank=16, tail_weight=5.),
        Model("fiber_curve_shannon", curved=ECO[:1]),
        Model("fiber_curve_fermentation", curved=ECO[1:]),
        Model("fiber_curve_ecology", curved=ECO),
        Model("fiber_curve_pc3_ecology", curved=legacy.pc_list("g", 3) + ECO),
        Model("fiber_curve_pc10_ecology", curved=legacy.pc_list("g", 10) + ECO, curve_weight=6.),
        Model("marker_hinge_shannon", extra=HINGES[:1]),
        Model("marker_hinge_fermentation", extra=HINGES[1:]),
        Model("marker_hinges", extra=HINGES),
        Model("marker_product", extra=("shannon_x_fermentation",)),
        Model("marker_surface", extra=HINGES + ("shannon_x_fermentation",), extra_weight=4.),
    ))
    for k, markers, name in ((2, ECO[:1], "shannon"), (2, ECO[1:], "fermentation"),
                             (2, ECO, "both"), (4, ECO, "both")):
        products = tuple(f"v7_gate_{marker}_g{j}" for marker in markers for j in range(1, k + 1))
        routes.append(Model(f"pc{k}_marker_gate_{name}", extra=products, extra_weight=4.))
    routes.extend((
        Model("three_species_balance", extra=BALANCES),
        Model("three_species_detection_count", extra=("panel_count",)),
        Model("three_species_balance_count", extra=BALANCES + ("panel_count",), extra_weight=3.),
        Model("three_species_balance_fiber_curve", extra=BALANCES, curved=ECO),
    ))
    return tuple(routes)


MODELS = make_models()
# Both arms mix the same fitted parent models with the same weights and penalty.
BLENDS = {
    "blend_shrinkage": (CORE, "pc_tail_ecology_light", "ecology_fermentation_light"),
    "blend_dose": (CORE, "fiber_curve_ecology", "fiber_curve_pc3_ecology"),
    "blend_shapes": (CORE, "marker_surface", "pc2_marker_gate_both"),
    "blend_panel": (CORE, "three_species_balance", "three_species_balance_count"),
    "blend_cross_family": (CORE, "pc_tail_soft", "fiber_curve_ecology",
                           "marker_product", "three_species_balance"),
}
FAMILIES = tuple(m.name for m in MODELS) + tuple(BLENDS)
KEYS = tuple(legacy.setting_key(name, penalty) for name in FAMILIES for penalty in PENALTIES)


def design():
    return dict(models=[asdict(model) for model in MODELS], equal_weight_parents=BLENDS,
                penalties=PENALTIES, main_penalty=legacy.MAIN_PENALTY,
                three_species_log_offset=MASS_UNIT)


def centered(train, valid):
    location = train.mean(axis=0)
    return train - location, valid - location


def background_and_features(source, split):
    """Training-fitted coordinates. Validation targets never enter this function."""
    bank = legacy._feature_bank(source, split)
    for marker, label in zip(ECO, HINGES, strict=True):
        a, b = bank[marker]
        knot = np.median(a)
        bank[label] = centered(np.maximum(a - knot, 0.), np.maximum(b - knot, 0.))
    bank["shannon_x_fermentation"] = centered(
        *(s * f for s, f in zip(bank["shannon"], bank["fermentation"], strict=True)))

    matches = resolve_species(source.taxa["gut"])
    columns = [matches[species][0] for species in
               ("Ruminococcus bromii", "Bifidobacterium adolescentis", "Eubacterium rectale")]
    panel_views = []
    for rows in (split.train_rows, split.rows):
        raw = source.raw["gut"][np.ix_(rows, columns)]
        r, b, e = np.log1p(raw / MASS_UNIT).T
        panel_views.append(((r - b) / np.sqrt(2.), (r + b - 2. * e) / np.sqrt(6.),
                            (raw > 0).sum(axis=1)))
    for label, pair in zip(BALANCES + ("panel_count",), zip(*panel_views, strict=True), strict=True):
        bank[label] = centered(*pair)

    # Separate the nonlinear dose basis from its training intercept and linear part.
    dose_linear = [np.column_stack((np.ones(len(d)), d[:, 0])) for d in (split.dt, split.dv)]
    coefficients = np.linalg.lstsq(dose_linear[0], split.dt[:, 1], rcond=None)[0]
    curve = [d[:, 1] - x @ coefficients
             for d, x in zip((split.dt, split.dv), dose_linear, strict=True)]
    bases, doses = [], []
    main_pcs = legacy.pc_list("o", 5) + legacy.pc_list("g", 5)
    for side, (host, dose) in enumerate(((split.ht, split.dt), (split.hv, split.dv))):
        bases.append(np.column_stack((np.ones(len(host)), host, dose, *(bank[p][side] for p in main_pcs))))
        doses.append((dose[:, 0], curve[side]))
    return bank, bases, doses


def evaluate_split(source, split):
    """Return each setting's own refitted off and on predictions, in row order."""
    if np.intersect1d(source.groups[split.train_rows], source.groups[split.rows]).size:
        raise ValueError("Household crosses a model training role")
    bank, bases, doses = background_and_features(source, split)
    marginals = tuple(dict.fromkeys(p for model in MODELS for p in ECO + model.extra))
    products = tuple(dict.fromkeys(p for model in MODELS for p in model.products))
    main_index = {p: i for i, p in enumerate(marginals)}
    interaction_index = {p: i + len(marginals) for i, p in enumerate(products)}
    columns = []
    for side in range(2):
        main = [bank[p][side] for p in marginals]
        interaction = [doses[side][d] * bank[p][side] for d, p in products]
        columns.append(np.column_stack(main + interaction))
    profile = legacy._Profile(*bases, np.column_stack((split.ot, split.gt)),
                              np.column_stack((split.ov, split.gv)), split.yt, *columns)
    nulls, pairs = {}, {}
    for model in MODELS:
        extra = tuple(main_index[p] for p in ECO + model.extra)
        if extra not in nulls:
            nulls[extra] = profile.prediction(list(extra), legacy.MAIN_PENALTY)
        on_indices = [*extra, *(interaction_index[p] for p in model.products)]
        weights = model.product_weights()
        for penalty in PENALTIES:
            penalties = np.r_[np.full(len(extra), legacy.MAIN_PENALTY), penalty * weights]
            pairs[legacy.setting_key(model.name, penalty)] = nulls[extra], profile.prediction(on_indices, penalties)
    for name, parents in BLENDS.items():
        for penalty in PENALTIES:
            members = [pairs[legacy.setting_key(parent, penalty)] for parent in parents]
            pairs[legacy.setting_key(name, penalty)] = tuple(np.mean(arm, axis=0) for arm in zip(*members))
    return pairs
