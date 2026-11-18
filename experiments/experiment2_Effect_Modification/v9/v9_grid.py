"""Frozen candidate catalogue. Each product is (dose basis, microbial feature)."""

from dataclasses import asdict, dataclass
from itertools import combinations
from math import isfinite

import v7_models as v7
import v7_modifier_views as views
import v8_models as v8


ANCHOR = "pc4_marker_gate_both"
ECO = v8.ECO
PC = v7.pc_list("g", 10)
BASE = PC + ECO
PENALTIES = v8.PENALTIES
TREE_COUNT = 5
TREE_VIEWS = {"gut2": (2, "gut"), "gut3": (3, "gut"), "joint2": (2, "both")}
REGION_VIEWS = {"ecology3": (3, ECO), "gut5": (5, PC[:6] + ECO)}
MATCHED_BLOCKS = ("shannon_gate", "fermentation_gate", "fermentation_dose_curve")
MATCHED_PENALTY = 0.1


def gates(count, markers=ECO, first=1):
    return tuple(f"v7_gate_{m}_g{i}" for m in markers for i in range(first, count + 1))


def tree_labels(name):
    depth, _ = TREE_VIEWS[name]
    return tuple(f"leaf_{name}_{i}" for i in range(TREE_COUNT * (2**depth - 1)))


def region_labels(name):
    return tuple(f"region_{name}_{i}" for i in range(REGION_VIEWS[name][0] - 1))


def is_pc(name):
    return name[0] in "go" and name[1:].isdigit()


@dataclass(frozen=True)
class Recipe:
    name: str
    group: str
    mains: tuple
    terms: tuple
    weights: tuple


def make_grid():
    recipes, signatures, blends, parents = {}, {}, {}, {}

    def add(name, group, terms, weights, *, shared_mains=()):
        # All nonlinear microbial features in the on model get their own main
        # effects in both arms. Raw PCs are already in the frozen CLR span.
        terms, weights = tuple(terms), tuple(float(x) for x in weights)
        mains = list(dict.fromkeys((*shared_mains, *(m for _, m in terms if not is_pc(m)))))
        for d, _ in terms:
            if d in (3, 4) and f"dose_{d}" not in mains:
                mains.append(f"dose_{d}")
        if (
            not terms
            or len(set(terms)) != len(terms)
            or len(terms) != len(weights)
            or any(not isfinite(w) or w <= 0 for w in weights)
        ):
            raise ValueError("Invalid interaction specification: " + name)
        sig = (tuple(sorted(mains)), tuple(sorted(zip(terms, weights, strict=True))))
        if sig in signatures:
            return signatures[sig]
        if name in recipes:
            raise ValueError("Duplicate route name: " + name)
        recipes[name] = Recipe(name, group, tuple(mains), terms, weights)
        signatures[sig] = name
        return name

    def linear(name, group, features, *, weights=None, curves=(), basis=1, curve_weight=3.0):
        weights = {} if weights is None else weights
        return add(
            name,
            group,
            [(0, m) for m in features] + [(basis, m) for m in curves],
            [weights.get(m, 1.0) for m in features] + [curve_weight] * len(curves),
        )

    def mix(name, members):
        members = tuple(dict.fromkeys(members))
        if len(members) < 2 or not all(m in recipes for m in members):
            raise ValueError("Invalid fixed blend: " + name)
        if name in recipes or name in blends or any(set(old) == set(members) for old in blends.values()):
            raise ValueError("Duplicate blend: " + name)
        blends[name] = members
        parents[name] = members

    # Keep the released V8 term order and all eight penalties, including the
    # reference family's original strong shrinkage on the eight gate terms.
    for model in v8.MODELS:
        name = add(model.name, "v8", model.products, model.product_weights())
        if name != model.name:
            raise ValueError("Duplicate V8 structure")
    for name, members in v8.BLENDS.items():
        mix(name, members)

    # Retain unique older feature models alongside the newer gate models.
    # Dose index 2 is the original V7 spline, and 1 the V8 residualized spline.
    old_names = {}
    for route in v7.ROUTES:
        terms = [(0 if d == 0 else 2, m) for d in route.doses for m in route.modifiers]
        old_names[route.name] = add("v7_" + route.name, "v7", terms, v7._weights(route))
    for name, members in v7.BLENDS.items():
        mix("v7_" + name, [old_names[m] for m in members])

    # Gate complexity, PC windows, differential shrinkage, and core rank.
    for k in (1, 2, 3, 4, 6, 8, 10):
        for weight in (1.0, 4.0, 12.0):
            g = gates(k)
            linear(f"gate{k}_w{weight:g}", "gate_geometry", BASE + g, weights=dict.fromkeys(g, weight))
    for k, start in ((4, 3), (6, 3), (8, 5), (10, 7)):
        g = gates(k, first=start)
        linear(f"gate_window_{start}_{k}", "gate_geometry", BASE + g, weights=dict.fromkeys(g, 4.0))
    g4 = gates(4)
    gw = dict.fromkeys(g4, 4.0)
    for rank in (6, 8, 12, 16, 24):
        pcs = v7.pc_list("g", rank)
        linear(
            f"gate4_core{rank}",
            "gate_geometry",
            pcs + ECO + g4,
            weights={**gw, **dict.fromkeys(pcs[10:], 3.0)},
        )
    for sw, fw in ((1.0, 8.0), (8.0, 1.0), (2.0, 6.0), (6.0, 2.0)):
        linear(
            f"gate_asymmetric_{sw:g}_{fw:g}",
            "gate_geometry",
            BASE + g4,
            weights={**dict.fromkeys(gates(4, ECO[:1]), sw), **dict.fromkeys(gates(4, ECO[1:]), fw)},
        )

    # Full 2x2x2 feature factorial: gate4, light ecology and fermentation dose
    # curvature. Four cells already exist in V8. Save actual parents for contrasts.
    light = dict.fromkeys(ECO, 0.35)
    cells = {}
    for size in range(4):
        for parts in combinations(("gate", "light", "curve"), size):
            has_gate = "gate" in parts
            nm = linear(
                "factor_" + "_".join(parts),
                "historical_factorial",
                BASE + (g4 if has_gate else ()),
                weights={**(gw if has_gate else {}), **(light if "light" in parts else {})},
                curves=("fermentation",) if "curve" in parts else (),
            )
            cells[parts] = nm
            if size >= 2:
                parents[nm] = tuple(cells[tuple(x for x in parts if x != drop)] for drop in parts)

    def overlay(name, additions, *, parent=ANCHOR, curves=(), basis=1, weight=3.0):
        """Layer a new feature block on a matched historical parent."""
        par = recipes[parent]
        old = dict(zip(par.terms, par.weights, strict=True))
        old.update({(0, m): weight for m in additions if (0, m) not in old})
        old.update({(basis, m): weight for m in curves if (basis, m) not in old})
        child = add(name, "overlay", old.keys(), old.values())
        if child != parent:
            parents.setdefault(child, (parent,))
        return child

    for tag, features in (
        ("hinges", v8.HINGES),
        ("marker_product", ("shannon_x_fermentation",)),
        ("species_balance", v8.BALANCES),
        ("species_count", ("panel_count",)),
        ("three_log_species", ("log_bromii", "log_adolescentis", "log_rectale")),
        ("gut_dominance", ("gut_dominance", "gut_simpson")),
        ("oral_ecology", ("oral_shannon", "oral_dominance")),
        ("site_joint", v7.pc_list("o", 2) + ("g1_x_o1", "g1_x_o2", "g2_x_o1", "g2_x_o2")),
        ("oral_pc", v7.pc_list("o", 5)),
        ("marker_ranks", ("rank_shannon", "rank_fermentation")),
        ("bounded_markers", ("bounded_shannon", "bounded_fermentation")),
    ):
        for parent, suffix in ((v8.CORE, "core"), (ANCHOR, "gate")):
            overlay(f"{tag}_{suffix}", features, parent=parent)
    for kind, width in views.COMPOSITION_WIDTHS.items():
        overlay("composition_gate_" + kind, views.labels(kind, width))
    for band in views.RFF_BANDS:
        overlay("kernel_gate_" + band, views.labels("rff_" + band, 2 * views.N_FREQUENCIES))
    for tag, features in (("score", ECO), ("gate", g4), ("pc_score", PC[:4] + ECO)):
        overlay("gate_curve_" + tag, (), curves=features)
        for d, side in ((3, "lower"), (4, "upper")):
            overlay("dose_" + side + "_" + tag, (), curves=features, basis=d)
    for marker in ("rank_shannon", "rank_fermentation", "bounded_shannon", "bounded_fermentation"):
        gate = gates(4, (marker,))
        linear(
            "transformed_gate_" + marker,
            "nonlinear",
            BASE + (marker,) + gate,
            weights=dict.fromkeys(gate, 4.0),
        )

    # Standalone alternatives have no gate requirement. Each block also has a
    # corresponding combined model, so a useful older idea can win on its own.
    standalone = {
        "shannon": ("shannon",),
        "fermentation": ("fermentation",),
        "eco": ECO,
        "ferment_balance": ("fermentation",) + v8.BALANCES,
        "dominance": ("gut_dominance", "gut_simpson"),
        "oral": ("oral_shannon", "oral_dominance"),
        "rank": ("rank_shannon", "rank_fermentation"),
        "species": ("log_bromii", "log_adolescentis", "log_rectale"),
        "cross_site": v7.pc_list("g", 4) + v7.pc_list("o", 4),
    }
    for tag, block in standalone.items():
        par = linear("standalone_" + tag, "standalone", block)
        child = overlay("coupled_" + tag, block)
        if child != ANCHOR and child != par:
            parents[child] = tuple(dict.fromkeys((*parents.get(child, (ANCHOR,)), par)))
    for tag, block in (("shannon", ECO[:1]), ("fermentation", ECO[1:]), ("joint", ECO)):
        linear("standalone_dose_curve_" + tag, "standalone", block, curves=block)
        rich = block + tuple(m + "_hinge" for m in block)
        smooth = linear("standalone_score_surface_" + tag, "standalone", rich, curves=rich)
        stacked = overlay("gate_score_surface_" + tag, rich, curves=rich)
        parents[stacked] = (ANCHOR, smooth)
    for tag, features in {
        **{n: tree_labels(n) for n in TREE_VIEWS},
        **{n: region_labels(n) for n in REGION_VIEWS},
    }.items():
        par = linear("partition_" + tag, "partition", features)
        core = overlay("partition_core_" + tag, features, parent=v8.CORE)
        comb = overlay("partition_gate_" + tag, features)
        parents[core] = (v8.CORE, par)
        parents[comb] = (ANCHOR, par)

    # Earlier curves and composition views can also combine without PC4 gates.
    earlier = old_names["gut_pc10_fermentation"]
    for tag, features in (
        ("balance", v8.BALANCES),
        ("rank", ("rank_fermentation",)),
        ("rff", views.labels("rff_broad", 2 * views.N_FREQUENCIES)),
    ):
        overlay("fermentation_" + tag, features, parent=earlier, curves=("fermentation",))
    fixed = {
        "blend_best_v8": (ANCHOR, "ecology_both_light", "fiber_curve_fermentation"),
        "blend_historical_free": (earlier, "ecology_both_light", "fiber_curve_fermentation"),
        "blend_gate_dose": (ANCHOR, "gate_curve_gate", "dose_lower_score", "dose_upper_score"),
        "blend_gate_ecology": (
            ANCHOR,
            "transformed_gate_rank_shannon",
            "transformed_gate_rank_fermentation",
        ),
        "blend_gate_site": (ANCHOR, "site_joint_gate", "oral_ecology_gate"),
        "blend_partitions": ("partition_gut2", "partition_joint2", "partition_ecology3"),
        "blend_gate_sparse_taxa": (ANCHOR, "composition_gate_pivot", "three_log_species_gate"),
        "blend_broad": (
            cells["gate", "light", "curve"],
            "kernel_gate_broad",
            "partition_gate_gut2",
            "composition_gate_pivot",
            earlier,
        ),
    }
    for name, members in fixed.items():
        mix(name, members)

    # Keep the entire V8 gate-model additive background in every cell. PC10 x
    # fiber and the two score x fiber terms stay in all ON arms. Only the three
    # named blocks vary. Two cells coincide with existing V8/V9 recipes.
    matched = []
    base_terms = tuple((0, m) for m in BASE)
    blocks = (
        (tuple((0, m) for m in gates(4, ECO[:1])), 4.0),
        (tuple((0, m) for m in gates(4, ECO[1:])), 4.0),
        (((1, "fermentation"),), 3.0),
    )
    for mask in range(2 ** len(blocks)):
        terms, weights = list(base_terms), [1.0] * len(base_terms)
        included = []
        for j, (block_terms, penalty_weight) in enumerate(blocks):
            if mask & (1 << j):
                terms.extend(block_terms)
                weights.extend([penalty_weight] * len(block_terms))
                included.append(MATCHED_BLOCKS[j])
        name = add(
            "common_main_" + ("_".join(included) or "core"),
            "common_main",
            terms,
            weights,
            shared_mains=recipes[ANCHOR].mains,
        )
        matched.append(name)
        direct = tuple(matched[mask ^ (1 << j)] for j in range(len(blocks)) if mask & (1 << j))
        if direct:
            parents[name] = tuple(dict.fromkeys((*parents.get(name, ()), *direct)))
    if len(set(matched)) != len(matched) or matched[3] != ANCHOR:
        raise ValueError("Shared-baseline cells do not reproduce the PC4 anchor")

    # Shrink only the fitted difference in predictions, leaving the parent's off
    # arm exactly intact. Coefficient regularization remains a separate axis.
    attenuation = {}
    for parent in (
        ANCHOR,
        v8.CORE,
        "ecology_both_light",
        "fiber_curve_fermentation",
        cells["gate", "light", "curve"],
        old_names["gut_pc10_ridge"],
    ):
        for percent in (25, 50, 75):
            name = f"retain{percent}__{parent}"
            attenuation[name] = (parent, percent / 100.0)
            parents[name] = (parent,)

    # These learners share a parent's exact baseline, but estimate its
    # interaction coefficients by sparse, spectral or supervised low-rank fits.
    alternatives = {}
    for parent in (
        ANCHOR,
        v8.CORE,
        earlier,
        cells["gate", "light", "curve"],
        "composition_gate_pivot",
        "partition_core_gut2",
        "kernel_gate_broad",
        "site_joint_gate",
    ):
        for method in ("elastic", "pcr", "krylov"):
            child = method + "__" + parent
            alternatives[child] = (method, parent)
            parents[child] = (parent,)
    return recipes, blends, alternatives, parents, old_names, tuple(matched), attenuation


RECIPES, BLENDS, ALTERNATIVES, PARENTS, HISTORICAL, MATCHED_CELLS, ATTENUATION = make_grid()
FAMILIES = (*RECIPES, *BLENDS, *ALTERNATIVES, *ATTENUATION)
OPTIONS = {"elastic": (0.8, 0.45, 0.2, 0.08, 0.02), "pcr": (1, 2, 4, 8), "krylov": (1, 2, 3, 5)}
ALT_RIDGE = 0.15
SETTINGS = {n: OPTIONS[ALTERNATIVES[n][0]] if n in ALTERNATIVES else PENALTIES for n in FAMILIES}
KEYS = tuple(v7.setting_key(n, p) for n, pp in SETTINGS.items() for p in pp)
FIXED_PC10 = v7.setting_key(HISTORICAL["gut_pc10_ridge"], 0.1)


def design():
    return dict(
        ridge_models=[asdict(m) for m in RECIPES.values()],
        fixed_blends=BLENDS,
        alternative_learners=ALTERNATIVES,
        alternative_parameters=OPTIONS,
        alternative_ridge=ALT_RIDGE,
        comparison_parents=PARENTS,
        shared_main_ablation=dict(
            blocks=MATCHED_BLOCKS,
            cells_by_bitmask=MATCHED_CELLS,
            fixed_penalty=MATCHED_PENALTY,
            common_mains=RECIPES[ANCHOR].mains,
        ),
        retained_interaction_fraction=ATTENUATION,
        interaction_penalties=PENALTIES,
        trees=TREE_VIEWS,
        trees_per_view=TREE_COUNT,
        regions=REGION_VIEWS,
    )
