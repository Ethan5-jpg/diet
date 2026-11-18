"""Frozen design for paired discovery prediction with a 10 percent target."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
from math import isclose, isfinite

TARGET = 10.0
PENALTIES = (3.0, 1.0, 0.3, 0.1, 0.05, 0.03, 0.015, 0.01, 0.005, 0.003, 0.001)
LEGACY_PENALTIES = (100.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)
ANCHOR = "standalone_score_surface_fermentation"
REFERENCE = ANCHOR + "@0.01"
PRIMARY = "ensemble_knot_neighborhood@0.01"
ADAPTIVE = ("inner_maximin", "inner_anchor_guard", "inner_diverse_blend")
F, H, S, SH = "fermentation", "fermentation_hinge", "shannon", "shannon_hinge"
U, C = "fiber", "legacy_curve"
CORE_TERMS = ((U, F), (U, H), (C, F), (C, H))
BLOCKS = ("linear_fiber_by_score_bend", "curved_fiber_by_score", "curved_fiber_by_score_bend")
KNOTS = (20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80)
PANEL_VIEWS = ("omit_bromii", "omit_adolescentis", "omit_rectale", "mean_logs", "max_log", "root_mean")
MEMBERS = ("bromii", "adolescentis", "rectale")
ALT_OPTIONS = {"elastic": (0.8, 0.45, 0.2, 0.08, 0.02), "pcr": (1, 2, 3, 4)}
SELECTION_OFF_TOLERANCE = 0.002
SEED = 101811
# A crossed comparison around the three-species anchor. The dose hinge and all
# marker main effects are shared even in the cell with neither extra interaction.
CROSS_MARKERS = {"composition": ("panel_rb", "panel_rb_e"), "diversity": (S, SH)}
CROSS_CELLS = {
    kind: tuple(f"cross_{kind}_{cell}" for cell in ("base", "dose", "marker", "both"))
    for kind in CROSS_MARKERS
}
CROSS_PENALTY = 0.01


def key(name, value):
    return f"{name}@{value:g}"


def hinge(q):
    return H if q == 50 else f"f_h{q}"


@dataclass(frozen=True)
class Recipe:
    group: str
    mains: tuple
    terms: tuple
    weights: tuple


def build():
    fits, signatures, blends, fractions, alternatives = {}, {}, {}, {}, {}

    def register(n, group, terms, weights, background=()):
        terms, weights = tuple(terms), tuple(map(float, weights))
        # Raw g1..g4 already have unpenalized main effects in B.
        mains = list(background)
        for dose, marker in terms:
            if marker not in ("g1", "g2", "g3", "g4") and marker not in mains:
                mains.append(marker)
            if dose not in (U, C) and dose not in mains:
                mains.append(dose)
        if (
            not terms
            or len(set(terms)) != len(terms)
            or len(terms) != len(weights)
            or not all(isfinite(x) and x > 0 for x in weights)
        ):
            raise ValueError(f"Invalid products in {n}")
        if len(set(mains)) != len(mains):
            raise ValueError("Duplicate shared main term")
        signature = tuple(sorted(mains)), tuple(sorted(zip(terms, weights, strict=True)))
        if n in fits or signature in signatures:
            raise ValueError(f"Duplicate recipe {n} and {signatures.get(signature)}")
        fits[n] = Recipe(group, tuple(mains), terms, weights)
        signatures[signature] = n
        return n

    def surface(n, group, markers, doses=(U, C), *, background=(), extra=()):
        terms = tuple(product(doses, markers)) + tuple((d, m) for d, m, _ in extra)
        weights = [1.0 if d == U else 3.0 for d, m in product(doses, markers)] + [w for d, m, w in extra]
        return register(n, group, terms, weights, background)

    def blend(n, pairs):
        pairs = tuple(pairs)
        if (
            n in blends
            or len(set(x for x, _ in pairs)) != len(pairs)
            or any(x not in fits or w <= 0 for x, w in pairs)
            or not isclose(sum(w for _, w in pairs), 1)
        ):
            raise ValueError("Invalid paired ensemble " + n)
        blends[n] = pairs

    surface(ANCHOR, "anchor", (F, H))
    surface("legacy_joint_surface", "anchor", (S, F, SH, H))
    surface("legacy_shannon_surface", "anchor", (S, SH))
    for q in KNOTS:
        if q != 50:
            surface(f"score_knot_{q}", "score_shape", (F, hinge(q)))
    for qs in ((20, 80), (30, 70), (35, 65), (40, 60), (30, 50, 70), (20, 50, 80)):
        surface("score_knots_" + "_".join(map(str, qs)), "score_shape", (F, *(hinge(q) for q in qs)))
    for q, w in product((35, 50, 65), (10, 25, 50)):
        surface(f"score_soft_{q}_{w}", "score_shape", (F, f"f_soft{q}_{w}"))
    for name, labels in (
        ("bounded", (F, "f_sat")),
        ("rank", ("f_rank", "f_rank_h")),
        ("rank_augmented", (F, H, "f_rank")),
        ("rbf_center", (F, "f_rbf50")),
        ("rbf_three", (F, "f_rbf25", "f_rbf50", "f_rbf75")),
        ("natural_three", (F, "f_rc3_0")),
        ("natural_four", (F, "f_rc4_0", "f_rc4_1")),
        ("winsor", ("f_winsor", "f_winsor_h")),
    ):
        surface("score_" + name, "score_shape", labels)

    for ds, doses in DOSE_SETS.items():
        for ms, markers in (
            ("median", (F, H)),
            ("soft", (F, "f_soft50_25")),
            ("two", (F, hinge(35), hinge(65))),
        ):
            surface(f"dose_{ds}__{ms}", "dose_shape", markers, doses)
    for dc, bend in product((0.3, 1.0, 3.0, 9.0), (0.25, 1.0, 4.0, 12.0)):
        if (dc, bend) != (3.0, 1.0):
            register(
                f"anisotropic_d{dc:g}_s{bend:g}",
                "differential_shrinkage",
                CORE_TERMS,
                (1, bend, dc, bend * dc),
            )
    for d, w in (("legacy_light", 0.25), ("legacy_heavy", 9), ("strong_bend", 20)):
        register("corner_" + d, "differential_shrinkage", CORE_TERMS, (1, 1, 3, w))

    for v in PANEL_VIEWS:
        surface("panel_" + v, "panel_composition", (v, v + "_h"))
        surface("total_plus_" + v, "panel_composition", (F, H), extra=((U, v, 4), (C, v, 8)))
    for species in MEMBERS:
        surface("member_" + species, "panel_composition", (species, species + "_h"))
    for name, extra in (
        ("members", MEMBERS),
        ("balances", ("panel_rb", "panel_rb_e")),
        ("detect", ("detect_any", "panel_count")),
        ("detect_members", tuple("present_" + s for s in MEMBERS)),
        ("evenness", ("panel_evenness",)),
        ("dominance", ("panel_dominance",)),
        ("shannon", (S,)),
        ("shannon_gate", (S, "shannon_x_fermentation")),
        ("gut_pcs", ("g1", "g2", "g3", "g4")),
        ("pc_gates", ("g1", "g2", "f_gate_g1", "f_gate_g2")),
        ("shannon_balance", (S, "panel_rb", "panel_rb_e")),
    ):
        # Triple products retain each constituent's own diet interaction. The
        # microbial pair itself also enters as a shared additive coordinate.
        for wt in (1.0, 6.0):
            surface(
                f"overlay_{name}_{wt:g}",
                "ecological_overlay",
                (F, H),
                extra=tuple((d, m, wt * (1 if d == U else 3)) for d, m in product((U, C), extra)),
            )
    for kind, views in (
        ("members", MEMBERS),
        ("balances", ("panel_rb", "panel_rb_e")),
        ("occupancy", tuple("present_" + s for s in MEMBERS)),
    ):
        surface("only_" + kind, "panel_composition", views)

    # Exact same additive fit in all eight cells. Mask 7 is the released V9 anchor.
    cells = []
    for mask in range(8):
        if mask == 1:
            cells.append("dose_linear__median")
            continue
        if mask == 7:
            cells.append(ANCHOR)
            continue
        keep = [0] + [j + 1 for j in range(3) if mask & (1 << j)]
        cells.append(
            register(
                f"common_anchor_cell_{mask}",
                "common_main_ablation",
                [CORE_TERMS[j] for j in keep],
                [(1, 1, 3, 3)[j] for j in keep],
                (F, H),
            )
        )

    # Compare shapes again with precisely the same F, bend and dose main effects.
    shared = (F, H, hinge(35), hinge(65), "f_soft50_25", "d_rc4_0", "d_rc4_1")
    for name, mk, d in (
        ("anchor", (F, H), (U, C)),
        ("two", (F, hinge(35), hinge(65)), (U, C)),
        ("soft", (F, "f_soft50_25"), (U, C)),
        ("dose", (F, H), DOSE_SETS["rc4"]),
        ("both", (F, hinge(35), hinge(65)), DOSE_SETS["rc4"]),
    ):
        surface("controlled_" + name, "common_background", mk, d, background=shared)

    for kind, markers in CROSS_MARKERS.items():
        main = (F, H, "d_h35", *markers)
        for mask, name in enumerate(CROSS_CELLS[kind]):
            # bit 0: lower fiber-range bend x original score; bit 1: ecology overlay.
            # No third block turns on only in the combined cell.
            dose_block = [("d_h35", m, 3.0) for m in (F, H)] if mask & 1 else []
            ecology_block = (
                [(d, m, 2.0 if d == U else 6.0) for d, m in product((U, C), markers)] if mask & 2 else []
            )
            surface(name, "crossed_blocks", (F, H), background=main, extra=dose_block + ecology_block)

    for n in (
        ANCHOR,
        "score_soft_50_25",
        "score_knots_35_65",
        "dose_rc4__median",
        "overlay_members_6",
        "overlay_shannon_6",
    ):
        for method in ALT_OPTIONS:
            alternatives[method + "__" + n] = (method, n)
    for name in (ANCHOR, "score_soft_50_25", "dose_rc4__median"):
        for fraction in (0.65, 0.85):
            fractions[f"shrink_{int(fraction*100)}__" + name] = (name, fraction)
    for suffix, left, right in (
        ("knot_neighborhood", "score_knot_40", "score_knot_60"),
        ("smoothness", "score_soft_50_10", "score_soft_50_50"),
        ("dose", "dose_rc3__median", "dose_h50__median"),
        ("composition", "total_plus_mean_logs", "overlay_balances_6"),
        ("microbial", "overlay_shannon_6", "overlay_pc_gates_6"),
        ("joint_shape", "dose_rc4__soft", "score_natural_four"),
        ("kernels", "score_rbf_three", "dose_rbf__median"),
        ("historical", "legacy_joint_surface", "overlay_shannon_gate_6"),
        ("controlled", "controlled_two", "controlled_dose"),
    ):
        blend("ensemble_" + suffix, ((ANCHOR, 0.5), (left, 0.25), (right, 0.25)))
    return fits, blends, fractions, alternatives, tuple(cells)


DOSE_SETS = {
    "h35": (U, "d_h35"),
    "h50": (U, "d_h50"),
    "h65": (U, "d_h65"),
    "h35_65": (U, "d_h35", "d_h65"),
    "rc3": (U, "d_rc3_0"),
    "rc4": (U, "d_rc4_0", "d_rc4_1"),
    "rbf": (U, "d_rbf30", "d_rbf70"),
    "legacy_tails": (U, C, "d_h25", "d_h75"),
    "soft": (U, "d_soft50"),
    "linear": (U,),
}
FITS, BLENDS, FRACTIONS, ALT, CELLS = build()
FAMILIES = (*FITS, *ALT, *FRACTIONS, *BLENDS)
SETTINGS = {
    n: (
        ALT_OPTIONS[ALT[n][0]]
        if n in ALT
        else (
            tuple(sorted(set(PENALTIES + LEGACY_PENALTIES), reverse=True))
            if n in (ANCHOR, "legacy_joint_surface", "legacy_shannon_surface")
            else PENALTIES
        )
    )
    for n in FAMILIES
}
KEYS = tuple(key(n, p) for n in FAMILIES for p in SETTINGS[n])
FAMILY_INDEX = {n: i for i, n in enumerate(FAMILIES)}
INDEX = {k: i for i, k in enumerate(KEYS)}
OFF_INDEX = tuple(FAMILY_INDEX[n] for n in FAMILIES for p in SETTINGS[n])
COLUMN_SHA = sha256(
    json.dumps([FAMILIES, KEYS, OFF_INDEX, ADAPTIVE], separators=(",", ":")).encode()
).hexdigest()
GROUPS = {
    n: (
        FITS[n].group
        if n in FITS
        else "alternate_estimator" if n in ALT else "attenuation" if n in FRACTIONS else "paired_ensemble"
    )
    for n in FAMILIES
}


def design():
    return dict(
        recipes={n: asdict(r) for n, r in FITS.items()},
        blend_weights=BLENDS,
        shrink_fractions=FRACTIONS,
        other_estimators=ALT,
        settings=SETTINGS,
        settings_order=KEYS,
        family_order=FAMILIES,
        paired_column_sha256=COLUMN_SHA,
        primary_fixed=PRIMARY,
        reference_fixed=REFERENCE,
        adaptive=ADAPTIVE,
        anchor_ablation_cells=CELLS,
        crossed_cells=CROSS_CELLS,
        crossed_diagnostic_penalty=CROSS_PENALTY,
        selection_off_tolerance_r2=SELECTION_OFF_TOLERANCE,
        threshold_pct=TARGET,
    )
