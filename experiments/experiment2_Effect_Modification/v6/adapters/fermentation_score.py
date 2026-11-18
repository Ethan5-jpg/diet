"""Fixed species-abundance score motivated by resistant-starch studies."""

from pathlib import Path
import re

import numpy as np


SPECIES = {
    "Ruminococcus bromii": ("ruminococcus_bromii",),
    "Bifidobacterium adolescentis": ("bifidobacterium_adolescentis",),
    "Eubacterium rectale": ("eubacterium_rectale", "agathobacter_rectalis"),
}
MASS_UNIT = 0.0001


def resolve_species(taxa):
    if not isinstance(taxa, (list, tuple)) or not all(isinstance(name, str) for name in taxa):
        raise ValueError("INVALID_GUT_TAXON_DICTIONARY")
    by_species = {}
    for column, label in enumerate(taxa):
        term = label.rsplit("|", 1)[-1].strip().lower()
        if term.startswith("s__"):
            term = term[3:]
        elif "|" in label or "__" in term:
            continue
        term = re.sub(r"\s+", "_", term)
        by_species.setdefault(term, []).append((column, label))
    matches = {}
    for species, aliases in SPECIES.items():
        columns = [entry for alias in aliases for entry in by_species.get(alias, ())]
        if len(columns) != 1:
            detail = "MISSING" if not columns else "AMBIGUOUS"
            raise ValueError(f"PANEL_TAXON_{detail} {species} matches={len(columns)}")
        matches[species] = columns[0]
    return matches


def build_score(raw, taxa):
    """Keep archived zero values and flag unusable profiles without imputation."""
    matches = resolve_species(taxa)
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(taxa):
        raise ValueError("GUT_PROFILE_COLUMN_ALIGNMENT")
    if np.isinf(raw).any() or np.any(raw < 0):
        raise ValueError("INVALID_GUT_ABUNDANCE")
    total = raw.sum(axis=1)
    usable = np.isfinite(total) & (total > 0)
    if not usable.any() or not np.allclose(total[usable], 1, rtol=0, atol=0.001):
        raise ValueError("EXPECTED_GUT_RELATIVE_ABUNDANCE_FRACTIONS")
    panel = raw[:, [column for column, _ in matches.values()]]
    score = np.full(len(raw), np.nan)
    score[usable] = np.log1p(panel[usable].sum(axis=1) / MASS_UNIT)
    return score, matches, panel


def describe(score, matches, panel, rows):
    score, panel = score[rows], panel[rows]
    valid = np.isfinite(score)
    if not valid.any():
        raise ValueError("NO_USABLE_FERMENTATION_SCORE")
    q25, q50, q75 = np.quantile(score[valid], [0.25, 0.5, 0.75])
    return {
        "rows": len(score),
        "scored": int(valid.sum()),
        "any_detected": int(np.count_nonzero(score[valid] > 0)),
        "score_q25_q50_q75": [float(q25), float(q50), float(q75)],
        "taxa": {
            species: {"label": label, "detected": int(np.count_nonzero(panel[valid, j] > 0))}
            for j, (species, (_, label)) in enumerate(matches.items())
        },
    }


def check_archive_score(package, *, digest, read_json, v11, run_name):
    """Read the two registered gut files for an outcome-free coverage check."""
    import io

    package = Path(package)
    source = package / "inputs/discovery_v2"
    paths = (source / "gut_taxa.json", source / "gut_raw_TRE_ONLY.npz")
    registration = read_json(package / "experiments" / v11 / "runs" / run_name / "registration.json")
    if registration["config"]["version"] != "v11_20261109":
        raise ValueError("UNEXPECTED_V11_REGISTRATION")
    for path in paths:
        if digest(path) != registration["hashes"].get(str(path)):
            raise ValueError(f"V11_REGISTERED_GUT_SOURCE_MISMATCH {path.name}")
    taxa = read_json(paths[0])
    resolve_species(taxa)
    with np.load(io.BytesIO(paths[1].read_bytes()), allow_pickle=False) as archive:
        return build_score(archive["abundance"], taxa)


def score_selftest():
    taxa = ["s__Bifidobacterium_adolescentis", "g__Agathobacter|s__Agathobacter_rectalis",
            "Ruminococcus_bromii", "s__Ruminococcus_bromii_related"]
    matrix = np.array([[0.01, 0.02, 0.04, 0.93], [0, 0, 0, 1], [0, 0, 0, 0], [np.nan, 0, 0, 0]])
    result, matches, panel = build_score(matrix, taxa)
    np.testing.assert_allclose(result, [np.log1p(700), 0, np.nan, np.nan])
    assert describe(result, matches, panel, np.arange(len(result)))["scored"] == 2
    for labels, code in (
        (taxa[1:], "PANEL_TAXON_MISSING"),
        (taxa + ["s__Eubacterium_rectale"], "PANEL_TAXON_AMBIGUOUS"),
        (taxa[:2] + ["g__Ruminococcus|s__Ruminococcus_bromii|t__SGB4285"], "PANEL_TAXON_MISSING"),
    ):
        try:
            resolve_species(labels)
        except ValueError as exc:
            assert code in str(exc)
        else:
            raise AssertionError("Invalid species mapping was accepted")
    try:
        build_score(matrix * 100, taxa)
    except ValueError as exc:
        assert str(exc) == "EXPECTED_GUT_RELATIVE_ABUNDANCE_FRACTIONS"
    else:
        raise AssertionError("Percentage units were accepted as fractions")
