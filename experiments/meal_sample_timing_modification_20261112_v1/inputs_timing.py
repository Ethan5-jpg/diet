"""Link the actual archived stool profiles to their verified recorded dates."""

import csv
from io import BytesIO
from pathlib import Path
import sys

import numpy as np

from timing import TZ, align_dates

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PROCESSING = ROOT.parent / "meal_processing_modification_20261112_v1"
sys.path[:0] = [str(MEAL), str(PROCESSING)]
try:
    from audit import projected_meals
    from paired import archived_inputs, verified_result
    from prepare import digest, load, require, scoped_read
finally:
    del sys.path[:2]

PINS = {
    PROCESSING / "audit.py": "d093b86be0d041fc2eb1530329216518eb76ddae54615fcdcc2c3d0ca3463295",
    PROCESSING / "processing.py": "f249264bcdec0ed84b299cd3702ff5ed1e61937a6baf33ea8ed883fce697c567",
}


def timing_inputs(package):
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned meal projection dependencies differ")
    meals, index, _, inputs, io, prep, sources = projected_meals(package)
    roster = index.loc[index.input_row.isin(meals.input_row)]
    archive = MEAL / "runs_TRE_ONLY/models-1"
    fitted, _ = verified_result(archive)
    frozen = package / "inputs/discovery_v2"
    raw_path, taxa_path = frozen / "gut_raw_TRE_ONLY.npz", frozen / "gut_taxa.json"
    shared = package.parent / "analysis/shared"
    registry_path = shared / "sample_registry_TRE_ONLY.parquet"
    matrix_path = shared / "gut_selected_abundance_TRE_ONLY.parquet"
    visit_path = shared / "cgm_visit_phenotypes_TRE_ONLY.parquet"
    lock = package.parent / "config/w13_sample_selection_lock.json"
    manifest = package.parent / "data_qc/w13_microbiome_qc_20261029/final_artifact_manifest.json"
    dictionaries = Path("/home/ec2-user/studies/hpp_datasets/gut_microbiome/metadata")
    definitions = []
    for path in dictionaries.glob("*data_dictionary.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            definitions.extend(path for row in csv.reader(stream)
                               if "collection_date" in row and "Date sample was given" in row)
    require(len(definitions) == 1, "Gut collection-date field lacks the recorded handoff definition")
    paths = [raw_path, taxa_path, registry_path, matrix_path, visit_path, lock, manifest]
    sources.update({str(p): digest(p) for p in [*paths, *definitions]})
    require(all(fitted["sources"][str(p)] == sources[str(p)] for p in (raw_path, taxa_path)),
            "Archived gut matrix changed")
    require(load(prep)["sources"][str(visit_path)] == sources[str(visit_path)],
            "Archived CGM connection changed")
    locked = load(lock)
    assets = {entry["path"]: entry["sha256"] for entry in load(manifest)}
    require(locked["sample_registry"] == str(registry_path)
            and locked["sample_registry_sha256"] == sources[str(registry_path)]
            and assets[str(matrix_path)] == sources[str(matrix_path)], "W13 published artifacts changed")
    taxa = load(taxa_path)
    with np.load(BytesIO(raw_path.read_bytes()), allow_pickle=False) as saved:
        frozen_matrix = saved["abundance"]
    require(frozen_matrix.shape == (len(index), len(taxa)) and len(set(taxa)) == len(taxa),
            "Archived gut matrix or taxon order is malformed")
    columns = ["sample_id", "site", "sample_lock_status", "descriptive_sample_calendar_date",
               "linked_index_calendar_date", "date_conflict_between_linked_sources"]
    registry = scoped_read(registry_path, columns, roster, inputs)
    registry = registry.loc[registry.site.eq("gut")].copy()
    require(registry.input_row.is_unique and registry.sample_id.is_unique
            and registry.sample_lock_status.eq("locked_W13_independent_full_source_verification_passed").all()
            and set(registry.input_row) == set(roster.input_row), "Incomplete locked gut registry")
    abundance = scoped_read(matrix_path, ["sample_id", "site", *taxa], roster, inputs).sort_values("input_row")
    identity = registry.sort_values("input_row")
    require(np.array_equal(abundance.input_row, identity.input_row)
            and np.array_equal(abundance.sample_id, identity.sample_id) and abundance.site.eq("gut").all(),
            "Registry IDs differ from the W13 abundance matrix")
    check = abundance[taxa].to_numpy(dtype=frozen_matrix.dtype)
    require(np.isfinite(check).all() and np.array_equal(check, frozen_matrix[abundance.input_row.to_numpy(int)]),
            "Dated W13 profiles differ from the archived model profiles")
    visits = scoped_read(visit_path, ["connection_id", "point_qc_verified", "first_observed", "timezone"],
                         roster, inputs)
    require(visits.input_row.is_unique and visits.connection_id.is_unique and visits.point_qc_verified.eq(True).all()
            and visits.timezone.eq(TZ).all(), "Unexpected archived monitor contract")
    aligned = align_dates(meals, visits, registry)
    scripts = [Path(__file__).resolve(), ROOT / "timing.py", *PINS]
    sources.update({str(p): digest(p) for p in scripts})
    return aligned, io, sources
