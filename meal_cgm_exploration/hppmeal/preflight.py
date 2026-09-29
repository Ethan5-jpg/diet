"""Fast header/dependency checks before ingesting full source data."""
import glob
import importlib
import shutil
from pathlib import Path

import pandas as pd

from .common import KEYS, header, resolve_taxa, scope_columns


def check(run):
    rows, details, versions = [], [], {}
    dependencies_ok = True
    for module in ["numpy", "pandas", "scipy", "matplotlib"]:
        try:
            m = importlib.import_module(module)
            versions[module] = m.__version__
            if module == "scipy":
                importlib.import_module("scipy.stats")
            if module == "matplotlib":
                m.use("Agg")
                importlib.import_module("matplotlib.pyplot")
            rows.append(dict(source="dependency:" + module, status="OK", detail=versions[module]))
        except Exception as e:
            dependencies_ok = False
            rows.append(dict(source="dependency:" + module, status="MISSING_OR_BROKEN", detail=str(e)))

    def inspect(name, columns):
        p = run.path(name)
        if p is None or not p.is_file():
            rows.append(dict(source=name, status="MISSING", detail="Not configured or file not found"))
            return None
        try:
            found = header(p)
            missing = sorted(set(columns) - set(found))
            if missing:
                rows.append(dict(source=name, status="INVALID_HEADER", detail="Missing " + ", ".join(missing)))
                return None
            rows.append(dict(source=name, status="HEADER_OK", detail="{} columns; rows not inspected".format(len(found))))
            return found
        except (ValueError, OSError, UnicodeError) as e:
            rows.append(dict(source=name, status="INVALID_HEADER", detail=str(e)))
            return None

    diet = inspect("diet", KEYS + [run.c["time"]["diet_column"], "food_id", "calories_kcal", "carbohydrate_g"])
    files = sorted(glob.glob(str(run.path("cgm_glob")) or ""))
    valid, total_bytes = 0, 0
    required_cgm = set(KEYS + ["connection_id", run.c["time"]["cgm_column"], "glucose"])
    for i, name in enumerate(files, 1):
        try:
            p = Path(name)
            total_bytes += p.stat().st_size
            missing = sorted(required_cgm - set(header(p)))
            if missing:
                details.append(dict(file_number=i, path=str(p), reason="Missing " + ", ".join(missing)))
            else:
                valid += 1
        except (ValueError, OSError, UnicodeError) as e:
            details.append(dict(file_number=i, path=name, reason=str(e)))
        if i % 1000 == 0:
            print("CHECK CGM headers {}/{}".format(i, len(files)), flush=True)
    rows.append(dict(source="cgm_glob", status="HEADER_OK" if files and valid == len(files) else "INVALID_OR_MISSING",
                     detail="{} files; {} valid headers; source bytes {}".format(len(files), valid, total_bytes)))
    if details:
        run.table(pd.DataFrame(details), "00_invalid_cgm_headers.csv", private=True)
    mapping = inspect("nova_mapping", ["food_id", "nova_group", "nova_mapping_status"])
    cov = inspect("covariates", ["participant_id", "exclude_diabetes_or_a10"] + run.c["analysis"]["require_covariate_columns"])
    micro = inspect("microbiome", ["participant_id"])
    scope_ok = True
    for label, columns in [("covariates", cov), ("microbiome", micro)]:
        if columns:
            try:
                stage = scope_columns(columns, label)
                rows.append(dict(source=label + "_scope", status="HEADER_OK",
                                 detail="cohort and {}; values checked during analyze".format(stage)))
            except ValueError as e:
                scope_ok = False
                rows.append(dict(source=label + "_scope", status="REVIEW_REQUIRED", detail=str(e)))
    taxa_ok = not run.c["analysis"]["taxa"]
    if micro:
        taxa = resolve_taxa(micro, run.c["analysis"]["taxa"])
        run.table(pd.DataFrame(taxa), "00_taxon_columns.csv")
        mapped = [t["column"] for t in taxa if t["status"] == "FOUND"]
        taxa_ok = len(mapped) == len(taxa) and len(mapped) == len(set(mapped))
        rows.append(dict(source="taxon_columns", status="UNIQUE_MATCHES" if taxa_ok else "REVIEW_REQUIRED",
                         detail="{} of {} configured taxa found uniquely".format(len(mapped), len(taxa))))
    dates_ok = True
    if run.path("microbiome_dates") is not None:
        dates_ok = inspect("microbiome_dates", KEYS + ["collection_date"]) is not None
    else:
        rows.append(dict(source="microbiome_dates", status="NOT_CONFIGURED", detail="Sample timing unknown; never label microbiome baseline"))
    run.table(pd.DataFrame(rows), "00_preflight.csv")
    mapping_path = run.path("nova_mapping")
    mapping_loadable = mapping is not None or mapping_path is None or not mapping_path.exists()
    ready = bool(dependencies_ok and diet and files and valid == len(files) and mapping_loadable)
    model_inputs_ready = bool(mapping and cov and taxa_ok and scope_ok and dates_ok and (micro or not run.c["analysis"]["taxa"]))
    available = shutil.disk_usage(str(run.out)).free
    return dict(status="PASS" if ready else "FAILED", build_ready=ready,
                model_input_headers_ready=model_inputs_ready, dependencies=versions,
                consumption_time_verified=run.c["time"]["consumption_time_verified"],
                cgm_files=len(files), cgm_source_bytes=total_bytes, output_disk_free_bytes=available,
                reason=("Header-only check; confirm timestamps and inspect row-level QC after build."
                        if ready else "Fix 00_preflight.csv issues before full ingestion; invalid file details are in private/00_invalid_cgm_headers.csv."))
