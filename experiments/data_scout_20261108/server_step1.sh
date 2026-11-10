/home/ec2-user/.venvs/hpp_v8_20261108/bin/python -B - <<'PY'
"""Read Parquet metadata only; no row values, fitting or server file writes."""

import re
import textwrap
from pathlib import Path

import pyarrow.parquet as pq


def main(root):
    root = Path(root)
    project = root.parent
    tables = [
        ("CURRENT_PERSON", root / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet", ""),
        ("COMMON_DAYS", root / "inputs/daily/common_days_no_scores_TRE_ONLY.parquet", ""),
        ("FOOD_EVENTS", project / "analysis/shared/food_events_clean_TRE_ONLY.parquet",
         "carb|lipid|protein|sodium|alcohol|fiber|energy|calor|timestamp|time|date"),
        ("COVARIATES", project / "analysis/shared/covariate_registry_TRE_ONLY.parquet",
         "age|sex|smok|diabet|medic|bmi|weight|activity|sleep|batch|device|disease|site|center|ethnic|education|fast|health"),
        ("SAMPLES", project / "analysis/shared/sample_registry_TRE_ONLY.parquet",
         "date|time|batch|site|sample|microbio|fast|collect"),
        ("VISIT_TIME", project / "analysis/D_Glu/fiber_density/base_table_TRE_ONLY.parquet",
         "date|time|day|device|coverage|eligible|duration|valid"),
    ]
    print("STEP1 SCHEMA ONLY: no row values, fitting or file changes.")
    print("Upstream row counts describe source tables, not the model evaluation population.")
    for label, path, pattern in tables:
        if not path.is_file():
            print(label + " MISSING")
            continue
        try:
            parquet = pq.ParquetFile(path, pre_buffer=False)
            names = parquet.schema_arrow.names
            selected = [n for n in names if not pattern or re.search(pattern, n, re.IGNORECASE)]
            print(f"{label} rows={parquet.metadata.num_rows} cols={len(names)} selected={len(selected)}")
            content = ", ".join(selected).encode("ascii", "backslashreplace").decode("ascii")
            lines = textwrap.wrap(content, width=112, break_long_words=False, break_on_hyphens=False)
            for line in lines[:3]:
                print("  " + line)
            if not lines:
                print("  NO_MATCHED_FIELDS")
            if len(lines) > 3:
                print(f"  MORE_SCHEMA_LINES={len(lines) - 3} (not displayed)")
        except (OSError, ValueError) as error:
            print(f"{label} SCHEMA_ERROR={type(error).__name__}")


if __name__ == "__main__":
    main(Path("/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1"))
PY
