conda activate pheno &&
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person &&
python -B - outputs/daily_20261011T055621_479640Z_666d40 <<'PY'
"""Read-only, aggregate-only AMED alcohol/date audit; no models or imputation."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from daily.common import sha256, read_csv, scoped, date_only, number
from daily.models import EXPOSURES
from run_analysis import code_hashes, load_saved_panel

r = Path(sys.argv[1]).resolve()
m = json.loads((r / "manifest.json").read_text())
assert m["status"] in ["NO_ESTIMABLE_PRIMARY_MODELS", "COMPLETE", "COMPLETE_WITH_LIMITATIONS"]
assert code_hashes(sha256) == m["code_sha256"], "Runtime code changed"
for name in ["analysis_config.json", "private/daily_panel.csv"]:
    assert sha256(r / name) == m["output_sha256"][name], "Saved file changed: " + name
c = json.loads((r / "analysis_config.json").read_text())
assert c == m["config"], "Configuration mismatch"
p = load_saved_panel(r / "private/daily_panel.csv").replace([np.inf, -np.inf], np.nan)
b = p.loc[p.primary_eligible & p.diet_pass & p.cgm_pass].copy()
source = Path(m["data_root"]) / "diet_deal/outputs/04_score_intakes/amed/amed_daily_component_intakes.csv"
source_hash = sha256(source)
rename = {"alcohol_g_total": "amed_g", "event_count": "amed_events",
          "excluded_missing_labels_event_count": "excluded", "alcoholic_drink_event_count": "drinks",
          "alcohol_observed_event_count": "obs", "alcohol_energy_formula_event_count": "formula",
          "alcohol_standard_abv_event_count": "abv", "alcohol_zero_fallback_event_count": "zero",
          "alcohol_unresolved_event_count": "unresolved", "alcohol_invalid_observed_event_count": "invalid"}
required = ["participant_id", "cohort", "research_stage", "collection_date"] + list(rename)
columns = pd.read_csv(source, nrows=0).columns
missing = sorted(set(required) - set(columns))
print("AMED CHECK |", m["data_kind"], "| saved hashes verified |", r.name)
print("Missing required fields:", ",".join(missing) or "none")
if missing:
    raise SystemExit("STOP: source audit needs these fields; no model or file changed.")
a = scoped(read_csv(source, required, usecols=required), c, "AMED", [], unique_person=False, allow_empty=True)
scoped_rows = len(a)
a["date"] = date_only(a.collection_date, c["time"]["local_timezone"])
bad_dates = int(a.date.isna().sum())
a = a.loc[a.date.notna()].rename(columns=rename)
keys = ["participant_id", "date"]
duplicates = a.duplicated(keys, keep=False)
duplicate_keys = a.loc[duplicates, keys].drop_duplicates().assign(ambiguous=True)
a = a.loc[~duplicates, keys + list(rename.values())].copy()
for field in rename.values():
    a[field] = number(a[field])
b = b.merge(duplicate_keys, on=keys, how="left", validate="one_to_one")
b = b.merge(a, on=keys, how="left", validate="one_to_one", indicator=True)
joined = b["_merge"].eq("both")
counts = [v for v in rename.values() if v != "amed_g"]
valid_counts = b[counts].notna().all(axis=1) & b[counts].ge(0).all(axis=1) & b[counts].mod(1).eq(0).all(axis=1)
closure = b[["obs", "formula", "abv", "zero", "unresolved"]].sum(axis=1, min_count=5).eq(b.drinks)
closure &= b.drinks.le(b.amed_events) & (b.drinks.gt(0) | b.amed_g.eq(0))
event_match = (b.amed_events + b.excluded).eq(b.events)
safe = joined & valid_counts & closure & event_match & b.amed_g.ge(0)
safe &= b[["excluded", "zero", "unresolved", "invalid"]].eq(0).all(axis=1)
b["audit_recovered"] = safe
b["audit_observed_only"] = safe & b.formula.eq(0) & b.abv.eq(0)
assert sha256(source) == source_hash, "AMED file changed during reading"
print("Scoped AMED rows / invalid date rows:", scoped_rows, bad_dates)
print("QC days / unique matched / ambiguous date / unmatched:", len(b), int(joined.sum()), int(b.ambiguous.eq(True).sum()), int((~joined & ~b.ambiguous.eq(True)).sum()))
print("Matched-day event-count mismatch / invalid source counts / closure failure:", int((joined & ~event_match).sum()), int((joined & ~valid_counts).sum()), int((joined & ~closure).sum()))
print("Matched QC days with source flag >0 (overlapping counts):")
for field in ["drinks", "obs", "formula", "abv", "zero", "unresolved", "invalid", "excluded"]:
    print(field, int((joined & b[field].gt(0)).sum()))
print("Candidate QC days: observed-or-no-drink / incl.formula-ABV:", int(b.audit_observed_only.sum()), int(safe.sum()))
print("COUNTS ONLY | Diet Taxon Source CompletePeople CompleteDays FinalPeople FinalDays")
for exposure, meta in EXPOSURES.items():
    e = b.loc[b.nova_coverage.ge(c["diet"]["minimum_nova_coverage"])] if exposure == "NOVA4" else b
    for taxon in [t["id"] for t in c["taxa"]]:
        fields = [meta["field"], taxon, "daily_sd", "energy_kcal", "weekday", "observation_day"]
        for mode in ["no_alcohol_requirement", "audit_observed_only", "audit_recovered"]:
            d = e.dropna(subset=fields)
            if mode != "no_alcohol_requirement":
                d = d.loc[d[mode]]
            before = (d.participant_id.nunique(), len(d))
            g = d.groupby("participant_id")[meta["field"]]
            keep = g.transform("size").ge(c["analysis"]["minimum_days"]) & (g.transform("max") - g.transform("min")).ge(c["analysis"][meta["range_key"]])
            d = d.loc[keep]
            print(exposure, taxon, mode, *before, d.participant_id.nunique(), len(d))
print("AMED_SHA256=" + source_hash)
print("Recovered is a source-audit candidate, not measured truth or a fitted model.")
print("No participant rows, imputation, file writes or CGM rebuild; keep original outputs.")
PY
