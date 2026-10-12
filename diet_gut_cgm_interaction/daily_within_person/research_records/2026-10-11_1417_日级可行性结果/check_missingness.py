"""Read-only feasibility diagnostic; prints aggregates, never fits or imputes."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from daily.common import sha256
from daily.models import EXPOSURES
from run_analysis import code_hashes, load_saved_panel

r = Path(sys.argv[1]).resolve()
m = json.loads((r / "manifest.json").read_text())
assert m["status"] in ["NO_ESTIMABLE_PRIMARY_MODELS", "COMPLETE", "COMPLETE_WITH_LIMITATIONS"]
assert code_hashes(sha256) == m["code_sha256"], "Runtime code changed since this run"
for name in ["analysis_config.json", "private/daily_panel.csv", "tables/10_interactions.csv"]:
    assert sha256(r / name) == m["output_sha256"][name], "Saved file changed: " + name
c = json.loads((r / "analysis_config.json").read_text())
assert c == m["config"], "Saved configuration mismatch"
p = load_saved_panel(r / "private/daily_panel.csv").replace([np.inf, -np.inf], np.nan)
saved = pd.read_csv(r / "tables/10_interactions.csv")
saved = saved.loc[saved.spec.eq("primary") & saved.outcome.eq("SD")].set_index(["exposure", "taxon"])
base = p.loc[p.primary_eligible & p.diet_pass & p.cgm_pass].copy()

print("DAILY CHECK | " + m["data_kind"] + " | saved hashes verified")
print("Counts only: same QC, exposure completeness, minimum days and within-person range; no fitting.")
print("Diet Taxon AlcoholRequired CompletePeople CompleteDays FinalPeople FinalDays Ready")
for exposure, meta in EXPOSURES.items():
    b = base.loc[base.nova_coverage.ge(c["diet"]["minimum_nova_coverage"])].copy() if exposure == "NOVA4" else base
    for taxon in [t["id"] for t in c["taxa"]]:
        fields = [meta["field"], taxon, "daily_sd", "energy_kcal", "weekday", "observation_day"]
        for need_alcohol in [True, False]:
            d = b.dropna(subset=fields + (["alcohol_g"] if need_alcohol else [])).copy()
            before_people, before_days = d.participant_id.nunique(), len(d)
            g = d.groupby("participant_id")[meta["field"]]
            keep = g.transform("size").ge(c["analysis"]["minimum_days"]) & (g.transform("max") - g.transform("min")).ge(c["analysis"][meta["range_key"]])
            d = d.loc[keep]
            people, days = d.participant_id.nunique(), len(d)
            if need_alcohol:
                original = saved.loc[(exposure, taxon)]
                assert (people, days) == (int(original.n_people), int(original.n_days)), "Original sample not reproduced"
            print(exposure, taxon, "YES(original)" if need_alcohol else "NO(diagnostic)", before_people, before_days, people, days, people >= c["analysis"]["minimum_people"])

print("QUALITY-DAY FIELD AUDIT: Field MissingDays/AllDays KnownEnergyCoverage_P10/P50/P90")
for field, coverage in [("alcohol_g", "alcohol_g_energy_coverage"), ("fiber_density", "fiber_g_energy_coverage")]:
    quantiles = base[coverage].quantile([.1, .5, .9]).tolist()
    print(field, str(int(base[field].isna().sum())) + "/" + str(len(base)), "/".join("{:.4f}".format(x) for x in quantiles))

a = Path(m["data_root"]) / "diet_deal/outputs/04_score_intakes/amed/amed_daily_component_intakes.csv"
print("EXISTING_AMED_DAILY_FILE:", a.is_file())
if a.is_file():
    columns = pd.read_csv(a, nrows=0).columns.tolist()
    required = ["participant_id", "cohort", "research_stage", "collection_date", "alcohol_g_total"]
    provenance = ["alcohol_observed_event_count", "alcohol_energy_formula_event_count", "alcohol_standard_abv_event_count", "alcohol_zero_fallback_event_count"]
    print("AMED missing identity/value fields:", ",".join(x for x in required if x not in columns) or "none")
    print("AMED available source flags:", ",".join(x for x in provenance if x in columns) or "none")
print("READY means count threshold only; no revised model has been fitted.")
print("No missing values filled; no files written. Keep this completed daily panel on the server.")
