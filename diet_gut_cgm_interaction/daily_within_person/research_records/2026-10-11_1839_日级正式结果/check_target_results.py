"""Read saved aggregate tables only; no participant rows, fits or FDR recomputation."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from run_revision import own_hashes
from daily.common import sha256
from run_analysis import code_hashes
from revision_models import SPECS, EXPOSURES, OUTCOMES
from revision_reporting import ALIASES

r = Path(sys.argv[1]).resolve()
m = json.loads((r / "manifest.json").read_text())
assert m["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"], "Expected a completed revision"
assert m["version"] == "2026-10-11.alcohol.1", "Wrong revision version"
assert own_hashes(sha256) == m["code_sha256"], "Revision code changed"
assert code_hashes(sha256) == m["parent_code_sha256"], "Parent code changed"
files = ["tables/10_interactions.csv", "tables/13_primary_slopes.csv",
         "tables/14_primary_anchors.csv", "tables/02_paired_sample_identity.csv"]
for name in files:
    assert sha256(r / name) == m["output_sha256"][name], "Saved table changed: " + name
t, slopes, anchors, paired = [pd.read_csv(r / name) for name in files]
keys = ["spec", "exposure", "taxon", "outcome"]
expected = {(s, e, taxon, y) for s in SPECS for e in EXPOSURES
            for taxon in ["T068", "T096"] for y in OUTCOMES}
assert set(t[keys].itertuples(index=False, name=None)) == expected
assert len(t) == 132 and not t.duplicated(keys).any(), "Incomplete or duplicate model registry"
assert t.status.eq("ESTIMATED").all(), "Unexpected failed models"
assert np.allclose(t.beta, (t.low + t.high) / 2, rtol=1e-9, atol=1e-10), "Coefficient/CI mismatch"
assert len(paired) == 24 and paired.identical.astype(str).str.lower().eq("true").all()

def f(x):
    return format(float(x), ".6g") if pd.notna(x) and np.isfinite(float(x)) else "NA"

print("DAILY FOLLOWUP 1/2 |", m["data_kind"], "| saved aggregate table hashes verified")
print("Fixed targets: primary NOVA4 x T096 SD/Mean. Existing 4/12 q families; no new tests.")
print("P=primary S=strict Q=coverage95 D=days7 N=nutrients B=baseline W=equal-weight")
print("R=recovered; P0/R0=same samples without alcohol controls; A0=all eligible days, no alcohol controls")
print("Spec Y People Days Beta [L95,U95] p qSD4 qAll12")
for y in ["SD", "Mean"]:
    d = t.loc[t.exposure.eq("NOVA4") & t.taxon.eq("T096") & t.outcome.eq(y)].set_index("spec")
    for spec in SPECS:
        x = d.loc[spec]
        print("{} {} {} {} {} [{},{}] {} {} {}".format(
            ALIASES[spec], y, int(x.n_people), int(x.n_days), f(x.beta), f(x.low), f(x.high),
            f(x.p), f(x.q_primary4), f(x.q_all12)))

print("\nDAILY FOLLOWUP 2/2 | PRIMARY NOVA4 SLOPES / TIMING |", m["data_kind"])
print("Per NOVA4 +10pp; SD/Mean mg/dL. Pointwise 95% CI; controls held constant.")
print("Taxon Y Point MicrobeCLR_Z Slope_or_difference [L95,U95]")
for taxon, y in [("T068", "SD"), ("T096", "SD"), ("T096", "Mean")]:
    d = slopes.loc[slopes.exposure.eq("NOVA4") & slopes.taxon.eq(taxon) & slopes.outcome.eq(y)].set_index("point")
    assert not d.index.duplicated().any()
    a = anchors.loc[anchors.exposure.eq("NOVA4") & anchors.taxon.eq(taxon) & anchors.outcome.eq(y)]
    b = t.loc[t.spec.eq("primary") & t.exposure.eq("NOVA4") & t.taxon.eq(taxon) & t.outcome.eq(y)]
    assert len(a) == len(b) == 1
    a, b = a.iloc[0], b.iloc[0]
    iqr = a.q75 - a.q25
    assert iqr > 0 and np.allclose(d.loc[["q25", "q75"], "microbe"], [a.q25, a.q75])
    assert np.allclose(d.loc["q75_minus_q25", ["beta", "low", "high"]].astype(float),
                       b[["beta", "low", "high"]].astype(float) * iqr, rtol=1e-8, atol=1e-10)
    for point in ["q25", "q75", "q75_minus_q25"]:
        x = d.loc[point]
        print("{} {} {} {} {} [{},{}]".format(taxon, y, point, f(x.microbe), f(x.beta), f(x.low), f(x.high)))
print("Stored IQR contrasts match primary interactions and anchors.")
print("Primary SD stool timing in participant-DAYS: Diet Taxon Before Same After Unknown Total")
for x in t.loc[t.spec.eq("primary") & t.outcome.eq("SD")].itertuples():
    values = [int(x.sample_before_days), int(x.sample_same_day), int(x.sample_after_days), int(x.sample_timing_unknown_days)]
    assert sum(values) == int(x.n_days), "Timing counts do not sum to model days"
    print(x.exposure, x.taxon, *values, int(x.n_days))
print("Same-sample pairs verified from saved identities; no refits, new FDR, or file writes.")
print("RESULT_DIR=" + str(r))
