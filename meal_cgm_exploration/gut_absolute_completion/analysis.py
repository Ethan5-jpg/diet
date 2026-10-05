"""Forty verified references plus thirty-two previously unfit specifications."""
import numpy as np
import pandas as pd
from hppmeal.common import require
from hppmeal.models import bh_planned
from gut_absolute_robustness.inputs import selected
from gut_absolute_robustness.analysis import estimate_common, identity_checks
from .inputs import contract, KEYS, NUMERIC, compare


def panel_fdr(rows):
    d = rows.copy()
    d["q_panel12"] = np.nan
    for _, ix in d.groupby(["setting", "threshold"]).groups.items():
        require(len(ix) == 12, "Incomplete planned common-panel family")
        d.loc[ix, "q_panel12"] = bh_planned(d.loc[ix, "p"])
    d["planned_family_size"] = 12
    return d


def complete(data, progress=print):
    rows, audit = [], []
    c = contract()
    for setting in c["settings"]:
        for t in c["thresholds"]:
            p, m = selected(data["persons"], data["members"], setting, t["meals"], t["dates"])
            group = data["flow"].loc[data["flow"].setting.eq(setting) & data["flow"].threshold.eq(t["name"])].iloc[0]
            for taxon in c["taxa"]:
                for outcome in c["outcomes"]:
                    source_table = "NEW_FIT"
                    if t["name"] == "m5d3" or outcome == "baseline_glucose":
                        source_table = "previous_a" if t["name"] == "m5d3" else "previous_b"
                        frame = data[source_table]
                        found = frame.loc[frame.setting.eq(setting) & frame.threshold.eq(t["name"]) & frame.taxon.eq(taxon) & frame.outcome.eq(outcome)]
                        require(len(found) == 1, "Missing reusable estimate")
                        row = found.iloc[0].to_dict()
                        audit.append(dict(setting=setting, threshold=t["name"], taxon=taxon, outcome=outcome,
                                          source_table=source_table, check="COPIED_VERIFIED_FULL_PRECISION"))
                    else:
                        row = estimate_common(p, m, setting, taxon, outcome)
                    # A Pre copied from B has identical X, but acquires the NEW panel q.
                    for field in ["q_family", "q_meal12", "family", "q_panel12"]:
                        row.pop(field, None)
                    row.update(threshold=t["name"], model="common_adjustment_without_pre_or_slope",
                               estimate_origin=source_table, role="new_support_extension" if source_table == "NEW_FIT" else "reused_reference",
                               person_sha256=group.person_sha256, meal_sha256=group.meal_sha256)
                    rows.append(row)
            progress("COMMON {} {}: {} people; {} meals".format(setting, t["name"], len(p), len(m)))
    d = panel_fdr(pd.DataFrame(rows))
    require(len(d) == 72 and not d.duplicated(KEYS).any(), "Incomplete 72-row common design")
    require(d.estimate_origin.eq("NEW_FIT").sum() == 32 and len(audit) == 40, "Incorrect reuse/new-fit counts")
    compare(d.loc[d.threshold.eq("m5d3")], data["previous_a"], KEYS,
            NUMERIC + ["people", "meals", "q_panel12"], ["status", "adjustment", "unit"])
    high_pre = d.loc[d.threshold.ne("m5d3") & d.outcome.eq("baseline_glucose")]
    compare(high_pre, data["previous_b"].loc[data["previous_b"].threshold.ne("m5d3") & data["previous_b"].outcome.eq("baseline_glucose")],
            KEYS, NUMERIC + ["people", "meals"], ["status", "adjustment", "unit"])
    checks = []
    for threshold, g in d.groupby("threshold"):
        require(g.groupby("setting").person_sha256.nunique().eq(1).all() and g.groupby("setting").meal_sha256.nunique().eq(1).all(), "Common model members differ")
        checks.append(identity_checks(g).assign(threshold=threshold))
    identity = pd.concat(checks, ignore_index=True)
    require(len(identity) == 24, "Incomplete common-X identities")
    return d, identity, pd.DataFrame(audit)
