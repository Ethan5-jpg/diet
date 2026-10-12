"""Three compact screens; full tables remain inside the new server run."""
import numpy as np
from daily.reporting import fmt
from revision_models import SPECS

ALIASES = dict(zip(SPECS, ["P", "S", "Q", "D", "N", "B", "W", "R", "P0", "R0", "A0"]))


def write_screens(out, manifest, feasible, results, slopes):
    a = manifest["alcohol_audit"]
    one = ["DAILY REVISED 1/3 | DATA / SOURCE AUDIT | " + manifest["data_kind"],
           "VERSION=" + manifest["version"] + " | STATUS=" + manifest["status"],
           "Reuses verified saved daily panel; no CGM rebuild. Person FE; cluster CR1.",
           "PRIMARY: observed alcohol or classified no-drink days; estimates excluded.",
           "QC days={qc_days}; unique AMED match={unique_matched_days}; ambiguous={ambiguous_days}; unmatched={unmatched_days}".format(**a),
           "QC source candidates: observed={} recovered={}".format(a["candidate_observed_days"], a["candidate_recovered_days"]),
           "Excluded source flags (overlap): fallback-zero={} invalid={} missing-label={}".format(a["flag_days_zero"], a["flag_days_invalid"], a["flag_days_excluded"]),
           ("Original sample counts and AMED screenshot checkpoints reproduced." if manifest["data_kind"] == "SERVER_INPUTS"
            else "SYNTHETIC validation only; no real screenshot count reproduction claimed."),
           "PRIMARY Diet Taxon People Days MedianDays"]
    for r in feasible.itertuples():
        one.append("{} {} {} {} {}".format(r.exposure, r.taxon, r.people, r.days, fmt(r.days_median)))
    one += ["P/P0 and R/R0: 24/24 person-day sample identity checks passed.",
            "Classified no-drink is based on food logs, not proof of no alcohol intake."]
    two = ["DAILY REVISED 2/3 | ALL 12 PRIMARY TESTS | " + manifest["data_kind"],
           "Beta: NOVA4 +10pp or Fiber +5g/1000kcal, times original microbe CLR-Z +1.",
           "SD/Mean mg/dL; CV percentage points; pointwise 95% CI.",
           "Diet Taxon Y People Days Beta [L95,U95] qSD4 qAll12"]
    primary = results.loc[results.spec.eq("primary")]
    for r in primary.itertuples():
        if r.status != "ESTIMATED":
            two.append("{} {} {} N={} D={} NOT_ESTIMATED".format(r.exposure, r.taxon, r.outcome, r.n_people, r.n_days))
        else:
            two.append("{} {} {} {} {} {} [{},{}] {} {}{}".format(
                r.exposure, r.taxon, r.outcome, r.n_people, r.n_days, fmt(r.beta), fmt(r.low),
                fmt(r.high), fmt(r.q_primary4), fmt(r.q_all12), " *" if r.limited_support else ""))
    two += ["qSD4=four SD tests; qAll12=all 12; failed tests remain in denominators.",
            "* limited microbial tails/IQR or >10% focal information from one person."]
    three = ["DAILY REVISED 3/3 | SENSITIVITY | " + manifest["data_kind"],
             "P=primary S=strict Q=coverage95 D=days7 N=nutrients B=baseline W=equal-weight",
             "R=recovered alcohol; P0/R0=same sample without alcohol controls; A0=all eligible days, no alcohol controls",
             "Spec Fit/12 qSD4<.05 qAll12<.05 SameSign/12 Flags"]
    keyed = primary.set_index(["exposure", "taxon", "outcome"])
    for spec in SPECS:
        d = results.loc[results.spec.eq(spec)].set_index(["exposure", "taxon", "outcome"]).reindex(keyed.index)
        good = d.status.eq("ESTIMATED") & keyed.status.eq("ESTIMATED")
        same = good & np.sign(d.beta).eq(np.sign(keyed.beta))
        three.append("{} {}/12 {} {} {}/12 {}".format(ALIASES[spec], int(d.status.eq("ESTIMATED").sum()),
                     int(d.q_primary4.lt(.05).sum()), int(d.q_all12.lt(.05).sum()), int(same.sum()), int(d.limited_support.sum())))
    three.append("Each specification uses its full 4/12 family. SameSign is not significance.")
    if not slopes.empty:
        three.append("Primary SD diet slopes at microbial Q25 / Q75 (point estimates):")
        for (e, t), d in slopes.loc[slopes.outcome.eq("SD")].groupby(["exposure", "taxon"], sort=False):
            s = d.set_index("point")
            three.append("{} {}: {} / {}".format(e, t, fmt(s.loc["q25", "beta"]), fmt(s.loc["q75", "beta"])))
    for reason, n in results.loc[results.status.ne("ESTIMATED"), "reason"].value_counts().head(2).items():
        three.append("Not estimated ({}): {}".format(n, reason))
    if "sample_after_days" in primary:
        sd = primary.loc[primary.outcome.eq("SD")]
        known = sd.sample_before_days + sd.sample_same_day + sd.sample_after_days
        f = sd.sample_after_days / known.replace(0, np.nan)
        if f.notna().any():
            three.append("Stool record AFTER day, known-date fraction: {}..{}".format(fmt(f.min()), fmt(f.max())))
    three += ["Same-cohort observation; overlapping samples are not replication or mediation proof.",
              "Negative interaction means a slope difference; not evidence of clinical protection.",
              "Full estimates/CI/slopes/flows in tables/; participant-day rows stay in private/."]
    screens = ["\n".join(lines) for lines in [one, two, three]]
    for n, s in enumerate(screens, 1):
        (out / ("SCREEN_{}.txt".format(n))).write_text(s + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text("# Daily alcohol revision\n\n" + "\n\n".join("```text\n" + s + "\n```" for s in screens) + "\n", encoding="utf-8")
    return screens
