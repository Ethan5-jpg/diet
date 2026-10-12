"""Aggregate-only, three-screen result summaries for the screenshot workflow."""
import numpy as np
import pandas as pd

from .common import sha256, write_json
from .models import SPECS


def fmt(value, digits=4):
    try:
        return format(float(value), ".{}g".format(digits)) if np.isfinite(float(value)) else "NA"
    except (TypeError, ValueError):
        return "NA"


def table(out, name, frame):
    path = out / "tables" / name
    frame.to_csv(path, index=False)
    return path


def qc_tables(out, panel, people, feasible, audit):
    rows = []
    for label, d in [("matched", panel), ("diet_quality", panel.loc[panel.diet_pass]),
                     ("cgm_quality", panel.loc[panel.cgm_pass]),
                     ("both_quality", panel.loc[panel.diet_pass & panel.cgm_pass]),
                     ("eligible_both_quality", panel.loc[panel.diet_pass & panel.cgm_pass & panel.primary_eligible])]:
        rows.append(dict(stage=label, people=d.participant_id.nunique(), days=len(d)))
    table(out, "01_day_flow.csv", pd.DataFrame(rows))
    table(out, "02_feasibility.csv", feasible)
    reasons = panel.diet_reasons.fillna("").str.split(";").explode()
    table(out, "03_diet_exclusions.csv", reasons[reasons.ne("")].value_counts().rename_axis("reason").reset_index(name="days"))
    distributions = []
    for col in ["cgm_coverage", "max_gap_minutes", "min_quarter_coverage", "nova_coverage",
                "fiber_g_energy_coverage", "alcohol_g_energy_coverage", "logging_span_hours", "formula_energy_events"]:
        s = panel[col].dropna()
        distributions.append(dict(variable=col, nonmissing=len(s), p10=s.quantile(.1), p50=s.quantile(.5), p90=s.quantile(.9)))
    table(out, "04_day_quality_distribution.csv", pd.DataFrame(distributions))
    person_counts = panel.loc[panel.diet_pass & panel.cgm_pass & panel.primary_eligible].groupby("participant_id").size()
    table(out, "05_valid_day_count_distribution.csv", person_counts.value_counts().sort_index().rename_axis("valid_days").reset_index(name="people"))
    table(out, "06_sample_timing_identity.csv", people.timing_status.value_counts().rename_axis("status").reset_index(name="people"))
    write_json(out / "input_audit.json", audit)


def write_screens(out, manifest, feasible=None, results=None, slopes=None):
    kind, status = manifest["data_kind"], manifest["status"]
    one = ["DAILY 1/3 | DATA / FEASIBILITY | " + kind, "STATUS=" + status,
           "Person FE; participant-cluster CR1; same-visit observational associations.",
           "CGM: observed 15-min bins; local calendar days; no gap interpolation."]
    summary = manifest.get("person_summary", {})
    if summary:
        one.append("Selected connections={selected}; covariate missing={missing_cov}; exclusion unknown={unknown}; both taxa={both_taxa}".format(**summary))
    audit = manifest.get("build_audit", {})
    if audit:
        one.append("Diet days={diet_days}; CGM days={cgm_days}; matched={matched_days}".format(**audit))
        one.append("Timing-invalid people: diet={diet_time_issue_people}; CGM={cgm_time_issue_people}".format(**audit))
    if feasible is not None:
        one.append("Exposure Taxon People Days MedianDays Ready | QCdays MissDiet MissAlc MissGut")
        for r in feasible.itertuples():
            one.append("{:<8} {:<5} {:>6} {:>7} {:>10} {} | {} {} {} {}".format(
                r.exposure, r.taxon, r.people, r.days, fmt(r.days_median), bool(r.ready),
                r.quality_days, r.missing_exposure_days, r.missing_alcohol_days, r.missing_microbe_days))
        one.append("Missing counts overlap within QCdays, before complete-case / minimum-day filters.")
    one += ["Diet-day QC is a recording proxy; unlogged meals cannot be detected fully.",
            "NOVA4: classified-energy share; Fiber: g/1000 kcal. CLR-Z is relative composition."]
    two = ["DAILY 2/3 | ALL 12 PRIMARY-SPEC INTERACTIONS | " + kind,
           "Beta per NOVA4 +10pp or Fiber +5g/1000kcal, times original microbe CLR-Z +1.",
           "SD/Mean: mg/dL; CV: percentage points. Pointwise 95% CI.",
           "Diet Taxon Y People Days Beta [L95,U95] qSD4 qAll12"]
    three = ["DAILY 3/3 | SENSITIVITY / INTERPRETATION | " + kind]
    if results is None:
        two.append("No models requested or available. BUILD_COMPLETE is not an analysis result.")
        three.append("Next: analyze the exact completed build; see README for the command.")
    else:
        primary = results.loc[results.spec.eq("primary")]
        for r in primary.itertuples():
            if r.status != "ESTIMATED":
                two.append("{} {} {} N={} D={} NOT_ESTIMATED".format(r.exposure, r.taxon, r.outcome, r.n_people, r.n_days))
            else:
                two.append("{} {} {} {} {} {} [{},{}] {} {}{}".format(r.exposure, r.taxon, r.outcome,
                           r.n_people, r.n_days, fmt(r.beta), fmt(r.low), fmt(r.high), fmt(r.q_primary4), fmt(r.q_all12), " *" if r.limited_support else ""))
        two.append("qSD4 = frozen four SD tests; qAll12 = all 12. Failed tests stay in denominators.")
        two.append("* = sparse microbial tails, zero microbial IQR, or >10% focal information from one person.")
        three.append("Specification Fit/12 SD q4<.05 All q12<.05 SupportFlags")
        for spec in SPECS:
            d = results.loc[results.spec.eq(spec)]
            three.append("{:<20} {:>2}/12 {:>8} {:>10} {:>10}".format(spec, int(d.status.eq("ESTIMATED").sum()),
                         int(d.q_primary4.lt(manifest["config"]["analysis"]["alpha"]).sum()),
                         int(d.q_all12.lt(manifest["config"]["analysis"]["alpha"]).sum()), int(d.limited_support.sum())))
        if slopes is not None and not slopes.empty:
            three.append("Primary SD diet slopes at microbial Q25 / Q75 (point estimates):")
            for (exposure, taxon), d in slopes.loc[slopes.outcome.eq("SD")].groupby(["exposure", "taxon"], sort=False):
                s = d.set_index("point")
                three.append("{} {}: {} / {}".format(exposure, taxon, fmt(s.loc["q25", "beta"]), fmt(s.loc["q75", "beta"])))
        failed = results.loc[results.status.ne("ESTIMATED"), "reason"]
        for reason, count in failed.value_counts().head(2).items():
            three.append("Not estimated ({}): {}".format(count, reason))
        sd_primary = primary.loc[primary.outcome.eq("SD")]
        if "sample_after_days" in sd_primary:
            known = sd_primary.sample_before_days.fillna(0) + sd_primary.sample_same_day.fillna(0) + sd_primary.sample_after_days.fillna(0)
            fraction = sd_primary.sample_after_days / known.replace(0, np.nan)
            if fraction.notna().any():
                three.append("Stool record AFTER day, known-date fraction across SD models: {}..{}".format(fmt(fraction.min()), fmt(fraction.max())))
    three += ["Overlapping samples are not replication; negative interaction is a slope difference.",
              "Single stool measurement is not a baseline predictor or proof of causal mediation.",
              "Full estimates/slopes/flows: tables/. Participant/day records stay in private/."]
    screens = ["\n".join(one), "\n".join(two), "\n".join(three)]
    for i, text in enumerate(screens, 1):
        (out / ("SCREEN_{}.txt".format(i))).write_text(text + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text("# Daily diet–gut–CGM\n\n" + "\n\n".join("```text\n" + s + "\n```" for s in screens) + "\n", encoding="utf-8")
    return screens


def output_hashes(out):
    # Include the reusable private panel so a resumed fit cannot mix changed data.
    files = list((out / "tables").glob("*.csv")) + list(out.glob("SCREEN_*.txt")) + [out / "SUMMARY.md"]
    for p in [out / "private" / "daily_panel.csv", out / "input_audit.json", out / "analysis_config.json"]:
        if p.is_file(): files.append(p)
    return {str(p.relative_to(out)): sha256(p) for p in files if p.is_file()}
