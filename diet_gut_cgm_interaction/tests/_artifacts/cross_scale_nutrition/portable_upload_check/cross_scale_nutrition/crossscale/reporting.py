"""Three compact aggregate-only screens for screenshot return."""
import json
from pathlib import Path

import numpy as np

from .common import require, sha256


def f(value):
    return "NA" if not np.isfinite(float(value)) else "{:.4g}".format(float(value))


def ci(row):
    return "{} [{},{}]".format(f(row.beta), f(row.low), f(row.high))


def render_screens(out, manifest, samples, tables, historical):
    r, c = tables["interactions"], tables["comparisons"]
    def estimate(variant, scale, y, taxon):
        return r.loc[r.variant.eq(variant) & r.scale.eq(scale) & r.outcome.eq(y) & r.taxon.eq(taxon)].iloc[0]
    def comparison(label, scale, y, taxon):
        return c.loc[c.label.eq(label) & c.scale.eq(scale) & c.outcome.eq(y) & c.taxon.eq(taxon)].iloc[0]
    first = ["COMMON 1/3 | INPUT / SAMPLE / DIAGNOSTICS | " + manifest["data_kind"],
             "VERSION={} | STATUS={}".format(manifest["version"], manifest["status"]),
             "Saved revised panel reused; reference reproduction {}/8; no raw CGM rebuild.".format(int(historical.passed.sum())),
             "NOVA4 primary = total-energy %; paired sensitivity = classified-energy %.",
             "B=between person means; W=within person FE, equal-person weights; original CLR-Z.",
             "Sample People Days MedianDays MinDays MedianRange MicrobeCorr"]
    for row in samples.itertuples():
        first.append("{} {} {} {} {} {} {}".format(row.sample, row.n_people, row.n_days, f(row.median_days),
                                                    row.min_days, f(row.median_total_nova_range), f(row.microbe_pearson)))
    d = tables["diagnostics"]
    first += ["Fits {}/26; interaction rows {}/44; dependent contrasts {}/40.".format(
                int(d.status.eq("ESTIMATED").sum()), int(r.status.eq("ESTIMATED").sum()), int(c.status.eq("ESTIMATED").sum())),
              "F shared across single/joint/definitions/conditional SD; N0 and N1 identities identical.",
              "Support flags={} | max fitted condition={}".format(int(r.limited_support.sum()), f(d.get("condition", np.array([np.nan])).max())),
              "qMainSD4/qMain8=F_joint; qFamily=supporting family; qAll44=all interaction rows.",
              "Contrast qFamily uses separate frozen families; all 40 also have qAll40.",
              "* = limited tail/joint-neighborhood support or >10% information from one person.",
              "Failed tests remain in denominators; CI pointwise; no participant rows printed."]
    failed = d.loc[d.status.ne("ESTIMATED")]
    if len(failed):
        first += ["NOT ESTIMATED: " + str(reason) for reason in failed.reason.unique()[:3]]
    second = ["COMMON 2/3 | TWO TAXA / BETWEEN VS WITHIN | " + manifest["data_kind"],
              "Beta: mg/dL per NOVA4 +10pp x original microbe CLR-Z +1; 95% pointwise CI.",
              "Joint fits include both taxa; Single and Joint use the same F people/days.",
              "Scale Y Taxon SingleBeta JointBeta [L95,U95] qSD4 qMain8 qAll44"]
    for scale in ["B", "W"]:
        for y in ["SD", "Mean"]:
            for t in ["T068", "T096"]:
                a, b = estimate("F_" + t, scale, y, t), estimate("F_joint", scale, y, t)
                label = t + ("*" if b.limited_support else "")
                if b.status == "ESTIMATED":
                    second.append("{} {} {} {} {} {} {} {}".format(scale, y, label, f(a.beta), ci(b), f(b.q_main_sd4), f(b.q_main_all8), f(b.q_all44)))
                else:
                    second.append("{} {} {} NOT_ESTIMATED".format(scale, y, t))
    second += ["DIRECT W-minus-B (shared-participant covariance retained):", "Y Taxon Difference [L95,U95] qSD2 qFamily4"]
    for row in c.loc[c.family.eq("scale")].itertuples():
        second.append("{} {} {} {} {}".format(row.outcome, row.taxon + ("*" if row.limited_support else ""), ci(row), f(row.q_scale_sd2), f(row.q_family)))
    second += ["B describes the observed window, not long-term habitual diet.",
               "Significant vs nonsignificant alone is not evidence of a scale/taxon difference."]
    third = ["COMMON 3/3 | SAME-SAMPLE NUTRITION / CONDITIONAL SD | " + manifest["data_kind"],
             "F=full eligible sample; N0=nutrition-complete sample, base model; N1=same days, nutrients added.",
             "N1 adds carb/protein/fiber densities AND their microbe interactions.",
             "Scale Y Taxon F_beta N0_beta N1_beta qN1 Delta_N1-N0 [CI] qDelta8"]
    for scale in ["B", "W"]:
        for y in ["SD", "Mean"]:
            for t in ["T068", "T096"]:
                a, b, e = [estimate(v, scale, y, t) for v in ["F_joint", "N0_joint", "N1_joint"]]
                delta = comparison("N1_minus_N0", scale, y, t)
                label = t + ("*" if any(row.limited_support for row in [a, b, e]) else "")
                third.append("{} {} {} {} {} {} {} {} {}".format(scale, y, label, f(a.beta), f(b.beta), f(e.beta), f(e.q_family), ci(delta), f(delta.q_family)))
    third += ["SD CONDITIONAL ON MEAN + MEAN^2 (and microbe interactions):", "Scale Taxon BaseBeta ConditionalBeta [CI] qFamily4"]
    for scale in ["B", "W"]:
        for t in ["T068", "T096"]:
            a, b = estimate("F_joint", scale, "SD", t), estimate("F_mean_adjusted_joint", scale, "SD", t)
            third.append("{} {} {} {} {}".format(scale, t + ("*" if a.limited_support or b.limited_support else ""), f(a.beta), ci(b), f(b.q_family)))
    third += ["F-to-N0 changes sample/day composition; N0-to-N1 changes adjustment only.",
              "Attenuation is not mediation; conditional SD is not a direct causal effect.",
              "Same-window observational follow-up; same cohort, no independent validation.",
              "All coefficients/slopes/40 contrasts/QC: tables/; identities stay in private/."]
    screens = ["\n".join(lines) for lines in [first, second, third]]
    for i, screen in enumerate(screens, 1):
        (Path(out) / ("SCREEN_{}.txt".format(i))).write_text(screen + "\n", encoding="utf-8")
    return screens


def saved_screens(run):
    run = Path(run).expanduser().resolve()
    m = json.loads((run / "manifest.json").read_text())
    require(m["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS", "NO_ESTIMABLE_PRIMARY_MODELS"], "Run is incomplete")
    required = ["SCREEN_{}.txt".format(i) for i in [1, 2, 3]] + [
        "tables/10_interactions.csv", "tables/11_comparisons.csv", "tables/12_model_diagnostics.csv", "config.json"]
    for name in required:
        require(sha256(run / name) == m["output_sha256"][name], "Saved result changed: " + name)
    return [(run / "SCREEN_{}.txt".format(i)).read_text() for i in [1, 2, 3]]
