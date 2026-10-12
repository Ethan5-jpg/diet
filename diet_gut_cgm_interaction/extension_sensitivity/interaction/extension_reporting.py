"""Aggregate-only follow-up tables, all-extension plots and compact review screens."""
import numpy as np
import pandas as pd

from .extension_sensitivity import KEY, MODELS, MODEL_CODES
from .reporting import csv_out, number


def write_followup_tables(out, results, slopes, flows, families, comparison, stability, summary):
    tables = {"all_interactions.csv": results, "extension_all_specs.csv": results[results.family.eq("extension")],
              "conditional_diet_slopes.csv": slopes, "cohort_flow.csv": flows, "fdr_families.csv": families,
              "reference_reproduction.csv": comparison, "reference_hits_stability.csv": stability,
              "retention_by_specification.csv": summary}
    for name, frame in tables.items():
        csv_out(frame, out, name)


def write_followup_screens(out, results, config, comparison, stability, summary, status, kind, plots="COMPLETE"):
    extension = results[results.family.eq("extension")]
    one = ["EXTENSION 1/3 | SIX SENSITIVITIES | " + kind, "STATUS=" + status,
           "Design: same-cohort observational follow-up after primary results; no independent validation",
           "All fits={}/588 | extension={}/420 (60 primary + 360 sensitivity)".format(int(results.status.eq("OK").sum()), int(extension.status.eq("OK").sum())),
           "Reference reproduction={}/204 | original primary and five old focus sensitivities".format(int(comparison.matches.sum())),
           "Every specification: qAll84 BH across 84 tests; failed/skipped tests stay in denominator.",
           "U = common_all7_sample: seven diets + four taxa + three outcomes + required covariates.",
           "Original significant extension targets={} | same direction all six={} | retained all six={}".format(
               len(stability), int(stability.same_direction_all_six.sum()), int(stability.retained_all_six.sum())),
           "Retained = estimable + same interaction sign + qAll84<{}; overlapping samples are not replication.".format(config["alpha"])]
    two = ["EXTENSION 2/3 | ORIGINAL EXTENSION TARGETS | " + kind,
           "S=strict P=protocol_same A=protocol_adjusted Q=quadratic C=covariate_interactions U=all7",
           "Every q column is qAll84; Same6=all six estimable and same sign; Keep=retained / 6",
           "Diet      Taxon Y    Nmin  qBase      S      P      A      Q      C      U Same6 Keep"]
    for _, row in stability.head(24).iterrows():
        diet = "Carb_pct" if row.exposure == "Carbohydrate_pct" else row.exposure
        qs = " ".join("{:>6}".format(number(row[m + "__q_all84"], 3)) for m in MODELS)
        two.append("{:<9} {:<5} {:<4} {:>5} {:>6} {} {:>5} {:>4}".format(
            diet, row.taxon, row.outcome, int(row.min_sensitivity_n), number(row.reference_q_all84, 3), qs,
            "yes" if row.same_direction_all_six else "no", int(row.retained_sensitivity_count)))
    if stability.empty:
        two.append("No original significant extension targets; all 60 tests are still reported in tables/.")
    if len(stability) > 24:
        two.append("First 24 targets shown in registry order; full table contains {}. No q-based truncation.".format(len(stability)))
    three = ["EXTENSION 3/3 | ALL 60 EXTENSION TESTS PER SPECIFICATION | " + kind,
             "Spec Fit/60 qAll84<alpha OldRetained NewOutsideOld Nrange SupportFlags"]
    for _, row in summary.iterrows():
        three.append("{:>4} {:>2}/60 {:>13} {:>11} {:>13} {:>5}..{:<5} {:>5}".format(
            MODEL_CODES[row.model], int(row.extension_fitted), int(row.extension_q_all84_below_alpha),
            int(row.reference_hits_retained), int(row.new_significant_outside_reference), int(row.n_min), int(row.n_max), int(row.limited_support_tests)))
    three += ["NewOutsideOld is descriptive; new hits do not enter the frozen original-target set.",
              "Coefficients/pointwise CIs/units/failed fits/flows: tables/; all-extension plots="+plots,
              "Original units preserved: Mean/SDw mg/dL; CV percentage points. Negative interaction is a slope difference.",
              "No participant rows or IDs exported. Full precision and reference/input/code hashes: manifest.json."]
    for i, lines in enumerate([one, two, three], 1):
        (out / "SCREEN_{}.txt".format(i)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "SCREENSHOT_SUMMARY.txt").write_text("\n\n".join("\n".join(lines) for lines in [one, two, three]) + "\n", encoding="utf-8")


def make_followup_figures(out, results, config, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["primary"] + MODELS
    colors = ["#20252B", "#296A9C", "#8398A9", "#27958A", "#C78027", "#805AA6", "#BE5168"]
    labels = ["Primary", "Strict", "Protocol same sample", "Protocol adjusted", "Quadratic",
              "Covariate interactions", "Common seven diets"]
    combinations = [(e, t["id"]) for e in config["extension_exposures"] for t in config["taxa"]]
    lookup = results.set_index(KEY + ["model"])
    for outcome in config["outcomes"]:
        fig, ax = plt.subplots(figsize=(12, 15))
        for y, (diet, taxon) in enumerate(combinations):
            if y % 2 == 0:
                ax.axhspan(y-.48, y+.48, color="#F3F5F7", zorder=0)
            for model, offset, color, label in zip(models, np.linspace(-.32, .32, 7), colors, labels):
                r = lookup.loc[(diet, taxon, outcome["id"], model)]
                if r.status == "OK":
                    ax.errorbar(r.interaction_beta_native, y+offset,
                                xerr=[[r.interaction_beta_native-r.interaction_ci_low_native], [r.interaction_ci_high_native-r.interaction_beta_native]],
                                fmt="o", markersize=3, color=color, capsize=2, linewidth=.9)
                else:
                    ax.text(.99, y+offset, "NE", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, color=color)
        for color, label in zip(colors, labels):
            ax.plot([], [], "o", color=color, markersize=4, label=label)
        ax.axvline(0, color="#777777", linewidth=.8)
        ax.set_yticks(range(len(combinations)))
        ax.set_yticklabels([e+" / "+t for e,t in combinations], fontsize=9)
        ax.set_ylim(len(combinations)-.4, -.6)
        ax.set_xlabel("Interaction per original Diet Z x microbial CLR-Z ({})".format(outcome["unit"]))
        ax.set_title(outcome["id"]+" | All 60 extensions across outcome panels\nPointwise 95% HC3 CI | "+kind, fontsize=12, pad=58)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        fig.subplots_adjust(left=.27, right=.97, bottom=.06, top=.88)
        for ext in ["png", "pdf"]:
            fig.savefig(out / "figures" / ("extension_sensitivity_"+outcome["id"]+"."+ext), dpi=150)
        plt.close(fig)
