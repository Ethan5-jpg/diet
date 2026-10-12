"""Aggregate-only output and a maximum of three compact review screens."""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json


def csv_out(frame, out, name):
    frame.to_csv(out / "tables" / name, index=False, encoding="utf-8-sig")


def write_tables(out, metadata, results, slopes, flows, families):
    for key in ["scope_audit", "score_mapping", "taxa_mapping", "inventory"]:
        csv_out(pd.DataFrame(metadata[key]), out, key + ".csv")
    csv_out(pd.DataFrame([dict(outcome=k, **v) for k, v in metadata["outcome_scales"].items()]), out, "outcome_scales.csv")
    csv_out(results, out, "all_interactions.csv")
    csv_out(results.loc[results.model.eq("primary") & results.family.eq("focus")], out, "focus24_primary.csv")
    csv_out(slopes, out, "conditional_diet_slopes.csv")
    csv_out(flows, out, "cohort_flow.csv")
    csv_out(families, out, "fdr_families.csv")


def number(x, digits=3):
    return "NA" if pd.isna(x) else ("{:." + str(digits) + "g}").format(x)


def write_screens(out, metadata, results, config, status, data_kind):
    primary = results.loc[results.model.eq("primary")]
    focus = primary.loc[primary.family.eq("focus")]
    ok = primary.status.eq("OK")
    s = metadata["summary"]
    alpha = config["alpha"]
    one = ["SCREEN 1/3 | DIET x GUT -> WHOLE CGM | " + data_kind,
           "STATUS=" + status,
           "Design: original Diet Z x original species CLR-Z; OLS HC3; observational exploration",
           "CGM={} | primary eligible={} | strict eligible={} | microbe overlap={}".format(s["cgm_n"], s["primary_eligible_n"], s["strict_eligible_n"], s["microbiome_overlap_n"]),
           "Missing covariate record={} | unknown exclusion={} | protocol source={}".format(s["missing_covariate_record_n"], s["unknown_exclusion_n"], s["protocol_available"]),
           "Primary fits={}/{} | focus={} | extension={}".format(int(ok.sum()), len(primary), len(focus), len(primary)-len(focus)),
           "Focus q<{}: {} | all-primary q<{}: {}".format(alpha, int(focus.q_focus.lt(alpha).sum()), alpha, int(primary.q_all.lt(alpha).sum())),
           "Four taxa were previously selected in this cohort. This is not independent validation.",
           "A negative interaction is a slope difference, not a negative total dietary association.",
           "Native units: Mean/SDw=mg/dL; CV=percentage points. No clinical-benefit claim."]
    two = ["SCREEN 2/3 | ALL FOCUS TESTS | " + data_kind,
           "Beta/CI: native outcome units per original Diet Z x species CLR-Z; pointwise 95% CI",
           "Diet  Taxon Outcome    N       Beta [95%CI]              qFocus qAll"]
    for _, r in focus.iterrows():
        if r.status == "OK":
            two.append("{:<5} {:<5} {:<5} {:>5} {:>7} [{:>7},{:>7}] {:>7} {:>7}{}".format(
                r.exposure, r.taxon, r.outcome, int(r.n), number(r.interaction_beta_native),
                number(r.interaction_ci_low_native), number(r.interaction_ci_high_native), number(r.q_focus), number(r.q_all),
                " *" if r.limited_joint_support else ""))
        else:
            two.append("{} {} {} N={} {}: {}".format(r.exposure, r.taxon, r.outcome, int(r.n), r.status, r.reason))
    two.append("* Limited joint tail support. Failed tests remain in multiplicity denominators.")
    three = ["SCREEN 3/3 | SENSITIVITY / DIAGNOSTICS | " + data_kind,
             "Each sensitivity has its own fixed focus family; overlapping people are not replication."]
    for model in results.model.unique():
        if model == "primary":
            continue
        subset = results.loc[results.model.eq(model)]
        good = subset.status.eq("OK")
        three.append("{:<24} fit={:>2}/{:<2} qFocus<{}={:>2} N={}..{}".format(model, int(good.sum()), len(subset), alpha,
                      int(subset.q_focus.lt(alpha).sum()), int(subset.n.min()), int(subset.n.max())))
    if "limited_joint_support" in results:
        three.append("Primary limited-support tests={}".format(int(primary.get("limited_joint_support", pd.Series(dtype=bool)).fillna(False).sum())))
    three += ["Full coefficients, double-IQR contrasts, slopes, flow and all failures: tables/",
              "All focus tests plotted, including null results. CI intervals are pointwise.",
              "Inputs/config/code hashes and versions: manifest.json",
              "No participant-level rows or IDs are included in this result package."]
    for i, lines in enumerate([one, two, three], 1):
        (out / ("SCREEN_{}.txt".format(i))).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "SCREENSHOT_SUMMARY.txt").write_text("\n\n".join("\n".join(x) for x in [one, two, three]) + "\n", encoding="utf-8")


def make_figures(out, results, slopes, config, data_kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    focus = results.loc[results.family.eq("focus")]
    fig, axes = plt.subplots(1, len(config["outcomes"]), figsize=(16, 6), squeeze=False)
    combinations = [(e, t["id"]) for e in config["focus_exposures"] for t in config["taxa"]]
    for ax, outcome in zip(axes[0], config["outcomes"]):
        for y, (exposure, taxon) in enumerate(combinations):
            for model, offset, color in [("primary", -.12, "#176B87"), ("strict", .12, "#D17A22")]:
                subset = focus.loc[focus.exposure.eq(exposure) & focus.taxon.eq(taxon) & focus.outcome.eq(outcome["id"]) & focus.model.eq(model)]
                if len(subset) and subset.iloc[0].status == "OK":
                    r = subset.iloc[0]
                    ax.errorbar(r.interaction_beta_native, y + offset,
                                xerr=[[r.interaction_beta_native-r.interaction_ci_low_native], [r.interaction_ci_high_native-r.interaction_beta_native]],
                                fmt="o", color=color, capsize=2, label=model if y == 0 else None)
                else:
                    ax.text(.98, y + offset, "NE", transform=ax.get_yaxis_transform(), ha="right", color=color, fontsize=7)
        ax.axvline(0, color="0.5", linewidth=.8)
        ax.set_yticks(range(len(combinations)))
        ax.set_yticklabels([e + " / " + t for e, t in combinations])
        ax.invert_yaxis()
        ax.set_title(outcome["id"])
        ax.set_xlabel("Interaction beta ({})".format(outcome["unit"]))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False)
    fig.suptitle("Diet x gut interaction | original Z units | pointwise 95% HC3 CI\n" + data_kind)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ["png", "pdf"]:
        fig.savefig(out / "figures" / ("focus_interactions." + ext), dpi=180)
    plt.close(fig)
    for outcome in config["outcomes"]:
        fig, axes = plt.subplots(len(config["focus_exposures"]), len(config["taxa"]), figsize=(14, 6), squeeze=False, sharey=True)
        for i, exposure in enumerate(config["focus_exposures"]):
            for j, taxon in enumerate(config["taxa"]):
                ax = axes[i, j]
                if not slopes.empty:
                    s = slopes.loc[slopes.exposure.eq(exposure) & slopes.taxon.eq(taxon["id"]) & slopes.outcome.eq(outcome["id"]) & slopes.model.eq("primary")].sort_values("microbial_original_z")
                else:
                    s = slopes
                if not s.empty:
                    points = s.drop_duplicates("microbial_original_z")
                    x = points.microbial_original_z.to_numpy(float)
                    ax.errorbar(x, points.beta_native.to_numpy(float),
                                yerr=[(points.beta_native-points.ci_low_native).to_numpy(float), (points.ci_high_native-points.beta_native).to_numpy(float)],
                                fmt="o-", color="#176B87", capsize=3)
                    ax.set_xticks(x)
                    ax.set_xticklabels(["/".join("Q{}".format(int(q)) for q in s.loc[s.microbial_original_z.eq(value), "microbial_quantile"]) for value in x])
                else:
                    ax.text(.5, .5, "Not estimable", transform=ax.transAxes, ha="center")
                ax.axhline(0, color="0.6", linewidth=.8)
                ax.set_title(exposure + " / " + taxon["id"])
                ax.set_xlabel("Species original CLR-Z quantiles")
                if j == 0:
                    unit = "pp" if outcome["unit"] == "percentage_points" else outcome["unit"]
                    ax.set_ylabel("Diet slope ({}/Diet Z)".format(unit))
        fig.suptitle(outcome["id"] + " | Conditional diet slopes, primary model | " + data_kind)
        fig.tight_layout(rect=(0, 0, 1, .94))
        for ext in ["png", "pdf"]:
            fig.savefig(out / "figures" / ("conditional_slopes_" + outcome["id"] + "." + ext), dpi=180)
        plt.close(fig)


def bundle_results(out, package):
    with zipfile.ZipFile(out / "aggregate_results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.suffix != ".zip" and path.name != "error_traceback.txt":
                archive.write(path, str(path.relative_to(out)))
