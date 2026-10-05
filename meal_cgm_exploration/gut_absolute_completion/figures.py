"""Full-precision six-panel forests; all candidates, NE and thresholds retained."""
from pathlib import Path
import numpy as np
from .inputs import contract

LABELS = {"baseline_glucose": "Pre-meal glucose", "mean_increment": "Net increment",
          "absolute_post_mean": "Absolute post-meal mean", "peak_rise_0_120": "Peak rise"}
COLORS = ["#0072B2", "#009E73", "#D55E00"]
MARKERS = ["o", "s", "^"]


def limits(frame):
    good = frame.loc[frame.status.eq("ESTIMATED")]
    if not len(good):
        return -1., 1.
    lo, hi = min(0., float(good.ci_low.min())), max(0., float(good.ci_high.max()))
    margin = max((hi - lo) * .12, .01)
    return lo - margin, hi + margin


def make_figures(out, kind, common, original, flow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    target = Path(out) / "figures"
    target.mkdir(exist_ok=True)
    c = contract()
    specs = [("Figure1_common_adjustment", "Common-adjustment associations", common, c["outcomes"], "q_panel12: 12 planned tests per sample and threshold"),
             ("Figure2_original_models", "Original meal-model associations", original,
              ["baseline_glucose", "mean_increment", "peak_rise_0_120"], "Original families: Net 4; Pre + Peak 8 (q_meal12 also in tables)")]
    for name, title, frame, outcomes, correction in specs:
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.1), sharey=True)
        for i, setting in enumerate(c["settings"]):
            sf = flow.loc[flow.setting.eq(setting)].set_index("threshold")
            counts = " / ".join(str(int(sf.loc[t["name"], "people"])) for t in c["thresholds"])
            for j, outcome in enumerate(outcomes):
                ax = axes[i, j]
                ax.axvline(0, color="#AAAAAA", lw=.8, zorder=0)
                ax.set_xlim(*limits(frame.loc[frame.outcome.eq(outcome)]))
                for k, taxon in enumerate(c["taxa"]):
                    for h, t in enumerate(c["thresholds"]):
                        row = frame.loc[frame.setting.eq(setting) & frame.threshold.eq(t["name"]) & frame.taxon.eq(taxon) & frame.outcome.eq(outcome)].iloc[0]
                        y = 3 - k + [.22, 0., -.22][h]
                        if row.status == "ESTIMATED":
                            error = np.array([[row.estimate - row.ci_low], [row.ci_high - row.estimate]])
                            ax.errorbar(row.estimate, y, xerr=error, fmt=MARKERS[h], markersize=4,
                                        color=COLORS[h], capsize=2, linewidth=.9, zorder=3)
                        else:
                            ax.text(.03, y, "NE", transform=ax.get_yaxis_transform(), color=COLORS[h], fontsize=6)
                ax.set_yticks([3, 2, 1, 0])
                ax.set_yticklabels(c["taxa"])
                ax.set_ylim(-.55, 3.55)
                ax.set_title(LABELS[outcome], fontsize=9, pad=8)
                ax.set_xlabel("Beta (mg/dL per original CLR-Z)", fontsize=7)
                ax.tick_params(axis="x", labelsize=7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=.15)
                if j == 0:
                    ax.set_ylabel(("Primary" if i == 0 else "Strict") + " sample", fontsize=9)
                    ax.text(0., 1.20, "N: " + counts + " (5/3, 10/5, 20/7)", transform=ax.transAxes, fontsize=8)
        legend = [Line2D([0], [0], color=COLORS[i], marker=MARKERS[i], lw=.9,
                         label="At least {} meals / {} dates".format(t["meals"], t["dates"])) for i, t in enumerate(c["thresholds"])]
        fig.suptitle(title, fontsize=12, y=.98)
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .092), ncol=3, frameon=False, fontsize=8)
        data_label = "SYNTHETIC DATA - VALIDATION ONLY" if kind == "SYNTHETIC" else "REAL DATA - SAME-VISIT EXPLORATORY ASSOCIATIONS"
        fig.text(.5, .071, data_label, ha="center", fontsize=8, color="#9B2226" if kind == "SYNTHETIC" else "#333333")
        fig.text(.5, .050, "Points: participant-level coefficients; bars: pointwise HC3 95% CIs; NE: not estimated.", ha="center", fontsize=7)
        fig.text(.5, .032, correction + ". Zero-crossing is not an FDR decision.", ha="center", fontsize=7)
        fig.text(.5, .014, "Overlapping samples; retained original exposure scales. Complete estimates and q values accompany these figures.", ha="center", fontsize=7)
        fig.subplots_adjust(left=.10, right=.985, bottom=.21, top=.875, wspace=.28, hspace=.62)
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(target / (name + "." + extension), dpi=300)
        plt.close(fig)
    (target / "FIGURE_LEGENDS.txt").write_text(
        "DATA=" + kind + "\n" +
        "Figure 1: All four candidates against Pre, Net and absolute post-meal mean at three fixed meal/date thresholds. All outcomes use the original Pre adjustment set, excluding pre-meal level/slope.\n" +
        "Figure 2: Previously completed original outcome-specific models, retained unchanged. Net/Peak adjust for pre-meal level and slope.\n" +
        "Rows show overlapping primary/strict samples. Colors/shapes distinguish thresholds, not statistical significance. Same outcome uses the same x range in both samples. Pointwise HC3 95% CIs are shown; FDR q values belong to the separately specified families.\n" +
        "T023=GGB6012_SGB8640; T046=Clostridium phoceensis; T068=Eubacterium rectale; T096=Ruthenibacterium lactatiformans.\n", encoding="utf-8")
    return matplotlib.__version__
