"""Static scientific figures with their plotted source rows exported."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from common import TAGS, DOMAINS, num

DOMAIN_NAMES = {"glycemic_level": "Glucose level", "glycemic_variability": "Variability", "glycemic_risk_range": "Risk / range"}


def save(ctx, fig, name):
    if getattr(ctx, "synthetic", False):
        fig.text(.5, .001, "SYNTHETIC TEST DATA - NOT HPP RESULTS", ha="center", fontsize=10, color="red")
    for extension in ["png", "pdf"]:
        fig.savefig(ctx.out / "figures" / (name + "." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)


def correlation_plot(ctx, rho, dictionary, cohort, n):
    labels = dictionary.set_index("field").label.to_dict()
    fields = [f for d in DOMAINS for f in dictionary[dictionary.domain.eq(d)].field if f in rho]
    rho = rho.loc[fields, fields]
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#dedede")
    im = ax.imshow(rho.to_numpy(), vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(fields)))
    ax.set_yticks(range(len(fields)))
    ax.set_xticklabels([labels[f] for f in fields], rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels([labels[f] for f in fields], fontsize=9)
    boundaries = np.cumsum([sum(dictionary.set_index("field").loc[f, "domain"] == d for f in fields) for d in DOMAINS])[:-1]
    for b in boundaries:
        ax.axhline(b - .5, color="black", linewidth=1)
        ax.axvline(b - .5, color="black", linewidth=1)
    for i in range(len(fields)):
        for j in range(len(fields)):
            r = rho.iloc[i, j]
            if np.isfinite(r):
                ax.text(j, i, "{:.2f}".format(r), ha="center", va="center", fontsize=6, color="white" if abs(r) > .65 else "black")
    ax.set_title("{} common-base cohort (N={}) | Spearman correlation\nExisting Z values; pairwise N in companion table; gray = unavailable".format(cohort.capitalize(), n), pad=15)
    fig.colorbar(im, ax=ax, shrink=.75, label="Spearman rho")
    fig.tight_layout()
    save(ctx, fig, "20_{}_paper20_correlation".format(cohort))


def domain_plot(ctx, domains):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    full = domains[domains.panel.eq("full20")].set_index("domain").reindex(DOMAINS)
    x = np.arange(3)
    maximum = max(float(full.candidate_paths.max()), 1.)
    for offset, col, label, color in [(-.24, "candidate_paths", "Candidates", "#bcc7cf"), (0, "primary_robust", "Primary robust", "#477da0"), (.24, "retained_both", "Retained in both", "#c36a4b")]:
        axes[0].bar(x + offset, full[col], width=.24, label=label, color=color)
        for a, v in zip(x + offset, full[col]):
            axes[0].text(a, v + maximum * .025, str(v), ha="center", fontsize=8)
    axes[0].set_ylabel("Frozen path count")
    axes[0].set_ylim(0, maximum * 1.2)
    axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=3, frameon=False)
    for offset, panel, color in [(-.24, "full20", "#477da0"), (0, "no_GMI19", "#c36a4b"), (.24, "concept_panel9", "#65a28b")]:
        sub = domains[domains.panel.eq(panel)].set_index("domain").reindex(DOMAINS)
        axes[1].bar(x + offset, sub.both_over_primary * 100, width=.24, label=panel, color=color)
    axes[1].set_ylabel("Retention among primary robust (%)")
    axes[1].set_ylim(0, 110)
    axes[1].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=3, frameon=False)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_NAMES[d] for d in DOMAINS])
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("Descriptive domain retention | unequal denominators; dependent paths\nReduced panels are post-results checks, not a new inferential family", fontsize=11)
    fig.subplots_adjust(left=.07, right=.98, bottom=.17, top=.72, wspace=.3)
    save(ctx, fig, "20_domain_retention")


def core_figures(ctx, high, dictionary):
    labels = dictionary.set_index("field").label.to_dict()
    fields = [f for d in DOMAINS for f in dictionary[dictionary.domain.eq(d) & dictionary.family.eq("paper20_primary")].field]
    core = high[num(high.domain_breadth).eq(3)].copy()
    rows = core[["taxon_id", "display_species", "exposure"]].drop_duplicates().sort_values(["taxon_id", "exposure"])
    grid = np.full((len(rows), len(fields)), np.nan)
    index = {(r.taxon_id, r.exposure): i for i, r in enumerate(rows.itertuples())}
    for r in core.to_dict("records"):
        grid[index[(r["taxon_id"], r["exposure"])] ,fields.index(r["outcome_field"])]=np.sign(float(r["model_primary__indirect_effect"]))
    fig, ax = plt.subplots(figsize=(13, max(6, .34 * len(rows))))
    cmap = ListedColormap(["#387da7", "#999999", "#c95358"]).with_extremes(bad="white")
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3))
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([labels[f] for f in fields], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["{} {} | {}".format(r.taxon_id, r.display_species.replace("_", " "), r.exposure) for r in rows.itertuples()], fontsize=8)
    for i in range(len(rows)):
        for j in range(len(fields)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, "+" if grid[i, j] > 0 else "-" if grid[i, j] < 0 else "0", ha="center", va="center", fontsize=9)
    for boundary in [3, 12]:
        ax.axvline(boundary - .5, color="#333333", linewidth=1)
    ax.set_title("Pan-domain taxa: primary indirect-effect direction\nBlank = absent from retained set; signs mean outcome increase/decrease", fontsize=11, pad=12)
    fig.tight_layout()
    save(ctx, fig, "23_core_indirect_direction")
    # Fixed concept examples, augmented by a complete matched E. rectale hPDI/NOVA4 panel.
    selected = []
    for taxon, outcome in [("T023", "cgm_mean"), ("T023", "cgm_lbgi"), ("T046", "cgm_mean"),
                           ("T046", "cgm_lbgi"), ("T096", "cgm_mean"), ("T096", "cgm_cogi")]:
        sub = core[core.taxon_id.eq(taxon) & core.outcome_field.eq(outcome)].sort_values("exposure")
        if not sub.empty:
            selected.append(sub.iloc[0])
    paired = core[core.taxon_id.eq("T068") & core.exposure.isin(["NOVA4", "hPDI"]) & core.outcome_field.isin(["cgm_cv", "cgm_modd", "cgm_sdw"])]
    selected.extend([r for _, r in paired.sort_values(["outcome_field", "exposure"]).iterrows()])
    if not selected:
        return dict(core_paths=len(core), forest_paths=0)
    selected = pd.DataFrame(selected).reset_index(drop=True)
    ctx.csv(selected, "23_forest_selected_paths.csv")
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, .95 * len(selected)), squeeze=False)
    for i, r in selected.iterrows():
        ax = axes[i, 0]
        for j, tag in enumerate(TAGS):
            base = "model_" + tag + "__"
            value = float(r[base + "indirect_effect"])
            lo, hi = float(r[base + "indirect_CI95_lower"]), float(r[base + "indirect_CI95_upper"])
            # Draw endpoints directly: percentile CI need not bracket point estimate.
            ax.plot([lo, hi], [j, j], color=["#333333", "#3885bb", "#de8a26"][j], linewidth=1.6)
            ax.plot(value, j, "o", color=["#333333", "#3885bb", "#de8a26"][j], markersize=3)
        ax.axvline(0, color="#999999", linestyle="--", linewidth=.6)
        ax.set_yticks(range(3))
        ax.set_yticklabels(TAGS, fontsize=7)
        ax.invert_yaxis()
        ax.set_ylim(2.6, -.6)
        sample_ns = "/".join(str(int(r["model_" + tag + "__N"])) for tag in TAGS)
        ax.set_title("{} | {} | {} | N(p/s/prot)={}".format(r.taxon_id, r.exposure, labels[r.outcome_field], sample_ns), loc="left", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Selected illustrative paths: original indirect effect and bootstrap 95% CI\nSeparate x axes; examples selected by concepts, not effect magnitude", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(ctx, fig, "23_three_model_illustrative_forest")
    return dict(core_paths=len(core), forest_paths=len(selected), paired_erectale_paths=len(paired))
