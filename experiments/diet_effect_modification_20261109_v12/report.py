"""Three aggregate-only English screens; never print participant rows."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from bridge import Memo, sha
from inference import paired_interval, partial_inference, prediction_metrics
from models import specifications
from runtime import valid_id

ROOT = Path(__file__).resolve().parent
ASSOCIATION_NOTE = "PERSON-LEVEL ASSOCIATIONS; sample-calendar meaning unverified; no meal/causal response."


def ordered(parts, key, name=None):
    rows = np.concatenate([part["rows"] for part in parts])
    if len(set(rows.tolist())) != len(rows):
        raise ValueError("A participant occurs twice within one repeat")
    values = np.concatenate([part[key] if name is None else part[key][name] for part in parts])
    return values[np.argsort(rows)]


def aggregate(completed, config, groups, source):
    if len(completed) != config["outer_repeats"] * config["outer_folds"]:
        raise ValueError("All ten outer folds are required for final aggregation")
    parts = [[completed[r, f] for f in range(config["outer_folds"])] for r in range(2)]
    ids = [np.sort(np.concatenate([part["rows"] for part in items])) for items in parts]
    if not np.array_equal(ids[0], ids[1]):
        raise ValueError("Repeated evaluation participants differ")
    group = np.asarray(groups)[ids[0]]
    names = ["main"] + [spec["name"] for spec in specifications()]
    inference, raw_inference, scores = [], [], []
    assembled = []
    for repeat, items in enumerate(parts):
        y = ordered(items, "y")
        markers_iqr = np.mean([part["iqr"] for part in items], axis=0)
        target_residual = ordered(items, "target_residual")
        canonical = {row: j for j, row in enumerate(ids[repeat])}
        main_blocks = [dict(rows=[canonical[row] for row in part["rows"]], q=part["inference_main_q"]) for part in items]
        kwargs = dict(groups=group, iqr=markers_iqr, draws=config["score_multipliers"],
                      seed=config["model_seed"] + repeat, minimum_groups=config["minimum_households_for_inference"], main_blocks=main_blocks,
                      minimum_effective_groups=config["minimum_effective_households_for_inference"])
        inference.append(partial_inference(ordered(items, "y_residual"), target_residual, **kwargs))
        raw_inference.append(partial_inference(ordered(items, "raw_y_residual"), target_residual, **kwargs))
        value = dict(y=y, support=ordered(items, "support"), prediction={}, component={}, contrast={}, correction={})
        for name in names:
            value["prediction"][name] = ordered(items, "prediction", name)
            value["component"][name] = ordered(items, "interaction_component", name)
            value["contrast"][name] = ordered(items, "total_contrast", name)
        base = value["prediction"]["main"]
        base_r2 = prediction_metrics(y, base)["R2"]
        for name in names:
            value["correction"][name] = value["contrast"][name] - value["contrast"]["main"]
            prediction = value["prediction"][name]
            metrics = prediction_metrics(y, prediction)
            component = value["component"][name]
            support = value["support"]
            row = dict(model=name, repeat=repeat, **metrics)
            row.update(
                delta_R2=metrics["R2"] - base_r2,
                delta_R2_ci_conditional=[0.0, 0.0] if name == "main" else paired_interval(y, base, prediction, group, config["prediction_bootstraps"], config["model_seed"] + repeat),
                interaction_component_sd_mg_dl=float(np.std(component)),
                total_contrast_mean_mg_dl=float(np.mean(value["contrast"][name])),
                total_contrast_sd_mg_dl=float(np.std(value["contrast"][name])),
                change_from_null_contrast_sd_mg_dl=float(np.std(value["contrast"][name] - value["contrast"]["main"])),
                supported_interaction_sd_mg_dl=float(np.std(component[support])) if support.sum() >= 30 else None,
                selected_interaction_folds=0 if name == "main" else sum(part["choices"][name] is not None for part in items),
                nonzero_product_contrast_folds=int(sum(np.std(part["interaction_component"][name]) > 1e-10 for part in items)),
                nonzero_correction_contrast_folds=int(sum(np.std(part["total_contrast"][name] - part["total_contrast"]["main"]) > 1e-10 for part in items)),
            )
            scores.append(row)
        assembled.append(value)
    shared_support = assembled[0]["support"] & assembled[1]["support"]
    stability = []
    for name in names[1:]:
        a, b = [value["component"][name] for value in assembled]
        ca, cb = [value["correction"][name] for value in assembled]
        nonconstant = np.std(a) > 1e-10 and np.std(b) > 1e-10
        stability.append(dict(
            model=name,
            contrast_pearson=float(np.corrcoef(a, b)[0, 1]) if nonconstant else None,
            contrast_spearman=float(spearmanr(a, b).statistic) if nonconstant else None,
            correction_contrast_pearson=float(np.corrcoef(ca, cb)[0, 1]) if np.std(ca) > 1e-10 and np.std(cb) > 1e-10 else None,
            shared_supported_n=int(shared_support.sum()),
            supported_contrast_pearson=float(np.corrcoef(a[shared_support], b[shared_support])[0, 1])
            if shared_support.sum() >= 30 and np.std(a[shared_support]) > 1e-10 and np.std(b[shared_support]) > 1e-10 else None,
        ))
    return dict(
        status="COMPLETE", evidence_source=source, n=len(ids[0]), households=len(set(group)),
        primary_repeat=0, inference=inference, raw_outcome_sensitivity=raw_inference,
        scores=scores, stability=stability,
        support_fraction=[float(v["support"].mean()) for v in assembled],
        overlap_diagnostic="Training-neighbor central diet range only; no causal positivity guarantee and no participant exclusions",
        fold_diagnostics=[dict(repeat=r, fold=f, nuisance_rank=part["nuisance_rank"], main_columns=part["main_columns"],
                               main_penalty=part["main_penalty"], choices=part["choices"], warnings=part["warnings"])
                          for (r, f), part in sorted(completed.items())],
        contrast=config["contrast"], confirmation_outcomes_read=False, participant_data_exported=False,
        temporal_scope="Person-average exposure/person outcome; recorded microbiome calendars mostly after CGM; exact date semantics unverified; no strict daily alignment or within-person response estimand",
        background_scope="Identical full nutrition/lifestyle/UPF and both-site nonlinear main basis with nonmicrobial diet-background terms; equality does not prove nuisance adequacy; absolute null R2 must be reported",
        inference_limitations="Exploratory model-dependent finite-basis inference, with fold-specific nuisance coefficients and CR2-style household leverage adjustment. Fixed feature-map/working-model assumptions and historical adaptive discovery searches are not corrected.",
        prediction_interval_scope="Paired household bootstrap of fixed OOF predictions, conditional on fitted pipeline; no refitting or search correction",
        repeat_scope="Same participants twice; repeat 1 is split sensitivity, never independent replication",
    )


def number(value, digits=4):
    return "NA" if value is None else ("%.*f" % (digits, value))


def screen(result, mode=1):
    header = "%s | V12 fiber_density -> D_Glu (mg/dL) | N=%d HH=%d" % (result["evidence_source"], result["n"], result["households"])
    lines = [header]
    if mode == 1:
        lines += ["PRIMARY: joint linear fiber x [oral/gut Shannon, dominance]; repeat 0", "Repeat 1 is split sensitivity on the SAME people."]
        for repeat, info in enumerate(result["inference"]):
            if info["status"] == "OK":
                lines.append("repeat=%d PRIMARY_global_p=%s diagnostic_joint_p=%s oral_Holm=%s gut_Holm=%s" %
                             (repeat, number(info["omnibus"]["p_primary"]), number(info["omnibus"]["p_multiplier"]),
                              number(info["sites"]["oral"]["p_holm_sites"]), number(info["sites"]["gut"]["p_holm_sites"])))
            else:
                lines.append("repeat=%d UNAVAILABLE: %s" % (repeat, info["reason"]))
        first, second = result["inference"]
        minimum_information = [min(info["effective_households_per_coordinate"]) if "effective_households_per_coordinate" in info else None for info in (first, second)]
        lines.append("Effective HH minimum across four coordinates: r0=%s r1=%s (diagnostic, not df)" % tuple(number(value, 1) for value in minimum_information))
        if first["status"] == "OK":
            lines += ["10->15 fiber contrast: modifier Q75 minus Q25, other terms held equal", "%-19s %9s %22s %9s %9s" % ("marker", "diff0", "family95%CI0", "diff1", "Holm0")]
            for j, marker in enumerate(first["markers"]):
                other = second["markers"][j]["interaction_iqr_difference_mg_dl"] if second["status"] == "OK" else None
                ci = marker["interaction_iqr_ci_bonferroni95_mg_dl"]
                lines.append("%-19s %+9.3f [%+9.3f,%+9.3f] %9s %9s" %
                             (marker["name"], marker["interaction_iqr_difference_mg_dl"], ci[0], ci[1], number(other, 3), number(marker["p_holm_markers"])))
        lines += ["Primary global: Bonferroni union; joint Wald/multiplier is diagnostic only.",
                  "Shown CIs: four-marker Bonferroni family 95%; pointwise CIs stay in summary.json.",
                  "Follow-ups after primary test; no causal or historical search correction."]
    elif mode == 2:
        lookup = {(row["model"], row["repeat"]): row for row in result["scores"]}
        lines.append("MATCHED additive R2_0=%s R2_1=%s" % tuple(number(lookup["main", r]["R2"], 6) for r in range(2)))
        lines += ["All 12 planned models; dR2 vs SAME additive background, including negatives.",
                  "%-26s %10s %10s %7s %7s %15s %15s" % ("model", "dR2_0", "dR2_1", "opt0/1", "nz0/1", "rawSD0/1", "corrSD0/1")]
        for spec in specifications():
            a, b = [lookup[spec["name"], r] for r in range(2)]
            lines.append("%-26s %+10.6f %+10.6f %3d/%-3d %3d/%-3d %7.3f/%-7.3f %7.3f/%-7.3f" %
                         (spec["name"], a["delta_R2"], b["delta_R2"], a["selected_interaction_folds"], b["selected_interaction_folds"],
                          a["nonzero_product_contrast_folds"], b["nonzero_product_contrast_folds"],
                          a["interaction_component_sd_mg_dl"], b["interaction_component_sd_mg_dl"],
                          a["change_from_null_contrast_sd_mg_dl"], b["change_from_null_contrast_sd_mg_dl"]))
        lines += ["opt=positive option selected; nz=nonzero raw-product contrast SD; neither is a test.",
                  "rawSD/corrSD=raw product/full added 10->15 contrast SD, mg/dL; not observed responses."]
    else:
        lines += ["DIAGNOSTICS: conditional uncertainty, support and split sensitivity",
                  "Local contrast support fraction: r0=%.3f r1=%.3f" % tuple(result["support_fraction"])]
        for r, info in enumerate(result["raw_outcome_sensitivity"]):
            lines.append("Raw-outcome sensitivity r%d primary_global_p=%s" % (r, number(info["omnibus"]["p_primary"]) if info["status"] == "OK" else "UNAVAILABLE"))
        lines.append("%-26s %15s %25s %25s" % ("model", "raw_r/corr_r", "dR2 CI0", "dR2 CI1"))
        lookup = {(row["model"], row["repeat"]): row for row in result["scores"]}
        for item in result["stability"]:
            intervals = [lookup[item["model"], r]["delta_R2_ci_conditional"] for r in range(2)]
            correlations = number(item["contrast_pearson"], 3) + "/" + number(item["correction_contrast_pearson"], 3)
            lines.append("%-26s %15s [%+.5f,%+.5f] [%+.5f,%+.5f]" %
                         (item["model"], correlations, *intervals[0], *intervals[1]))
        lines += ["dR2 intervals resample households with fitted OOF predictions FIXED.",
                  "r compares raw-product/full-correction contrasts on repeated people, not replication.",
                  "Support is a range diagnostic; low support limits contrast interpretation.",
                  "Warnings=%d; complete loadings/choices remain in TRE_ONLY checkpoints." % sum(bool(v["warnings"]) for v in result["fold_diagnostics"])]
    lines.append(ASSOCIATION_NOTE)
    return "\n".join(lines) + "\n"


def completed_results(run, config):
    if not (Path(run) / "TRE_ONLY").is_dir():
        return {}
    memo, result = Memo(Path(run) / "TRE_ONLY"), {}
    for repeat in range(2):
        for fold in range(config["outer_folds"]):
            key = "r%d_f%d_COMPLETE" % (repeat, fold)
            # Memo.get creates a directory for a missing key. Avoid that write
            # when this function is used as an incomplete-run reader.
            if not (memo.root / key).is_dir():
                continue
            try:
                result[repeat, fold] = memo.get(key)
            except FileNotFoundError:
                pass
    return result


def verified_progress_units(run):
    """Count each completed unit once without creating files or unpickling data.

    A process can leave a partial receipt and later complete another receipt for
    the same key. Match Memo's first valid-receipt behavior, not a filename count.
    """
    private = Path(run) / "TRE_ONLY"
    completed = 0
    for repeat in range(2):
        for fold in range(5):
            key = "r%d_f%d_COMPLETE" % (repeat, fold)
            folder = private / key
            if not folder.is_dir():
                continue
            if folder.is_symlink():
                raise ValueError("Checkpoint directory must not be a symlink")
            for receipt in sorted(folder.glob("done_*.json")):
                try:
                    record = json.loads(receipt.read_text())
                except json.JSONDecodeError:
                    continue
                if (not isinstance(record, dict) or not isinstance(record.get("file"), str)
                        or not isinstance(record.get("sha256"), str)):
                    raise ValueError("Malformed completed checkpoint receipt: " + key)
                name = record["file"]
                path = folder / name
                if (Path(name).name != name or path.is_symlink() or not path.is_file()
                        or sha(path) != record["sha256"]):
                    raise ValueError("Corrupted checkpoint: " + key)
                completed += 1
                break
    return completed


def show(run, mode=1, progress=False):
    run = Path(run)
    if progress:
        completed = verified_progress_units(run)
        aggregate_state = "present" if (run / "summary.json").is_file() and (run / "completion.json").is_file() else "missing"
        print("RUN=%s completed_outer=%d/10 final_aggregate=%s running_lock=%s" %
              (run.name, completed, aggregate_state, (run / "RUNNING.lock").exists()))
        return
    path, proof = run / "summary.json", run / "completion.json"
    if not proof.is_file() or not path.is_file():
        print("INCOMPLETE: final audited aggregate is not available; use --progress.")
        return
    record = json.loads(proof.read_text())
    if (record["summary_sha256"] != sha(path) or record["outer_folds_complete"] != 10
            or record["registration_sha256"] != sha(run / "registration.json")):
        raise ValueError("Final aggregate checksum/completion mismatch")
    for name, digest in record["checkpoint_files"].items():
        if sha(run / "TRE_ONLY" / name) != digest:
            raise ValueError("Completed checkpoint checksum mismatch")
    print(screen(json.loads(path.read_text()), mode), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="effect_modification_v12_20261109")
    parser.add_argument("--screen", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    show(ROOT / "runs" / valid_id(args.run_id), args.screen, args.progress)
