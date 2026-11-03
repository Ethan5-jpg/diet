"""Bounded discovery pilot; TRE data and individual outputs stay in TRE."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from scipy import linalg
from core import Background, cluster_fit, metrics, rcs, save_npz


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(x) for x in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def host_design(index, training, u, h):
    """Reproduce W19 HostEncoder exactly, without importing its UPF machinery."""
    numeric = index[["age_calendar_years", "sex_code", "mean_daily_energy_fiber"]].to_numpy(float)
    mean, sd = numeric[training].mean(0), numeric[training].std(0)
    sd[sd < 1e-12] = 1.
    numeric = (numeric-mean)/sd
    levels = sorted(set(index.loc[training, "smoking_status"].astype(str)) | {"__UNKNOWN__"})
    category = index.smoking_status.astype(str).to_numpy()
    category = np.where(np.isin(category, levels), category, "__UNKNOWN__")
    parts = [np.ones((len(index), 1)), numeric]
    labels = ["intercept", "age", "sex", "energy"]
    for level in levels[1:]:
        parts.append((category == level)[:, None].astype(float))
        labels.append("smoking=="+level)
    parts.extend([u[:, None], h[:, None], (u*numeric[:, 0])[:, None], (u*numeric[:, 1])[:, None]])
    labels.extend(["diet_linear", "diet_RCS", "diet_age", "diet_sex"])
    raw = np.column_stack(parts)
    _, r, piv = linalg.qr(raw[training], mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > max(diagonal.max(initial=0), 1)*1e-10))
    keep = np.sort(piv[:rank])
    return raw[:, keep], dict(numeric_means=mean, numeric_scales=sd,
                             smoking_levels=levels, keep=keep, labels=[labels[j] for j in keep])


def model_specs(config):
    specs = []
    for site in config["sites"]:
        for representation in config["ridge_representations"]:
            specs.append(dict(name=f"ridge_{representation}_{site}", method="ridge",
                              representation=representation, site=site, nonlinear=False,
                              grid=[dict(penalty=x) for x in config["ridge_lambdas"]]))
        specs.append(dict(name=f"elastic_net_CLR_{site}", method="elastic_net",
                          representation="CLR", site=site, nonlinear=False,
                          grid=[dict(alpha=a, fraction=f) for a in config["elastic_net_alphas"]
                                for f in config["elastic_net_lambda_fractions"]]))
        specs.append(dict(name=f"nonlinear_PC5_{site}", method="ridge",
                          representation="PC5", site=site, nonlinear=True,
                          grid=[dict(penalty=x) for x in config["ridge_lambdas"]]))
    return specs


def modifier(data, specification):
    pieces, blocks = [], []
    sites = ["oral", "gut"] if specification["site"] == "oral_gut" else [specification["site"]]
    for site in sites:
        if specification["representation"] == "CLR":
            z = data[site+"_z"]
        else:
            k = int(specification["representation"][2:])
            z = data[site+"_PC"][:, :k]
            if z.shape[1] != k:
                raise ValueError("Not enough training PCs")
        for diet_basis in (["u", "h"] if specification["nonlinear"] else ["u"]):
            start = sum(p.shape[1] for p in pieces)
            pieces.append(data[diet_basis][:, None]*z)
            blocks.append(dict(site=site, basis=diet_basis, start=start, stop=start+z.shape[1], z=z))
    return np.column_stack(pieces), blocks


def contrast_axes(fit, blocks, training, diet, contrast):
    low, high = (np.asarray(contrast)-diet["center"])/diet["scale"]
    differences = dict(u=high-low, h=float(rcs(high, diet["knots"])-rcs(low, diet["knots"])))
    axes, centers, scales = {}, {}, {}
    for block in blocks:
        site = block["site"]
        contribution = differences[block["basis"]]*(block["z"]@fit["interaction_coef"][block["start"]:block["stop"]])
        axes[site] = axes.get(site, np.zeros(len(contribution)))+contribution
    for site in axes:
        centers[site] = float(axes[site][training].mean())
        axes[site] = axes[site]-centers[site]
        scales[site] = float(axes[site][training].std())
    return axes, centers, scales


def run(args):
    start = time.monotonic()
    package, source = Path(args.package).resolve(), Path(__file__).resolve().parent
    config = json.loads((source/"config.json").read_text())
    if config["confirmation_Y_access"] or config["formal_confirmation_tests"]:
        raise ValueError("This script is discovery-only")
    project = package.parent
    expected_name = "collaborator_handoff_20261102_v1"
    if package.name != expected_name:
        raise ValueError(f"Expected {expected_name}")
    loader_path = package/"examples/load_fiber_fold.py"
    spec = importlib.util.spec_from_file_location("hpp_existing_fold_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    if Path(loader.ROOT).resolve() != package:
        raise ValueError("Fold loader points to a different input package")
    old = project/"data_qc/w20_simplified_20261101_v2/results/models/D_Glu/fiber_density"
    old_config = json.loads((old/"scores_and_predictor.json").read_text())
    order = json.loads((old.parent.parent.parent/"configuration_order.json").read_text())
    null_index = old_config["selection"]["main_only_selected_index"]
    ridge_index = old_config["selection"]["interaction_selected_index"]
    expected = (config["background_PC_per_site"], config["main_lambda"])
    if tuple(order[null_index][:2]) != expected or tuple(order[ridge_index][:2]) != expected:
        raise ValueError("Archived baseline configuration changed; review before fitting")
    with np.load(io.BytesIO((old/"selected_CV_losses_TRE_ONLY.npz").read_bytes()), allow_pickle=False) as archive:
        archived_prediction = archive["all_predictions"]
        archived_y = archive["fold_Y"]
    # Confirm required dependencies before creating results.
    import pyarrow
    output = Path(args.output).resolve()
    if package/"experiments" not in output.parents:
        raise ValueError("Output must be under this package's experiments/")
    required = [loader_path, package/"preprocessing/code/preprocessing_v1_20261030.py",
                package/"inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet",
                package/"inputs/discovery_v2/D_Glu_raw_Y_TRE_ONLY.parquet",
                old/"scores_and_predictor.json", old/"selected_CV_losses_TRE_ONLY.npz",
                old.parent.parent.parent/"configuration_order.json"]
    for fold in range(5):
        folder = package/"preprocessing/folds"/f"fold{fold}"
        required.extend([folder/"transformed_discovery_TRE_ONLY.npz", folder/"diet_basis.json",
                         folder/"oral_transform.json", folder/"gut_transform.json"])
    provenance = {str(path.relative_to(project)): sha(path) for path in required}
    provenance.update({"code/"+p.name: sha(p) for p in source.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".sh", ".md")})
    if args.audit_only:
        print(json.dumps(clean(dict(status="AUDIT_ONLY", source_hashes=provenance,
                                    no_new_model_fitted=True, package=str(package))), indent=2), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/"fits_TRE_ONLY").mkdir()
    models = model_specs(config)
    registration = dict(status="registered_before_new_fits", config=config, models=models,
                        source_hashes=provenance, deployment_commit=args.code_commit,
                        python=sys.version, executable=sys.executable, platform=platform.platform(),
                        numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__,
                        output=str(output), confirmation_Y_accessed=False)
    write_json(output/"registration.json", registration)
    n = config["expected_discovery_n"]
    yheld, nullheld = np.full(n, np.nan), np.full(n, np.nan)
    predictions = {m["name"]: np.full((n, len(m["grid"])), np.nan) for m in models}
    losses = {m["name"]: np.full((len(m["grid"]), 5), np.nan) for m in models}
    counts, diagnostics = [], []
    identity = None
    for fold in range(5):
        data = loader.load_fold(fold)
        index, training, valid = data["index"], data["train"], data["valid"]
        if len(index) != n or not index.final_role.eq("discovery").all():
            raise ValueError("Unexpected discovery role")
        if not np.array_equal(index.input_row.to_numpy(), np.arange(n)):
            raise ValueError("Input row identity mismatch")
        if index.groupby("split_unit_id").cv_fold.nunique().max() != 1:
            raise ValueError("Household crosses CV folds")
        if set(index.loc[training, "split_unit_id"]) & set(index.loc[valid, "split_unit_id"]):
            raise ValueError("Household leakage")
        if int(valid.sum()) != config["expected_validation_counts"][fold]:
            raise ValueError("Validation eligibility changed")
        current_identity = index[["input_row", "split_unit_id", "cv_fold"]].copy()
        if identity is None:
            identity = current_identity
        else:
            pd.testing.assert_frame_equal(identity, current_identity)
        host, encoder = host_design(index, training, data["u"], data["h"])
        k = config["background_PC_per_site"]
        base = np.column_stack([host, data["oral_PC"][:, :k], data["gut_PC"][:, :k]])
        z = np.column_stack([data["oral_z"], data["gut_z"]])
        background = Background(base, z, data["y"], training, config["main_lambda"])
        null = background.null()
        np.testing.assert_allclose(null["prediction"][valid], archived_prediction[valid, null_index], atol=2e-8, rtol=0)
        np.testing.assert_allclose(data["y"][valid], archived_y[valid], atol=1e-12, rtol=0)
        yheld[valid], nullheld[valid] = data["y"][valid], null["prediction"][valid]
        folder = package/"preprocessing/folds"/f"fold{fold}"
        diet = json.loads((folder/"diet_basis.json").read_text())["fiber_density"]
        fiber = data["u"]*diet["scale"]+diet["center"]
        support = np.quantile(fiber[training], config["contrast_training_support_quantiles"])
        contrast = config["contrast_fiber_g_per_1000_kcal"]
        supported_contrast = bool(support[0] <= contrast[0] < contrast[1] <= support[1])
        count = dict(fold=fold, train_n=int(training.sum()), valid_n=int(valid.sum()),
                     train_families=int(index.loc[training, "split_unit_id"].nunique()),
                     valid_families=int(index.loc[valid, "split_unit_id"].nunique()),
                     training_marginal_fiber_support=support, contrast_marginally_supported=supported_contrast)
        counts.append(count)
        # No new threshold/eligibility changes are made for an unsupported display contrast.
        write_json(output/f"fold{fold}_encoding.json", dict(host_encoder=encoder, diet_basis=diet, counts=count))
        for model in models:
            name = model["name"]
            raw, blocks = modifier(data, model)
            interaction = background.interaction(raw)
            warm, previous_alpha = None, None
            for j, setting in enumerate(model["grid"]):
                if model["method"] == "ridge":
                    fit = interaction.ridge(setting["penalty"])
                else:
                    if previous_alpha != setting["alpha"]:
                        warm = None
                    fit = interaction.elastic_net(**setting, initial=warm)
                    warm, previous_alpha = fit["standardized_coef"], setting["alpha"]
                prediction = fit["prediction"]
                if not np.isfinite(prediction[valid]).all():
                    raise ValueError(f"Nonfinite predictions: {name}")
                # The direct full-CLR joint model must reproduce archived v2 at lambda=10.
                if name == "ridge_CLR_oral_gut" and setting == {"penalty": 10.}:
                    np.testing.assert_allclose(prediction[valid], archived_prediction[valid, ridge_index], atol=2e-8, rtol=0)
                predictions[name][valid, j] = prediction[valid]
                losses[name][j, fold] = np.mean((data["y"][valid]-prediction[valid])**2)
                axes, centers, scales = contrast_axes(fit, blocks, training, diet, contrast)
                arrays = dict(base_coef=fit["base_coef"], main_coef=fit["main_coef"],
                              interaction_coef=fit["interaction_coef"], prediction=prediction,
                              validation=valid, training=training, u=data["u"], h=data["h"],
                              main_scales=background.z_scale, interaction_scales=interaction.scale)
                for site, axis in axes.items():
                    arrays[site+"_contrast"] = axis
                    arrays[site+"_contrast_mean"] = np.asarray(centers[site])
                    arrays[site+"_contrast_SD"] = np.asarray(scales[site])
                save_npz(output/"fits_TRE_ONLY"/f"fold{fold}_{name}_grid{j}.npz", **arrays)
                diagnostics.append(dict(fold=fold, model=name, grid_index=j, setting=setting,
                                        MSE=losses[name][j, fold], penalty=fit["penalty"],
                                        kkt_error=fit["kkt_error"], iterations=fit["iterations"],
                                        nonzero_interactions=int(np.count_nonzero(fit["interaction_coef"]))))
        print("FOLD_COMPLETE", json.dumps(clean(count)), flush=True)
    common = np.isfinite(yheld)
    if common.sum() != config["expected_CV_n"]:
        raise ValueError("Unexpected pooled CV N")
    family = identity.loc[common, "split_unit_id"].to_numpy()
    if len(np.unique(family)) != config["expected_CV_families"]:
        raise ValueError("Unexpected CV household N")
    null_metrics = metrics(yheld[common], nullheld[common])
    held = identity.copy()
    held["Y_processed"], held["matched_null_prediction"] = yheld, nullheld
    summary, calibration = [], []
    for model in models:
        name = model["name"]
        means = losses[name].mean(1)
        if model["method"] == "ridge":
            chosen = min(range(len(means)), key=lambda j: (means[j], -model["grid"][j]["penalty"]))
        else:
            chosen = min(range(len(means)), key=lambda j: (means[j], -model["grid"][j]["fraction"], model["grid"][j]["alpha"]))
        predicted = predictions[name][:, chosen]
        score = metrics(yheld[common], predicted[common])
        difference = (yheld[common]-nullheld[common])**2-(yheld[common]-predicted[common])**2
        _, _, interval = cluster_fit(difference, np.ones((common.sum(), 1)), family)
        row = dict(model=name, **score, selected_grid_index=chosen, selected_setting=model["grid"][chosen],
                   delta_R2=score["R2"]-null_metrics["R2"],
                   paired_loss_improvement=float(difference.mean()),
                   delta_R2_conditional_CI95=interval[0]/np.var(yheld[common]),
                   selection_mean_fold_MSE=float(means[chosen]),
                   development_selected=True, independently_confirmed=False)
        held[name+"_prediction"] = predicted
        # Fixed-contrast score calibration includes both sites where present.
        scores, contrasts, fold_u, fold_h = {}, {}, np.full(n, np.nan), np.full(n, np.nan)
        for fold in range(5):
            with np.load(io.BytesIO((output/"fits_TRE_ONLY"/f"fold{fold}_{name}_grid{chosen}.npz").read_bytes()), allow_pickle=False) as stored:
                valid = stored["validation"]
                fold_u[valid], fold_h[valid] = stored["u"][valid], stored["h"][valid]
                for site in ("oral", "gut"):
                    if site+"_contrast" not in stored:
                        continue
                    scores.setdefault(site, np.full(n, np.nan))
                    contrasts.setdefault(site, np.full(n, np.nan))
                    axis, sd = stored[site+"_contrast"], float(stored[site+"_contrast_SD"])
                    # Display magnitude only when the prespecified contrast has marginal support.
                    if counts[fold]["contrast_marginally_supported"]:
                        contrasts[site][valid] = axis[valid]
                    if sd > 1e-10:
                        scores[site][valid] = axis[valid]/sd
        usable_sites = [s for s in scores if np.isfinite(scores[s][common]).all()]
        for site, axis in contrasts.items():
            supported = np.isfinite(axis) & common
            row[site+"_centered_interaction_contrast_SD"] = float(axis[supported].std()) if supported.any() else None
            row[site+"_contrast_supported_n"] = int(supported.sum())
            held[name+"_"+site+"_score"] = scores[site]
            held[name+"_"+site+"_interaction_contrast"] = axis
        if usable_sites:
            design = np.column_stack([np.ones(n), fold_u, fold_h,
                                      *[scores[s] for s in usable_sites],
                                      *[fold_u*scores[s] for s in usable_sites],
                                      np.eye(5)[identity.cv_fold.to_numpy(int)][:, 1:]])
            try:
                beta, _, ci = cluster_fit((yheld-nullheld)[common], design[common], family)
                for j, site in enumerate(usable_sites):
                    position = 3+len(usable_sites)+j
                    calibration.append(dict(model=name, site=site, beta=float(beta[position]), CI95=ci[position],
                                            exploratory_conditional_only=True,
                                            interpretation="linear fiber association calibration against fixed-contrast score; not full nonlinear validation"))
            except ValueError as error:
                calibration.append(dict(model=name, status="not_estimable", reason=str(error)))
        else:
            calibration.append(dict(model=name, status="no_nonzero_score_on_all_folds"))
        summary.append(row)
        print("MODEL_RESULT", json.dumps(clean(row)), flush=True)
    # Individual rows, weights, scores and predictions never leave TRE.
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(held, preserve_index=False),
                   output/"held_predictions_and_contrasts_TRE_ONLY.parquet", compression="zstd")
    pd.testing.assert_frame_equal(held.reset_index(drop=True), pq.ParquetFile(output/"held_predictions_and_contrasts_TRE_ONLY.parquet", pre_buffer=False).read().to_pandas(ignore_metadata=True), check_dtype=False)
    result = dict(status="completed_and_readback_verified", null=null_metrics, models=summary,
                  counts=counts, calibration=calibration, diagnostics=diagnostics,
                  n_people=int(common.sum()), n_families=len(np.unique(family)),
                  seconds=time.monotonic()-start, confirmation_Y_accessed=False,
                  limitations=["CV used for tuning: development estimates",
                               "Conditional intervals omit selection and shared-training uncertainty",
                               "Nonlinear score calibration is only a linear projection diagnostic",
                               "Marginal contrast support does not establish joint positivity",
                               "No independent confirmation or causal individual response labels"])
    write_json(output/"summary.json", result)
    pd.DataFrame(clean(summary)).to_csv(output/"model_comparison.csv", index=False)
    write_json(output/"artifact_manifest_TRE_ONLY.json", {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()})
    print("PILOT_COMPLETE", json.dumps(clean({k: result[k] for k in ["status", "n_people", "n_families", "seconds", "confirmation_Y_accessed"]})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())
