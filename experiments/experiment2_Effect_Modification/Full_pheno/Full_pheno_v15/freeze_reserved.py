"""Derive V15-equivalent reserved inputs and freeze predictions before outcome access."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from run_repeat import ENDPOINTS, HERE, RECIPES, REFERENCE, context, emit, parse, read_json, register, require, v1_run, write_json


V5 = HERE.parent.parent / "v5"
FOOD_SEED = 202621071
OUTCOME_PATHS = ("nightingale", "nightingale_tags", "serum_lipidomics")


def geometry(project, study, reserved, calibration):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix, load_npz, vstack
    from reserved_inputs import KEY, project_population

    source = project / "data_qc/preprocessing_20261030_v1/foodid_interval_geometry_v1b"
    rows = source / "matrix_person_rows_TRE_ONLY.parquet"
    columns = source / "matrix_foodid_columns_TRE_ONLY.parquet"
    weights = source / "unknown_foodid_equalday_weights_TRE_ONLY.npz"
    receipt = source / "verification_summary.json"
    sources = {p: v1_run.file_sha(p) for p in (rows, columns, weights, receipt)}
    require(not read_json(receipt)["new_real_Y_reads"], "V15_GEOMETRY_Y_SCOPE")
    order = pq.ParquetFile(columns).read(columns=["food_id"], use_threads=False)["food_id"].to_pylist()
    require(order == calibration.unknown.food_id.tolist(), "V15_GEOMETRY_COLUMN_ORDER")
    full = load_npz(weights).tocsr()
    require(full.shape[1] == len(order) and np.isfinite(full.data).all()
            and (full.data >= 0.).all(), "V15_GEOMETRY_WEIGHTS")
    col = ["matrix_row_ordinal", "UPF_lower_equalday_fraction", "UPF_upper_equalday_fraction"]

    def align(index):
        linked = index[KEY].merge(project_population(rows, KEY + col, index, unique_by=KEY),
                                  on=KEY, how="left", validate="one_to_one")
        ok = linked.matrix_row_ordinal.notna().to_numpy()
        slots = linked.loc[ok, "matrix_row_ordinal"].to_numpy(int)
        require(np.array_equal(slots, linked.loc[ok, "matrix_row_ordinal"].to_numpy())
                and len(slots) == len(set(slots)) and np.all((0 <= slots) & (slots < full.shape[0])),
                "V15_GEOMETRY_ROW_ORDINAL")
        source_rows = full[slots].tocoo()
        aligned = csr_matrix((source_rows.data, (np.flatnonzero(ok)[source_rows.row], source_rows.col)),
                             shape=(len(index), full.shape[1]))
        return aligned, *(linked[k].to_numpy(float) for k in col[1:])

    existing = pd.DataFrame(dict(zip(KEY, study.keys, strict=True)))
    mat, lower, upper = align(existing)
    diff = mat - calibration.weights
    require((not diff.nnz or abs(diff.data).max() <= 1e-12)
            and np.allclose(lower, calibration.lower, rtol=0., atol=1e-12, equal_nan=True)
            and np.allclose(upper, calibration.upper, rtol=0., atol=1e-12, equal_nan=True),
            "V15_GEOMETRY_DISCOVERY_REPLAY")
    held, lo, hi = align(reserved)
    days = reserved.n_energy_days.to_numpy(float)
    ok = days >= 3.
    require(np.isfinite(lo[ok]).all() and np.isfinite(hi[ok]).all()
            and np.allclose(np.asarray(held.sum(axis=1)).ravel()[ok], (hi-lo)[ok], rtol=0., atol=1e-8),
            "V15_GEOMETRY_RESERVED_REPLAY")
    return replace(calibration, weights=vstack([mat, held], format="csr"),
                   lower=np.r_[lower, lo], upper=np.r_[upper, hi], qualified=np.r_[calibration.qualified, ok]), sources


def combine(study, raw_sites, reserved, calibration):
    import numpy as np
    import protocol as p
    from source import Background, raw_ecology, species_indices
    from v3_inputs import Site, fit_site

    index = reserved.index
    keys = tuple(np.r_[x, index[k].to_numpy(dtype=x.dtype)] for k, x in zip(p.KEY, study.keys, strict=True))
    held_smoke = index.smoking_current_status_label.to_numpy()
    smoke = [str(v) if v is not None and str(v).strip() else "__UNKNOWN__" for v in held_smoke]
    family = np.r_[study.family, index.split_unit_id.to_numpy()]
    training = np.arange(len(family)) < study.n
    numeric = {k: np.r_[x, index[k].to_numpy(float)] for k, x in study.numeric.items()}
    ecology, pcs, supported, score = [], [], np.ones(len(family), dtype=bool), None
    for site, data in raw_sites.items():
        ab = np.vstack([data.abundance, reserved.raw[site]])
        valid = np.isfinite(ab).all(1)
        z = np.full((len(ab), 2), np.nan)
        z[valid] = raw_ecology(ab[valid])
        ecology.append(z)
        pc = np.zeros((len(ab), p.N_PC))
        pc[valid], present = fit_site(site, Site(ab[valid], data.taxa), keys[0][valid], training[valid])
        supported[valid] &= present
        supported &= valid
        pcs.append(pc)
        if site == "gut":
            score = np.log1p(ab[:, list(species_indices(data.taxa).values())].sum(axis=1) / p.PANEL_UNIT)
    ecology = np.column_stack(ecology)
    markers = np.column_stack((score, ecology[:, 2]))
    require(np.allclose(ecology[:study.n], study.ecology, rtol=0., atol=1e-12)
            and np.allclose(markers[:study.n], study.markers, rtol=0., atol=1e-12),
            "V15_MICROBIOME_DISCOVERY_REPLAY")
    values, parameters, _ = calibration.refit(training, FOOD_SEED)
    archived = {"UPF": values}
    fit = Background(training, supported, np.column_stack(pcs), archived, parameters)
    return SimpleNamespace(keys=keys, n=len(family), family=family, numeric=numeric, ecology=ecology,
                           markers=markers, smoking=np.r_[study.smoking, smoke], backgrounds={0: fit})


def diets_for_both(root, study, both, day):
    import numpy as np
    from batch_sources import read_diets
    from food_features import SOURCES

    discovered, _, files = read_diets(root, study)
    saved = both.backgrounds[0].archived["UPF"]
    placeholder = np.vstack([study.backgrounds[0].archived["UPF"], saved[study.n:]])
    both.backgrounds[0].archived["UPF"] = placeholder
    try:
        with patch.dict(SOURCES, day=str(day)):
            extended, _, other = read_diets(root, both)
    finally:
        both.backgrounds[0].archived["UPF"] = saved
    require(files["day"] != other["day"] and all(files[k] == other[k] for k in files if k != "day"),
            "V15_DIET_SOURCE_ROUTE")
    named = {name: (kind, x) for name, kind, x in extended}
    catalog = read_json(REFERENCE / "inventory.json")["diet_ids"]
    require([k for k, _, _ in discovered] == catalog, "V15_DISCOVERY_DIET_CATALOG")
    result = []
    for name, kind, values in discovered:
        require(name in named and named[name][0] == kind
                and np.allclose(named[name][1][:study.n], values, rtol=0., atol=1e-12, equal_nan=True),
                "V15_DIET_DISCOVERY_REPLAY " + name)
        result.append((name, kind, named[name][1]))
    return result, files


def select_values(paths, index, endpoints):
    """Read only the frozen assays on an explicitly named baseline roster."""
    import numpy as np
    from batch_sources import features_by_visit
    from reserved_inputs import KEY, project_population

    result = {}
    for type_, source in (("nightingale", paths["nightingale"]), ("serum_lipidomics", paths["serum_lipidomics"])):
        fields = [e.split(":", 1)[1] for e in endpoints if e.startswith(type_ + ":")]
        keys = [*KEY, "array_index"] if type_ == "nightingale" else KEY
        values = project_population(source, keys + fields, index).merge(
            index[KEY + ["input_row"]], on=KEY, validate="many_to_one")
        if type_ == "nightingale":
            tag_cols = [k + "_tag" for k in fields]
            tags = project_population(paths["nightingale_tags"], keys + tag_cols, index, unique_by=keys)
            values = values.merge(tags, on=keys, how="inner", validate="one_to_one")
            values[fields] = values[fields].mask(values[tag_cols].notna().to_numpy())
        panel, _ = features_by_visit(values, fields, len(index))
        result.update({type_ + ":" + k: panel[k].to_numpy(float) for k in fields})
    return np.column_stack([result[k] for k in endpoints])


def main():
    args = parse(__doc__)
    v1_run.init_runtime(1)
    sys.path.insert(1, str(V5))
    import numpy as np
    import pandas as pd
    from batch_fit import context_design
    from batch_sources import paths
    from category_fit import panels
    from main_fit import predict
    from reserved_inputs import KEY, load_reserved
    import protocol as p

    out = HERE / "runs/TRE_ONLY" / args.run_id
    repeat = HERE / "runs/TRE_ONLY/repeat-1/registration.json"
    prior = read_json(repeat)
    require(prior["endpoints"] == list(ENDPOINTS), "V15_RESERVED_ENDPOINTS_CHANGED")
    with v1_run.run_lock(out):
        require(not (out / "FROZEN.json").exists(), "V15_ALREADY_FROZEN")
        root, snapshot, study, calibration, sites, cal_files = context()
        pg = root.parent
        index = pd.DataFrame(dict(zip(KEY, study.keys, strict=True)))
        index["split_unit_id"] = study.family
        index["final_role"] = "discovery"
        index["input_row"] = np.arange(study.n)
        source = SimpleNamespace(index=index, taxa={k: v.taxa for k, v in sites.items()})
        reserved = load_reserved(V5, source)
        full, extra = geometry(pg, study, reserved.index, calibration)
        both = combine(study, sites, reserved, full)
        day = pg / "data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet"
        diets, inputs = diets_for_both(root, study, both, day)
        catalog_paths = paths(root)
        outcome_paths = {k: catalog_paths[k] for k in OUTCOME_PATHS}
        official = read_json(REFERENCE / "registration.json")["sources"]
        registered = {f: official[str(f)] for f in (*inputs.values(), *cal_files, *outcome_paths.values())}
        require(all(v1_run.file_sha(f) == d for f, d in registered.items()), "V15_REGISTERED_SOURCE_CHANGED")
        require(all(v1_run.file_sha(f) == d for f, d in extra.items()), "V15_GEOMETRY_INPUT_CHANGED")
        support, plain = context_design(both, 0, "UPF")
        panel, _ = panels(diets, both, 0, support)
        held = np.flatnonzero(support[study.n:]) + study.n
        require(len(np.unique(both.family[held])) >= 40, "V15_RESERVED_NO_COMMON_SUPPORT")
        library = HERE.parent / "Full_pheno_v14/main_fit.py"
        require(prior["code"]["main_fit.py"] == v1_run.file_sha(library), "V15_RESERVED_MODEL_CHANGED")
        code_paths = [Path(__file__), HERE / "RESERVED_PROTOCOL.md", library, repeat]
        register(out, dict(commit=args.commit, endpoints=list(ENDPOINTS), food_seed=FOOD_SEED,
                           code={str(f): v1_run.file_sha(f) for f in code_paths},
                           outcome_source={k: [str(f), registered[f]] for k, f in outcome_paths.items()},
                           predictors={str(f): digest for f, digest in {**reserved.hashes, **extra}.items()}))
        y = select_values(outcome_paths, index, ENDPOINTS)
        lower, upper = np.nanquantile(y, [.01, .99], axis=0)
        require(np.isfinite(lower).all() and np.isfinite(upper).all(), "V15_NO_DISCOVERY_ASSAY_SUPPORT")
        full_diet = panel["whole_diet"]
        data = {}
        for j, e in enumerate(ENDPOINTS):
            tr = np.flatnonzero(support[:study.n] & np.isfinite(y[:, j]))
            require(len(np.unique(both.family[tr])) >= 120, "V15_RESERVED_SMALL_TRAIN")
            sel = np.r_[tr, held]
            cooked = np.clip(y[tr, j:j+1], lower[j], upper[j])
            forecasts = predict(plain[sel], both.backgrounds[0].pcs[sel, p.N_PC:p.N_PC+2],
                                full_diet[sel], full_diet[sel, :1], cooked, both.numeric["n_energy_days"][sel],
                                len(tr), RECIPES)
            require(all(np.isfinite(arr).all() for bundle in forecasts.values() for arr in bundle),
                    "V15_NONFINITE_RESERVED_PREDICTIONS")
            data[f"assay{j}"] = np.column_stack([forecasts["reference:off"][0]]
                                               + [forecasts[f"{m}:off"][1] for m in RECIPES]
                                               + [forecasts[f"{m}:{a}"][2] for m in RECIPES for a in ("fixed", "half")])
        buffer = BytesIO()
        np.savez_compressed(buffer, **data, participant_id=both.keys[0][held],
                            households=np.asarray(both.family[held], dtype=str), lower=lower, upper=upper)
        unused = {p.outcome_file(e) for e in p.ENDPOINTS}
        require(set(snapshot.paths) - unused <= snapshot.verified, "V15_RESERVED_DISCOVERY_SNAPSHOT")
        snapshot.verify_manifest()
        pred = out / "predictions.npz"
        v1_run.atomic_write(pred, buffer.getvalue())
        frozen = dict(predictions_sha=v1_run.file_sha(pred), registration_sha=v1_run.file_sha(out / "registration.json"),
                      heads=["A", "reference:B", "ridge_100:B", "reference:fixed", "reference:half",
                             "ridge_100:fixed", "ridge_100:half"], participants=len(held),
                      households=len(np.unique(both.family[held])), diet_features=len(diets),
                      confirmation_outcomes_read=False)
        write_json(out / "FROZEN.json", frozen)
    emit("V15_RESERVED_FROZEN", frozen)


if __name__ == "__main__":
    main()
