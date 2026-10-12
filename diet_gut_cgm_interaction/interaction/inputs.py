"""Explicit relative-path inputs; never import or execute the legacy pipelines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import header, need, norm_ids, numeric, read_csv, require, sha256, text_series

BASE = ["age_years", "sex", "education_level", "smoking_status", "sleep_duration_hours_day",
        "physical_activity_met_h_week", "vitamin_use", "hormone_use", "bmi", "cgm_device_type"]
CATEGORICAL = {"sex", "education_level", "smoking_status", "cgm_device_type"}
ALIASES = {
    "AHEI": ["AHEI_z", "ahei_z", "ahei_score_z", "ahei_energy_adjusted_z", "ahei_score_energy_adjusted_z", "mahei7_score_energy_adjusted_z", "mahei7_z"],
    "AMED": ["AMED_z", "amed_z", "amed_score_z", "amed_energy_adjusted_z", "amed_score_energy_adjusted_z"],
    "rEDIH": ["rEDIH_z", "redih_z", "redih_score_z", "redih_energy_adjusted_z", "redih_score_energy_adjusted_z", "redih_energy_adjusted_score_z"]}
NEW_COLUMNS = {"EAT13": "modified_eat_lancet13_z", "NOVA4": "nova4_total_energy_pct_cov80_z",
               "Carbohydrate_pct": "carbohydrate_energy_pct_z"}


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    exposures = c["focus_exposures"] + c["extension_exposures"]
    require(len(exposures) == len(set(exposures)) and set(exposures).issubset(set(ALIASES) | set(NEW_COLUMNS) | {"hPDI"}), "Invalid/duplicate exposures")
    for rows, key in [(c["taxa"], "id"), (c["outcomes"], "id")]:
        require(rows and len(rows) == len(set(x[key] for x in rows)), "Empty/duplicate registry")
    require(len(set(x["species"] for x in c["taxa"])) == len(c["taxa"]), "Duplicate species")
    require(len(set(x["field"] for x in c["outcomes"])) == len(c["outcomes"]), "Duplicate outcomes")
    require(c["focus_exposures"] and c["protocol_mode"] in ["required", "if_available", "disabled"], "Invalid configuration")
    require(0 < c["alpha"] < 1, "Invalid alpha")
    for key in ["minimum_model_n", "minimum_residual_df", "minimum_tail_cell_n"]:
        require(type(c[key]) is int and c[key] > 0, "Invalid " + key)
    return c


def resolve_paths(config, root):
    root = Path(root).expanduser().resolve()
    paths = {}
    for label, value in config["inputs"].items():
        path = Path(value).expanduser()
        # No guessed 'latest' run and no automatic fallback to another data release.
        paths[label] = path.resolve() if path.is_absolute() else (root / path).resolve()
    return paths


def path_inventory(config, paths):
    rows = []
    for label, path in paths.items():
        required = label != "protocol" or config["protocol_mode"] == "required"
        enabled = label != "protocol" or config["protocol_mode"] != "disabled"
        rows.append(dict(input=label, path=str(path), required=required, enabled=enabled, exists=path.is_file()))
    return rows


def covariates(exposure, protocol=False):
    result = list(BASE)
    if exposure == "hPDI":
        result.append("alcohol_intake_g_day")
    if exposure in NEW_COLUMNS:
        result.append("mean_daily_energy_kcal")
    if protocol:
        result += ["antibiotic_use", "ppi_use"]
    return result


def select_column(columns, choices, label):
    for choice in choices:
        hits = [x for x in columns if x.lower() == choice.lower()]
        if hits:
            require(len(hits) == 1, "Ambiguous " + label)
            return hits[0]
    raise ValueError("No recognized column for " + label + ": " + str(choices))


def resolve_taxa(columns, taxa):
    result = []
    for taxon in taxa:
        token = taxon["species"]
        hits = [col for col in columns if col == token or "s__" + token in col.split("|")]
        require(len(hits) == 1, "{}: expected one exact species column; found {}".format(token, len(hits)))
        result.append(dict(taxon=taxon["id"], species=token, column=hits[0], scale="original CLR-Z"))
    require(len(set(x["column"] for x in result)) == len(result), "Duplicate taxon mapping")
    return result


def scoped(frame, label, config, audit):
    need(frame, ["participant_id"], label)
    original_n = len(frame)
    if label in ["microbiome", "cgm"]:
        need(frame, ["cohort", "research_stage"], label)
    for field in ["cohort", "research_stage"]:
        if field in frame:
            frame = frame.loc[frame[field].astype(str).str.strip().eq(config[field])].copy()
    require(not frame.empty, label + ": no rows in specified cohort/visit")
    frame = frame.copy()
    frame["participant_id"] = norm_ids(frame.participant_id)
    require(not frame.participant_id.duplicated().any(), label + ": duplicate participant IDs after scope/normalization")
    audit.append(dict(input=label, input_rows=original_n, scope_rows=len(frame), scope_excluded=original_n - len(frame)))
    return frame.set_index("participant_id")


def load_inputs(config, paths):
    inventory = path_inventory(config, paths)
    absent = [r for r in inventory if r["enabled"] and r["required"] and not r["exists"]]
    require(not absent, "Required inputs missing:\n" + "\n".join(r["input"] + ": " + r["path"] for r in absent))
    files, audit, frames, mappings = [], [], {}, []
    taxa = resolve_taxa(header(paths["microbiome"]), config["taxa"])
    for item in inventory:
        if not item["enabled"] or not item["exists"]:
            continue
        label, path = item["input"], paths[item["input"]]
        fingerprint = dict(label=label, path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
        columns = None
        if label == "microbiome":
            wanted = {"participant_id", "cohort", "research_stage"} | {t["column"] for t in taxa}
            columns = lambda col: col in wanted
        frame = read_csv(path, columns)
        frames[label] = scoped(frame, label, config, audit)
        require(fingerprint["sha256"] == sha256(path), "Input changed while reading: " + label)
        files.append(fingerprint)

    old, new, hp, candidate = [frames[x] for x in ["current4", "new3", "corrected_hpdi", "candidate_master"]]
    scores = {}
    for exposure, choices in ALIASES.items():
        col = select_column(old.columns, choices, exposure)
        scores[exposure] = numeric(old[col], exposure)
        mappings.append(dict(exposure=exposure, source="current4", column=col))
    col = "hpdi_score_energy_adjusted_z"
    need(hp, [col], "corrected_hpdi")
    scores["hPDI"] = numeric(hp[col], col)
    mappings.append(dict(exposure="hPDI", source="corrected_hpdi", column=col))
    need(new, [e + "_z" for e in NEW_COLUMNS] + ["mean_daily_energy_kcal"], "new3")
    need(candidate, list(NEW_COLUMNS.values()) + ["mean_daily_energy_kcal"], "candidate_master")
    require(new.index.isin(candidate.index).all(), "new3 contains participants absent from candidate master")
    for e, original in NEW_COLUMNS.items():
        a, b = numeric(new[e + "_z"], e), numeric(candidate[original], original).reindex(new.index)
        require(np.allclose(a, b, rtol=1e-7, atol=1e-8, equal_nan=True), "new3/candidate mismatch: " + e)
        scores[e] = a
        mappings.append(dict(exposure=e, source="new3", column=e + "_z", verified_candidate_column=original))
    energy = numeric(new.mean_daily_energy_kcal, "energy")
    require(np.allclose(energy, numeric(candidate.mean_daily_energy_kcal, "candidate_energy").reindex(new.index), rtol=1e-7, atol=1e-8, equal_nan=True), "Energy/candidate mismatch")
    score_table = pd.DataFrame(scores).join(energy.rename("mean_daily_energy_kcal"))

    cov = frames["covariates"].copy()
    alcohol = select_column(cov.columns, ["alcohol_intake_g_day", "alcohol_g_day", "alcohol_intake"], "alcohol")
    if alcohol != "alcohol_intake_g_day":
        cov = cov.rename(columns={alcohol: "alcohol_intake_g_day"})
    need(cov, BASE + ["exclude_diabetes_or_a10", "alcohol_intake_g_day"], "covariates")
    for col in BASE + ["exclude_diabetes_or_a10", "alcohol_intake_g_day", "known_diabetes", "a10_medication_use"]:
        if col not in cov:
            continue
        cov[col] = text_series(cov[col]) if col in CATEGORICAL else numeric(cov[col], col)
    for col in ["vitamin_use", "hormone_use", "exclude_diabetes_or_a10", "known_diabetes", "a10_medication_use"]:
        if col in cov:
            require(cov[col].dropna().isin([0, 1]).all(), "Nonbinary " + col)
    for col in ["known_diabetes", "a10_medication_use"]:
        if col in cov:
            require(not (cov[col].eq(1) & ~cov.exclude_diabetes_or_a10.eq(1)).any(), "Exclusion flag contradicts " + col)
    if {"known_diabetes", "a10_medication_use"}.issubset(cov.columns):
        # Match the covariate builder's three-state union exactly. In particular,
        # one negative plus one unknown must never enter the strict sample.
        expected = pd.Series(np.nan, index=cov.index, dtype=float)
        expected.loc[cov.known_diabetes.eq(1) | cov.a10_medication_use.eq(1)] = 1.0
        expected.loc[cov.known_diabetes.eq(0) & cov.a10_medication_use.eq(0)] = 0.0
        observed = cov.exclude_diabetes_or_a10
        require((observed.eq(expected) | (observed.isna() & expected.isna())).all(),
                "Exclusion flag contradicts the diabetes/A10 three-state union")

    cgm = frames["cgm"]
    need(cgm, ["connection_id"], "cgm")
    require(text_series(cgm.connection_id).notna().all(), "Missing CGM connection_id")
    scales, cgm_columns = {}, {}
    for outcome in config["outcomes"]:
        field = outcome["field"]
        need(cgm, [field + "_clean", field + "_z"], "cgm")
        clean, z = [numeric(cgm[field + suffix], field + suffix) for suffix in ["_clean", "_z"]]
        require(clean.notna().equals(z.notna()), "CGM clean/Z availability mismatch: " + field)
        sd = clean.std(ddof=1)
        require(np.isfinite(sd) and sd > 0, "Constant/empty CGM " + field)
        require(np.allclose((clean - clean.mean()) / sd, z, rtol=1e-6, atol=1e-7, equal_nan=True), "CGM clean/Z mismatch: " + field)
        require(clean.dropna().ge(0).all(), "Negative CGM clean values: " + field)
        scales[outcome["id"]] = dict(sd=float(sd), mean=float(clean.mean()), unit=outcome["unit"], field=field, n=int(clean.notna().sum()))
        cgm_columns[outcome["id"]] = z
    if "cgm_device_type" in cgm:
        ids = cgm.index.intersection(cov.index)
        a, b = [text_series(s).str.lower().str.replace(r"\.0$", "", regex=True) for s in [cgm.loc[ids, "cgm_device_type"], cov.loc[ids, "cgm_device_type"]]]
        require(not (a.notna() & b.notna() & a.ne(b)).any(), "CGM/covariate device conflict")

    micro = pd.DataFrame({t["taxon"]: numeric(frames["microbiome"][t["column"]], t["taxon"]) for t in taxa})
    master = pd.DataFrame(cgm_columns).join(cov[BASE + ["alcohol_intake_g_day", "exclude_diabetes_or_a10"]], how="left")
    master["has_covariate_record"] = master.index.isin(cov.index)
    master["has_microbiome_record"] = master.index.isin(micro.index)
    flag = master.exclude_diabetes_or_a10
    master["primary_eligible"] = master.has_covariate_record & ~flag.eq(1)
    master["strict_eligible"] = master.has_covariate_record & flag.eq(0)
    master = master.join(score_table, how="left").join(micro, how="left")
    protocol_available = "protocol" in frames
    for source, target in [("antibiotic_use_protocol_window", "antibiotic_use"), ("ppi_use_protocol_window", "ppi_use")]:
        master[target] = np.nan
        if protocol_available:
            need(frames["protocol"], [source], "protocol")
            s = numeric(frames["protocol"][source], source)
            require(s.dropna().isin([0, 1]).all(), "Nonbinary protocol medication")
            master[target] = s.reindex(master.index)
    summary = dict(cgm_n=len(master), primary_eligible_n=int(master.primary_eligible.sum()),
                   strict_eligible_n=int(master.strict_eligible.sum()), missing_covariate_record_n=int((~master.has_covariate_record).sum()),
                   unknown_exclusion_n=int((master.has_covariate_record & flag.isna()).sum()),
                   microbiome_overlap_n=int(master.has_microbiome_record.sum()), protocol_available=protocol_available)
    return master, dict(files=files, scope_audit=audit, score_mapping=mappings, taxa_mapping=taxa,
                        outcome_scales=scales, summary=summary, inventory=inventory)
