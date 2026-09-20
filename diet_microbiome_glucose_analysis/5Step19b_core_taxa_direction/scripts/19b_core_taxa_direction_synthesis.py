#!/usr/bin/env python3
"""Read-only synthesis of frozen HPP paths. Compatible with Python 3.8+."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
KEYS = ["exposure", "outcome_field", "species"]
TAGS = ["primary", "strict", "protocol"]
DOMAINS = ["glycemic_level", "glycemic_variability", "glycemic_risk_range"]
EFFECTS = {"a": "a_diet_to_microbiome", "b": "b_microbiome_to_cgm", "indirect": "indirect_effect"}
SIG = ["mediation_FDR05_primary", "mediation_FDR05_primary_primary", "significant_mediation_FDR05"]
CON = ["bridge_direction_consistent_mediator", "candidate_mediator_consistent", "canonical_direction_consistent"]
SOURCE_FILES = {
    "18f_all": "paper20_mediation_threeway_robustness/reports/18f_threeway_path_robustness.csv",
    "18f_high": "paper20_mediation_threeway_robustness/reports/18f_highest_robustness_paths.csv",
    "18g": "paper20_mediation_phenotype_domains/reports/18g_threeway_paths_with_domains.csv",
    "19a": "paper20_final_biological_architecture/reports/19a_highest_robust_paths_annotated.csv",
    "19a_species": "paper20_final_biological_architecture/reports/19a_species_architecture_summary.csv",
    "19a_pan": "paper20_final_biological_architecture/reports/19a_pan_domain_core_taxa.csv",
    "19a_cross": "paper20_final_biological_architecture/reports/19a_cross_domain2_taxa.csv",
}
EXPECTED = {"candidates": 1490, "high": 871, "species": 109, "pan": 4, "cross": 17, "single": 88,
            "primary": 1003, "retained_strict": 887, "retained_protocol": 940,
            "glycemic_level": 202, "glycemic_variability": 598, "glycemic_risk_range": 71}


class AuditError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def need(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise AuditError("{}: missing columns {}".format(label, missing))


def booleans(series):
    # Missing/unknown is an error, never silently False.
    mapping = {"true": True, "1": True, "1.0": True, "yes": True, "y": True, "t": True,
               "false": False, "0": False, "0.0": False, "no": False, "n": False, "f": False}
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise AuditError("Invalid boolean {}: {}".format(series.name, series[result.isna()].head(3).tolist()))
    return result.astype(bool)


def unique(df, keys, label):
    need(df, keys, label)
    if df[keys].isna().any().any() or any(df[k].astype(str).str.strip().eq("").any() for k in keys):
        raise AuditError(label + ": empty identity")
    if df.duplicated(keys).any():
        raise AuditError(label + ": duplicate identities " + str(df.loc[df.duplicated(keys, keep=False), keys].head(3).to_dict("records")))


def signs(values, eps=1e-12):
    x = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select([~np.isfinite(x), x.abs() <= eps, x > eps, x < -eps],
                               ["unknown", "near_zero", "positive", "negative"], default="unknown"), index=values.index)


def sign_class(values):
    vals = set(values)
    if "positive" in vals and "negative" in vals:
        return "mixed" if vals <= {"positive", "negative"} else "mixed_with_uncertainty"
    if vals == {"positive"}:
        return "all_positive"
    if vals == {"negative"}:
        return "all_negative"
    return "uncertain_or_near_zero"


def stability(frame):
    known = frame.isin(["positive", "negative"]).all(axis=1)
    return pd.Series(np.where(~known, "uncertain", np.where(frame.nunique(axis=1).eq(1), "stable", "sign_flip")), index=frame.index)


def selected_column(df, options, label):
    # Same explicit priority as 18f; record actual choice in manifest.
    for col in options:
        if col in df.columns:
            return col
    raise AuditError(label + ": no supported column; expected one of " + str(options))


class Audit:
    def __init__(self, out, args):
        self.out = out
        self.args = args
        self.files = {}
        self.events = []
        self.candidates = []
        self.differences = []
        self.mappings = {}
        self.selected = {}
        self.notes = []

    def record_file(self, path, label):
        path = Path(path).resolve()
        if not path.is_file():
            raise AuditError("Missing {}: {}".format(label, path))
        if str(path) not in self.files:
            self.files[str(path)] = {"path": str(path), "label": label, "sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    def read(self, path, label):
        path = self.record_file(path, label)
        with path.open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if len(header) != len(set(header)):
            raise AuditError(label + ": duplicate column names")
        df = pd.read_csv(path, low_memory=False, dtype={k: str for k in KEYS})
        self.files[str(path)]["rows"] = len(df)
        self.files[str(path)]["columns"] = df.columns.tolist()
        return df

    def event(self, check, observed, expected=None):
        ok = expected is None or observed == expected
        self.events.append({"check": check, "observed": str(observed), "expected": str(expected), "status": "PASS" if ok else "FAIL"})
        if not ok:
            raise AuditError("{}: observed={}, expected={}".format(check, observed, expected))

    def align(self, reference, other, label, keys=KEYS):
        unique(reference, keys, label + " reference")
        unique(other, keys, label)
        a = reference.set_index(keys)
        b = other.set_index(keys)
        left = a.index.difference(b.index)
        right = b.index.difference(a.index)
        if len(left) or len(right):
            for direction, items in [("missing", left), ("extra", right)]:
                for item in items:
                    val = item if isinstance(item, tuple) else (item,)
                    self.differences.append(dict(zip(keys, val), check=label, difference=direction))
            raise AuditError("{}: {} missing / {} extra identities".format(label, len(left), len(right)))
        return a.reset_index(), b.reindex(a.index).reset_index()

    def compare(self, ref, other, pairs, label):
        a, b = self.align(ref, other, label)
        for left, right, kind in pairs:
            need(a, [left], label)
            need(b, [right], label)
            if kind == "bool":
                ok = booleans(a[left]).eq(booleans(b[right])).to_numpy()
            elif kind == "number":
                x = pd.to_numeric(a[left], errors="raise").to_numpy(float)
                y = pd.to_numeric(b[right], errors="raise").to_numpy(float)
                ok = np.isclose(x, y, atol=self.args.atol, rtol=self.args.rtol, equal_nan=True)
            else:
                ok = a[left].fillna("<MISSING>").astype(str).eq(b[right].fillna("<MISSING>").astype(str)).to_numpy()
            if not ok.all():
                for i in np.flatnonzero(~ok):
                    row = a.loc[i, KEYS].to_dict()
                    row.update(check=label, field=left, expected=str(a.at[i, left]), observed=str(b.at[i, right]))
                    self.differences.append(row)
                raise AuditError("{}: {} differs on {} paths".format(label, left, int((~ok).sum())))

    def save(self, status, error=None):
        pd.DataFrame(self.events, columns=["check", "observed", "expected", "status"]).to_csv(self.out / "19b_input_audit.csv", index=False)
        pd.DataFrame(self.candidates, columns=["kind", "path", "sha256", "matches_18f", "reason"]).to_csv(self.out / "19b_model_candidates.csv", index=False)
        pd.DataFrame(self.differences).to_csv(self.out / "19b_input_differences.csv", index=False)
        pd.DataFrame(self.files.values()).to_csv(self.out / "19b_input_files.csv", index=False)
        code_files = [p for folder in ["scripts", "config"] for p in (PACKAGE / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)]
        manifest = {"status": status, "error": error, "utc": datetime.now(timezone.utc).isoformat(),
                    "data_kind": "server_result_tables" if not getattr(self.args, "synthetic", False) else "SYNTHETIC_TEST_ONLY",
                    "parameters": vars(self.args), "python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "matplotlib": getattr(sys.modules.get("matplotlib"), "__version__", None),
                    "inputs": list(self.files.values()),
                    "selected_models": self.selected, "field_mapping": self.mappings, "notes": self.notes,
                    "code_and_config": [{"file": str(p.relative_to(PACKAGE)), "sha256": sha256(p)} for p in code_files],
                    "models_refit": False, "fdr_recomputed": False,
                    "all_inputs_unchanged_verified": any(e["check"] == "all_inputs_unchanged" and e["status"] == "PASS" for e in self.events),
                    "outputs": [{"file": p.name, "sha256": sha256(p)} for p in sorted(self.out.iterdir()) if p.is_file() and p.name != "manifest.json"]}
        (self.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def snapshot_pairs():
    pairs = [("highest_robustness", "highest_robustness", "bool")]
    for tag in TAGS:
        pairs += [(base + "_" + tag, base + "_" + tag, kind) for base, kind in
                  [("significant", "bool"), ("consistent", "bool"), ("robust", "bool"),
                   ("indirect_effect", "number"), ("N", "number"), ("FDR", "number"), ("bootstrap_p", "number")]]
    return pairs


def load_frozen(audit, paths, expected):
    tables = {name: audit.read(path, name) for name, path in paths.items()}
    all_paths = tables["18f_all"]
    unique(all_paths, KEYS, "18f_all")
    need(all_paths, [p[0] for p in snapshot_pairs()] + ["primary_retained_strict", "primary_retained_protocol"], "18f_all")
    audit.event("frozen_candidates", len(all_paths), expected["candidates"])
    highest = booleans(all_paths["highest_robustness"])
    high = all_paths.loc[highest].copy()
    # Verify existing labels without replacing them or computing new significance.
    robust = {}
    for tag in TAGS:
        robust[tag] = booleans(all_paths["robust_" + tag])
        conjunction = booleans(all_paths["significant_" + tag]) & booleans(all_paths["consistent_" + tag])
        audit.event(tag + "_inherited_robust_definition", bool(robust[tag].equals(conjunction)), True)
    audit.event("inherited_highest_definition", bool(highest.equals(robust["primary"] & robust["strict"] & robust["protocol"])), True)
    audit.event("primary_robust", int(robust["primary"].sum()), expected["primary"])
    for tag in ["strict", "protocol"]:
        retained = robust["primary"] & robust[tag]
        audit.event("retained_" + tag, int(retained.sum()), expected["retained_" + tag])
        audit.event("retained_label_" + tag, bool(booleans(all_paths["primary_retained_" + tag]).equals(retained)), True)
    audit.event("frozen_high", len(high), expected["high"])
    audit.compare(all_paths, tables["18g"], snapshot_pairs(), "18f_to_18g")
    audit.compare(high, tables["18f_high"], snapshot_pairs(), "18f_high_subset")
    audit.compare(high, tables["19a"], snapshot_pairs(), "18f_to_19a")
    ghigh = tables["18g"].loc[booleans(tables["18g"]["highest_robustness"])]
    audit.compare(ghigh, tables["19a"], [("phenotype_domain", "phenotype_domain", "text")], "18g_to_19a_domains")
    h = tables["19a"].copy()
    need(h, ["phenotype_domain"], "19a")
    if h["phenotype_domain"].isna().any() or set(h["phenotype_domain"]) - set(DOMAINS):
        raise AuditError("Unknown/missing phenotype domains")
    counts = h.groupby("species")["phenotype_domain"].nunique()
    audit.event("species", len(counts), expected["species"])
    for name, n in [("pan", 3), ("cross", 2), ("single", 1)]:
        audit.event(name, int(counts.eq(n).sum()), expected[name])
    for domain in DOMAINS:
        audit.event(domain, int(h["phenotype_domain"].eq(domain).sum()), expected[domain])
    summary = tables["19a_species"]
    a, b = audit.align(counts.rename("phenotype_domains").reset_index(), summary, "19a_species", keys=["species"])
    need(b, ["phenotype_domains", "highest_robust_paths", "diet_exposures", "cgm_outcomes"], "19a_species")
    audit.event("species_domain_counts", bool(a["phenotype_domains"].eq(b["phenotype_domains"]).all()), True)
    for column, values in [("highest_robust_paths", h.groupby("species").size()),
                           ("diet_exposures", h.groupby("species")["exposure"].nunique()),
                           ("cgm_outcomes", h.groupby("species")["outcome_field"].nunique())]:
        audit.event(column, bool(b[column].eq(b["species"].map(values)).all()), True)
    for name, n in [("19a_pan", 3), ("19a_cross", 2)]:
        audit.align(counts[counts.eq(n)].reset_index(), tables[name], name, keys=["species"])
    h["domain_breadth"] = h["species"].map(counts)
    h["breadth_class_19b"] = h["domain_breadth"].map({3: "pan_domain_3", 2: "cross_domain_2", 1: "single_domain_detected"})
    ids = {species: "T{:03d}".format(i + 1) for i, species in enumerate(sorted(counts.index))}
    h["taxon_id"] = h["species"].map(ids)
    h["display_species"] = h["species"].map(lambda s: s.split("|")[-1].replace("s__", "", 1))
    return all_paths, h


def discover(source):
    found = {tag: [] for tag in TAGS}
    for run in sorted((source / "outputs").glob("paper20_mediation_*")):
        name = run.name
        if "protocol_abxppi" in name and name.endswith("_primary"):
            tag = "protocol"
        elif "protocol_abxppi" not in name and name.endswith("_strict"):
            tag = "strict"
        elif "protocol_abxppi" not in name and name.endswith("_primary"):
            tag = "primary"
        else:
            continue
        found[tag].extend(sorted((run / "models").glob("*mediation_paths*.csv")))
    return found


def match_model(audit, reference, frame, tag):
    sig = selected_column(frame, SIG, tag + " significance")
    con = selected_column(frame, CON, tag + " consistency")
    indirect = selected_column(frame, ["indirect_effect", "indirect"], tag + " indirect")
    mapping = {"significant": sig, "consistent": con, "indirect_effect": indirect,
               "N": "N", "FDR": "FDR_BH_within_score_outcome", "bootstrap_p": "bootstrap_p_indirect"}
    pairs = [(base + "_" + tag, col, "bool" if base in ["significant", "consistent"] else "number") for base, col in mapping.items()]
    audit.compare(reference, frame, pairs, "model_" + tag)
    return mapping


def select_models(audit, all_paths, source, explicit):
    candidates = discover(source)
    selected = {}
    for tag in TAGS:
        files = [Path(explicit[tag])] if tag in explicit else candidates[tag]
        matches = []
        for path in files:
            try:
                frame = audit.read(path, "candidate_" + tag)
                mapping = match_model(audit, all_paths, frame, tag)
                matches.append((path.resolve(), frame, mapping))
                reason = "all frozen keys/effects/N/FDR/p/labels match within recorded tolerance"
                ok = True
            except (AuditError, ValueError, OSError, pd.errors.ParserError) as exc:
                reason = str(exc)
                ok = False
            audit.candidates.append({"kind": tag, "path": str(path), "sha256": audit.files.get(str(path.resolve()), {}).get("sha256", ""), "matches_18f": ok, "reason": reason})
        if len(matches) != 1:
            raise AuditError("{}: {} matching models among {} candidates; need exactly one. See 19b_model_candidates.csv; use --inputs config/inputs.json for explicit paths.".format(tag, len(matches), len(files)))
        path, frame, mapping = matches[0]
        audit.selected[tag] = str(path)
        audit.mappings[tag] = mapping
        selected[tag] = frame
        # Legacy run metadata is evidence to retain, not authority to overwrite CSV checks.
        manifest = path.parent.parent / "reports" / "manifest.json"
        if manifest.is_file():
            audit.record_file(manifest, "legacy_manifest_" + tag)
            audit.notes.append({"legacy_manifest": str(manifest), "content": json.loads(manifest.read_text(encoding="utf-8"))})
    audit.notes.append("Selection proves content compatibility with 18f; legacy 18f has no source-hash manifest, so historical provenance is not independently proven.")
    return selected


def validate_model(audit, frame, tag):
    numeric = list(EFFECTS.values()) + ["direct_effect", "total_effect", "N", "FDR_BH_within_score_outcome", "bootstrap_p_indirect", "bootstrap_n"]
    numeric += [effect + "_CI95_" + side for effect in ["indirect", "direct", "total"] for side in ["lower", "upper"]]
    need(frame, numeric + ["sample_sha256", "covariates", "role"], tag)
    for col in numeric:
        x = pd.to_numeric(frame[col], errors="coerce")
        invalid = ~np.isfinite(x)
        if invalid.any():
            for _, row in frame.loc[invalid, KEYS].iterrows():
                audit.differences.append(dict(row, check=tag, field=col, difference="missing_or_nonfinite"))
            raise AuditError("{}: {} has {} missing/nonfinite/nonnumeric values".format(tag, col, int(invalid.sum())))
    for col in ["N", "bootstrap_n"]:
        x = pd.to_numeric(frame[col])
        if ((x <= 0) | (x % 1 != 0)).any():
            raise AuditError(tag + ": invalid positive integer " + col)
    for col in ["FDR_BH_within_score_outcome", "bootstrap_p_indirect"]:
        if not pd.to_numeric(frame[col]).between(0, 1).all():
            raise AuditError(tag + ": invalid probability " + col)
    for effect in ["indirect", "direct", "total"]:
        if (frame[effect + "_CI95_lower"] > frame[effect + "_CI95_upper"]).any():
            raise AuditError(tag + ": reversed CI " + effect)
    product = frame[EFFECTS["a"]] * frame[EFFECTS["b"]]
    ok = np.isclose(product, frame["indirect_effect"], atol=audit.args.atol, rtol=audit.args.rtol)
    if not ok.all():
        for i in np.flatnonzero(~ok):
            audit.differences.append(dict(frame.iloc[i][KEYS], check=tag, field="a*b", expected=frame.iloc[i]["indirect_effect"], observed=product.iloc[i]))
        raise AuditError(tag + ": a*b differs from existing indirect effect")
    for col in ["sample_sha256", "covariates", "role"]:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise AuditError(tag + ": empty " + col)
    if not frame["sample_sha256"].astype(str).str.match(r"^[0-9a-fA-F]{64}$").all():
        raise AuditError(tag + ": invalid sample_sha256")
    if tag == "protocol":
        covs = frame["covariates"].astype(str).map(lambda s: set(x.strip() for x in s.split(";")))
        entered = covs.map(lambda s: bool(s & {"antibiotic_use", "antibiotic_use_protocol_window"}) and bool(s & {"ppi_use", "ppi_use_protocol_window"}))
        if not entered.all():
            raise AuditError("Protocol CSV covariates do not include both antibiotic and PPI on every path")
    audit.event(tag + "_model_numeric_and_product_checks", "PASS")


def attach_models(audit, high, models):
    h = high.copy()
    for tag, frame in models.items():
        validate_model(audit, frame, tag)
        # Preserve every original model column under an explicit namespace.
        cols = {c: "model_" + tag + "__" + c for c in frame.columns if c not in KEYS}
        h = h.merge(frame.rename(columns=cols), on=KEYS, how="left", validate="one_to_one")
        for name, column in EFFECTS.items():
            h[name + "_sign_" + tag] = signs(h["model_" + tag + "__" + column], audit.args.zero_eps)
        lo = h["model_" + tag + "__indirect_CI95_lower"]
        hi = h["model_" + tag + "__indirect_CI95_upper"]
        h["indirect_ci_" + tag] = np.select([lo > 0, hi < 0], ["positive", "negative"], default="includes_zero")
        # Point signs and bootstrap CI labels are separate; b/a have no CI in source.
        if not h["indirect_ci_" + tag].isin(["positive", "negative"]).all():
            audit.notes.append(tag + ": some inherited highest paths have indirect CI including zero; preserved as-is")
    for name in EFFECTS:
        h[name + "_stability"] = stability(h[[name + "_sign_" + t for t in TAGS]])
    for tag in ["strict", "protocol"]:
        audit.event(tag + "_N_not_greater_than_primary", bool((h["model_" + tag + "__N"] <= h["model_primary__N"]).all()), True)
    return h


def add_dictionary(audit, h, path):
    dictionary = audit.read(path, "direction_dictionary")
    need(dictionary, ["kind", "field", "label", "higher_means", "verification", "domain", "source", "limitations"], "dictionary")
    unique(dictionary, ["kind", "field"], "dictionary")
    for kind, key in [("exposure", "exposure"), ("outcome", "outcome_field")]:
        d = dictionary[dictionary["kind"].eq(kind)].set_index("field")
        missing = set(h[key]) - set(d.index)
        if missing:
            raise AuditError("Unmapped {} dictionary fields: {}".format(kind, sorted(missing)))
        for col in ["label", "higher_means", "verification", "source", "limitations"]:
            if d.loc[sorted(set(h[key])), col].isna().any():
                raise AuditError("Missing dictionary text: " + kind + " " + col)
            h[kind + "_" + col + "_19b"] = h[key].map(d[col])
        if kind == "outcome":
            audit.event("dictionary_inherited_domains", bool(h[key].map(d["domain"]).eq(h["phenotype_domain"]).all()), True)
    h["biological_direction_status"] = "descriptive_only_no_health_classification"
    h["microbiome_scale"] = "CLR-Z relative composition, not absolute abundance (local code)"
    dictionary.to_csv(audit.out / "19b_direction_dictionary_used.csv", index=False)
    return h


def summarize(h, keys):
    rows = []
    grouper = keys[0] if len(keys) == 1 else keys
    for key, group in h.groupby(grouper, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row.update(taxon_id=group["taxon_id"].iloc[0], display_species=group["display_species"].iloc[0],
                   paths=len(group), diets=group["exposure"].nunique(), outcomes=group["outcome_field"].nunique(),
                   domains=group["phenotype_domain"].nunique(), domain_breadth=int(group["domain_breadth"].iloc[0]))
        for tag in TAGS:
            for name in EFFECTS:
                values = group[name + "_sign_" + tag]
                prefix = name + "_" + tag
                row[prefix + "_class"] = sign_class(values)
                for sign in ["positive", "negative", "near_zero", "unknown"]:
                    row[prefix + "_" + sign] = int(values.eq(sign).sum())
        for name in EFFECTS:
            for state in ["stable", "sign_flip", "uncertain"]:
                row[name + "_" + state + "_paths"] = int(group[name + "_stability"].eq(state).sum())
        row["N_primary_min"] = int(group["model_primary__N"].min())
        row["N_primary_max"] = int(group["model_primary__N"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def draw_plots(h, out, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    def save_figure(fig, filename, dpi):
        if synthetic:
            fig.text(.995, .002, "SYNTHETIC TEST ONLY", ha="right", va="bottom", fontsize=9,
                     color="#a00000", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        fig.savefig(out / filename, dpi=dpi)

    palette = ListedColormap(["#3274a1", "#e1e1e1", "#c44e52"])
    palette.set_bad("#ffffff")
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    mapping = {"negative": -1, "near_zero": 0, "positive": 1, "unknown": np.nan}
    symbols = {"negative": "-", "near_zero": "0", "positive": "+", "unknown": "?"}
    focus = h[h["domain_breadth"].ge(2)]
    columns = [name + "_sign_" + tag for tag in TAGS for name in EFFECTS]
    for species, group in focus.groupby("species", sort=True):
        g = group.sort_values(["phenotype_domain", "exposure", "outcome_field"])
        matrix = np.array([[mapping[v] for v in row] for row in g[columns].to_numpy()])
        fig, ax = plt.subplots(figsize=(10, max(4.5, len(g) * .25 + 2)))
        ax.imshow(matrix, cmap=palette, norm=norm, aspect="auto")
        for i, row in enumerate(g[columns].to_numpy()):
            for j, value in enumerate(row):
                ax.text(j, i, symbols[value], ha="center", va="center", fontsize=8)
        ax.set_xticks(range(9))
        ax.set_xticklabels([t + "\n" + n for t in TAGS for n in EFFECTS], fontsize=8)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([r.exposure + " | " + r.outcome_field for r in g.itertuples()], fontsize=7)
        ax.set_title(g["taxon_id"].iloc[0] + " " + g["display_species"].iloc[0] + "\nPoint-estimate signs; no health classification", fontsize=10)
        fig.tight_layout()
        save_figure(fig, "19b_" + g["taxon_id"].iloc[0] + "_path_directions.png", 160)
        plt.close(fig)

    # One cell = one exact taxon/exposure/outcome path; never average coefficients.
    pan = h[h["domain_breadth"].eq(3)].copy()
    pan["row_label"] = pan["taxon_id"] + " " + pan["display_species"] + " | " + pan["exposure"]
    values = pan.pivot(index="row_label", columns="outcome_field", values="indirect_sign_primary")
    matrix = np.array([[mapping.get(v, np.nan) for v in row] for row in values.to_numpy()])
    fig, ax = plt.subplots(figsize=(13, max(5, len(values) * .32 + 2)))
    ax.imshow(matrix, cmap=palette, norm=norm, aspect="auto")
    for i, row in enumerate(values.to_numpy()):
        for j, value in enumerate(row):
            if value in symbols:
                ax.text(j, i, symbols[value], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(values.index, fontsize=8)
    ax.set_title("Pan-domain taxa: primary indirect-effect sign\nBlank = absent from frozen set; +/- = outcome increase/decrease, not benefit/harm")
    fig.tight_layout()
    save_figure(fig, "19b_core_direction_heatmap.png", 180)
    plt.close(fig)

    reps = representative_paths(h)
    fig, axes = plt.subplots(len(reps), 1, figsize=(11, max(4, len(reps) * 1.15)), squeeze=False)
    for ax, (_, row) in zip(axes.flat, reps.iterrows()):
        for i, tag in enumerate(TAGS):
            prefix = "model_" + tag + "__"
            ax.hlines(i, row[prefix + "indirect_CI95_lower"], row[prefix + "indirect_CI95_upper"], color=["#222222", "#3274a1", "#d98e32"][i], linewidth=2)
            ax.plot(row[prefix + "indirect_effect"], i, "o", color=["#222222", "#3274a1", "#d98e32"][i], markersize=4)
        ax.axvline(0, color="gray", linestyle="--", linewidth=.7)
        ax.set_yticks(range(3))
        ax.set_yticklabels(TAGS, fontsize=7)
        ax.set_ylim(-.6, 2.6)
        ax.set_title(row["taxon_id"] + " | " + row["exposure"] + " | " + row["outcome_field"], fontsize=8, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Illustrative paths: indirect effect and original bootstrap 95% CI\nFirst exposure/outcome alphabetically within each pan taxon/domain; separate x scales", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .965])
    save_figure(fig, "19b_core_indirect_forest.png", 170)
    plt.close(fig)


def representative_paths(h):
    # Predeclared display rule, not an effect-size ranking or new selection test.
    pan = h[h["domain_breadth"].eq(3)]
    return pan.sort_values(["species", "phenotype_domain", "exposure", "outcome_field"]).groupby(
        ["species", "phenotype_domain"], sort=True).head(1)


def write_core_pages(audit, h):
    """Each text file is one core taxon and can be displayed in a terminal screenshot."""
    symbols = {"positive": "+", "negative": "-", "near_zero": "0", "unknown": "?"}
    for species, group in h[h["domain_breadth"].eq(3)].groupby("species", sort=True):
        g = group.sort_values(["phenotype_domain", "exposure", "outcome_field"])
        taxon_id = g["taxon_id"].iloc[0]
        lines = ["=== STEP19B CORE DETAIL: {} ===".format(taxon_id),
                 g["display_species"].iloc[0],
                 "DATA_KIND=" + ("SYNTHETIC_TEST_ONLY" if getattr(audit.args, "synthetic", False) else "SERVER_RESULT_TABLES"),
                 "Each row is one frozen path. Signs are not health labels.",
                 "a/b/I = primary point signs; I = indirect; stable = I across 3 models.",
                 "{:<17} {:<16} {:<5} {:>10} {:>24} {:>6} {:>9}".format(
                     "Diet", "Outcome", "a/b/I", "I", "95% CI", "N", "stable")]
        for _, row in g.iterrows():
            lines.append("{:<17} {:<16} {:<5} {:>10.3g} [{:>10.3g}, {:>10.3g}] {:>6} {:>9}".format(
                row["exposure"], row["outcome_field"],
                "/".join(symbols[row[n + "_sign_primary"]] for n in EFFECTS),
                row["model_primary__indirect_effect"], row["model_primary__indirect_CI95_lower"],
                row["model_primary__indirect_CI95_upper"], int(row["model_primary__N"]), row["indirect_stability"]))
        lines.append("Full taxonomy and all 3 models: 19b_pan_domain_core_details.csv")
        (audit.out / ("19b_CORE_" + taxon_id + "_SCREENSHOT.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(audit, h, all_paths):
    out = audit.out
    h = h.sort_values(KEYS).reset_index(drop=True)
    h.to_csv(out / "19b_path_direction_details.csv", index=False)
    summary = summarize(h, ["species"])
    summary.to_csv(out / "19b_species_direction_summary.csv", index=False)
    summarize(h, ["species", "exposure", "phenotype_domain"]).to_csv(out / "19b_species_diet_domain_summary.csv", index=False)
    # Compact per-path table remains explicit about a, b and indirect signs.
    compact = KEYS + ["taxon_id", "display_species", "phenotype_domain", "domain_breadth"]
    compact += [n + "_sign_" + t for t in TAGS for n in EFFECTS] + [n + "_stability" for n in EFFECTS]
    h[compact].to_csv(out / "19b_species_diet_outcome_directions.csv", index=False)
    for name, breadth in [("pan_domain_core", 3), ("cross_domain2", 2)]:
        h[h["domain_breadth"].eq(breadth)].to_csv(out / ("19b_" + name + "_details.csv"), index=False)
        summary[summary["domain_breadth"].eq(breadth)].to_csv(out / ("19b_" + name + "_summary.csv"), index=False)
    representative_paths(h).to_csv(out / "19b_representative_paths.csv", index=False)
    write_core_pages(audit, h)
    if not audit.args.no_plots:
        draw_plots(h, out, synthetic=getattr(audit.args, "synthetic", False))
    lines = ["=== STEP19B DIRECTION SYNTHESIS ===", "STATUS=PASS", "MODELS_REFIT=False | FDR_RECOMPUTED=False",
             "DATA_KIND=" + ("SYNTHETIC_TEST_ONLY" if getattr(audit.args, "synthetic", False) else "SERVER_RESULT_TABLES"),
             "CANDIDATES={} | FROZEN_PATHS={} | TAXA={}".format(len(all_paths), len(h), len(summary)),
             "PAN={} | CROSS2={} | SINGLE={}".format(int(summary.domain_breadth.eq(3).sum()), int(summary.domain_breadth.eq(2).sum()), int(summary.domain_breadth.eq(1).sum())),
             "INPUT_MATCH=PASS | A_TIMES_B=PASS | INHERITED_LABELS=PASS", "", "--- PAN-DOMAIN CORE: PRIMARY SIGNS ---"]
    for _, row in summary[summary.domain_breadth.eq(3)].iterrows():
        lines.append("{} {} | paths={}".format(row.taxon_id, row.display_species, row.paths))
        lines.append("  b(+/-/0/?)={}/{}/{}/{} | indirect(+/-/0/?)={}/{}/{}/{} | stable={}/{}".format(
            *[row[n + "_primary_" + s] for n in ["b", "indirect"] for s in ["positive", "negative", "near_zero", "unknown"]], row.indirect_stable_paths, row.paths))
    lines += ["", "--- ALL FROZEN PATHS: SIGN STABILITY ---"]
    for name in EFFECTS:
        lines.append("{}: stable={} | flip={} | uncertain={}".format(name, *[int(h[name + "_stability"].eq(state).sum()) for state in ["stable", "sign_flip", "uncertain"]]))
    pending = h[h["outcome_verification_19b"].str.contains("pending")]["outcome_field"].unique()
    lines += ["PENDING_OUTCOME_MEANING=" + (",".join(sorted(pending)) or "none"),
              "INTERPRETATION=associations only; +/- are mathematical signs, not benefit/harm",
              "FIGURES=" + ("SKIPPED_EXPLICITLY" if audit.args.no_plots else "CREATED"), "OUTPUT_DIR=" + str(out)]
    (out / "19b_SCREENSHOT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = ["# Step19b 核心菌方向整合", "", "数据状态：" + lines[3], "",
              "保留原始冻结路径和所有模型字段；不拟合模型、不重算FDR、不按文献或健康方向筛选。", "",
              "## 终端摘要", "", "```text", *lines, "```", "", "## 阅读方法", "",
              "- a：饮食分析指标升高与菌群CLR-Z分析值的关系；不是绝对菌量。",
              "- b：控制饮食及协变量后菌群分析值与结局的关系。",
              "- indirect=a×b：饮食经该单菌统计路径与结局的关联方向。",
              "- 三套模型的a/b/间接效应分别检查符号；近零或未知不计作稳定正/负。",
              "- a与b在原模型中没有置信区间；其图仅显示点估计符号。间接效应保留原bootstrap CI。",
              "- mixed只是跨不同饮食/结局的数值正负混合，不能解释成有益与有害混合。",
              "- 不平均不同结局的系数；不累加单菌中介比例；相关路径数量不是独立重复次数。",
              "- 跨领域标签是显著路径分布，不证明生物学特异性；这些是横断面关联。",
              "- 森林图预先采用每个三领域菌×领域内按exposure/outcome_field字典序首条路径，仅作示例，各小图横轴独立。",
              "- SDw、COGI和GRADE eugly等解释核查状态见字典；未核实状态不会生成健康分类。", "",
              "## 输入版本限制", "", "18f旧代码没有来源哈希清单。自动选择要求完整1490路径的间接效应、N、FDR、p及标签唯一匹配；这证明内容兼容，不独立证明历史文件身份。有多个匹配候选时要求显式输入。", "",
              "## 文件索引", "", "- `19b_path_direction_details.csv`：完整逐路径三模型原字段、方向、CI、N和样本哈希。",
              "- `19b_species_direction_summary.csv`：全部菌的符号分布，保留完整taxonomy。",
              "- `19b_species_diet_domain_summary.csv`：分饮食/领域汇总。",
              "- `19b_pan_domain_core_*`、`19b_cross_domain2_*`：重点菌全量结果。",
              "- `19b_CORE_Txxx_SCREENSHOT.txt`：每个三领域菌的逐路径方向、间接效应、CI、N，便于终端截图。",
              "- `19b_Txxx_path_directions.png`：每个三/双领域菌逐路径九列符号图。",
              "- `19b_core_direction_heatmap.png`、`19b_core_indirect_forest.png`：核心菌概览与原CI。",
              "- `manifest.json`、`19b_input_*.csv`、`19b_model_candidates.csv`：来源、选择依据、字段映射和核查。"]
    (out / "19b_direction_synthesis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return lines


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", default=str(PACKAGE.parent / "4New_cgm"), help="Existing 4New_cgm folder")
    p.add_argument("--inputs", help="JSON with explicit primary/strict/protocol and optional 18f/18g/19a paths")
    p.add_argument("--dictionary", default=str(PACKAGE / "config" / "direction_dictionary.csv"))
    p.add_argument("--output-parent", default=str(PACKAGE / "outputs"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Full checks; write audit only")
    mode.add_argument("--inventory-only", action="store_true", help="List existing source files only")
    p.add_argument("--no-plots", action="store_true", help="Explicitly skip plots; tables still generated")
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--zero-eps", type=float, default=1e-12)
    return p


def run(args, expected=None):
    expected = EXPECTED if expected is None else expected
    source = Path(args.source_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.output_parent).resolve() / ("step19b_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    audit = Audit(out, args)
    try:
        for name in ["atol", "rtol", "zero_eps"]:
            if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                raise AuditError(name + " must be finite and nonnegative")
        explicit = {}
        if args.inputs:
            config = audit.record_file(args.inputs, "input_selection")
            explicit = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(explicit, dict) or set(explicit) - set(TAGS + list(SOURCE_FILES)):
                raise AuditError("inputs JSON must contain only documented input path keys")
            explicit = {k: str((config.parent / v).resolve()) if not Path(v).is_absolute() else v for k, v in explicit.items()}
        paths = {name: Path(explicit.get(name, source / "outputs" / rel)) for name, rel in SOURCE_FILES.items()}
        if args.inventory_only:
            lines = ["=== STEP19B INPUT INVENTORY ===", "SOURCE_ROOT=" + str(source)]
            lines += ["{} | exists={} | {}".format(k, v.is_file(), v) for k, v in paths.items()]
            for tag, files in discover(source).items():
                lines.append("{} models={}".format(tag, len(files)))
                lines += [str(p) for p in files]
            (out / "19b_SCREENSHOT_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
            print("\n".join(lines))
            audit.save("INVENTORY_ONLY")
            return 0, out
        all_paths, high = load_frozen(audit, paths, expected)
        models = select_models(audit, all_paths, source, explicit)
        high = attach_models(audit, high, models)
        high = add_dictionary(audit, high, args.dictionary)
        if args.check_only:
            lines = ["STEP19B_CHECK_ONLY=PASS", "NO_DIRECTION_RESULTS_GENERATED=True", "OUTPUT_DIR=" + str(out)]
            (out / "19b_SCREENSHOT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            lines = write_results(audit, high, all_paths)
        # Detect concurrent mutation as well as accidental writes before declaring PASS.
        for path, record in audit.files.items():
            if sha256(path) != record["sha256"]:
                raise AuditError("Input changed during run: " + path)
        audit.event("all_inputs_unchanged", True, True)
        audit.save("CHECK_ONLY_PASS" if args.check_only else "PASS")
        print("\n".join(lines))
        return 0, out
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        lines = ["STEP19B_STATUS=FAIL", "DO_NOT_INTERPRET_PARTIAL_OUTPUTS=True", "ERROR=" + error, "OUTPUT_DIR=" + str(out),
                 "Send this summary screenshot. Audit/candidate details are saved in this folder."]
        if audit.candidates:
            lines.append("--- MODEL CANDIDATES (first 6; full CSV saved) ---")
            for item in audit.candidates[:6]:
                lines.append("{} | match={} | {}".format(item["kind"], item["matches_18f"], item["path"]))
                if not item["matches_18f"]:
                    lines.append("  " + item["reason"][:180])
        (out / "19b_SCREENSHOT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        audit.save("FAIL", error)
        print("\n".join(lines))
        return 2, out


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args())[0])
