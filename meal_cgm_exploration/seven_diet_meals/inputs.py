"""Strict source contracts; old outputs are read-only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hppmeal.common import (KEYS, require, sha256, read_csv, header, norm,
                            unique, select_participant_scope, validate_config)
from . import VERSION

PACKAGE = Path(__file__).resolve().parents[1]
MODULE = PACKAGE / "seven_diet_meals"
DIETS = ["AHEI", "AMED", "hPDI", "rEDIH", "EAT13", "NOVA4", "Carbohydrate_pct"]
SCORE_SOURCE = {"AHEI": ("current4", "AHEI_z"), "AMED": ("current4", "AMED_z"),
                "rEDIH": ("current4", "rEDIH_z"), "hPDI": ("corrected_hpdi", "hpdi_score_energy_adjusted_z"),
                "EAT13": ("new3", "EAT13_z"), "NOVA4": ("new3", "NOVA4_z"),
                "Carbohydrate_pct": ("new3", "Carbohydrate_pct_z")}
BASE = ["age_years", "sex", "education_level", "smoking_status", "sleep_duration_hours_day",
        "physical_activity_met_h_week", "vitamin_use", "hormone_use", "bmi", "cgm_device_type"]
CATEGORIES = ["sex", "education_level", "smoking_status", "cgm_device_type"]
MEAL_NUM = ["start", "local_hour", "duration_minutes", "carbohydrate_g", "calories_kcal",
            "baseline_glucose", "pre_slope_per_min", "previous_gap_minutes", "iauc_net_0_120", "peak_rise_0_120"]
MEAL_COLS = KEYS + ["meal_id", "local_date", "analysis_ready"] + MEAL_NUM


def strict_number(s, label):
    clean = norm(s)
    result = pd.to_numeric(clean, errors="coerce").astype(float)
    require(not (clean.notna() & ~np.isfinite(result)).any(), "Invalid numeric values: " + label)
    return result


def bundle():
    f = MODULE / "MANIFEST.json"
    require(f.is_file(), "Upload complete seven_diet_meals folder: MANIFEST.json missing")
    digest = sha256(f)
    b = json.loads(f.read_text(encoding="utf-8"))
    required = {"seven_diet_meals/" + x for x in ["__init__.py", "inputs.py", "models.py", "reporting.py",
                 "run_analysis.py", "run_seven_diets.sh", "sources.json", "README.md", "PLAN.md"]}
    require(b.get("version") == VERSION and required.issubset(b.get("files", {})), "Incomplete or mismatched upload manifest")
    for name, value in b["files"].items():
        p = (PACKAGE / name).resolve()
        require(MODULE in p.parents and p.is_file() and sha256(p) == value, "Incomplete/changed upload: " + name)
    require(sha256(f) == digest, "Manifest changed while reading")
    result = {str(PACKAGE / k): v for k, v in b["files"].items()}
    result[str(f)] = digest
    return result


def inspect(source, root_override, config_path):
    source = Path(source).resolve()
    manifest = source / "manifest.json"
    require(manifest.is_file(), "Original meal manifest missing: " + str(manifest))
    expected = {str(manifest): sha256(manifest)}
    old = json.loads(manifest.read_text(encoding="utf-8"))
    c = validate_config(old["config"])
    require(old["data_kind"] == c["data_kind"], "Original data-kind mismatch")
    require(c["time"]["consumption_time_verified"], "Use the completed meal run with reviewed food-time interpretation")
    for stage in ["audit", "build"]:
        require(old["status"].get(stage, {}).get("status") == "PASS", "Original " + stage + " must be PASS")
    require(not any(x.get("status") in ["RUNNING", "FAILED", "INTERRUPTED"] for x in old["status"].values()), "Original run incomplete")
    fp = source / "code_fingerprint.json"
    require(fp.is_file(), "Original code fingerprint missing")
    expected[str(fp)] = sha256(fp)
    core = json.loads(fp.read_text())
    current = {str(p.relative_to(PACKAGE)): sha256(p) for folder in ["hppmeal", "scripts"] for p in sorted((PACKAGE / folder).glob("*.py"))}
    require(core == current, "Original hppmeal/scripts differs from successful BUILD")
    expected.update({str(PACKAGE / name): digest for name, digest in core.items()})
    meals = source / "private/02_meals.csv"
    require(meals.is_file(), "Original meal table missing")
    expected[str(meals)] = old["status"]["build"]["meals_sha256"]
    root = Path(root_override or old["root"]).expanduser().resolve()
    require(root.is_dir(), "Data root missing: " + str(root))
    require(not (root / "SYNTHETIC_TEST_ONLY.json").exists() or c["data_kind"] == "SYNTHETIC", "Synthetic inputs cannot be labelled REAL")
    config_path = Path(config_path).expanduser().resolve()
    require(config_path.is_file(), "Score sources configuration missing")
    expected[str(config_path)] = sha256(config_path)
    settings = json.loads(config_path.read_text())
    require(set(settings) == {"current4", "new3", "corrected_hpdi"}, "sources.json must specify current4, new3, corrected_hpdi")
    paths = {k: (root / Path(v).expanduser()).resolve() for k, v in settings.items()}
    cov = (root / Path(c["inputs"]["covariates"]).expanduser()).resolve()
    previous = [x for x in old["inputs"] if x["label"] == "covariates"]
    require(len(previous) == 1, "Original ANALYZE must have recorded its covariate table")
    expected[str(cov)] = previous[0]["sha256"]
    paths.update(covariates=cov, meals=meals)
    required = {"current4": ["participant_id", "AHEI_z", "AMED_z", "rEDIH_z"],
                "new3": ["participant_id", "EAT13_z", "NOVA4_z", "Carbohydrate_pct_z", "mean_daily_energy_kcal"],
                "corrected_hpdi": ["participant_id", "hpdi_score_energy_adjusted_z"],
                "covariates": ["participant_id", "cohort", "exclude_diabetes_or_a10"] + BASE,
                "meals": MEAL_COLS}
    issues, audit = [], []
    for name, p in paths.items():
        missing = required[name] if not p.is_file() else sorted(set(required[name]) - set(header(p)))
        audit.append(dict(source=name, file_present=p.is_file(), missing_columns=";".join(missing)))
        if not p.is_file() or missing:
            issues.append("{}: {} ; missing fields: {}".format(name, p, ", ".join(missing)))
        else:
            if str(p) not in expected:
                expected[str(p)] = sha256(p)
    require(not issues, "Input check failed:\n" + "\n".join(issues))
    for name, digest in expected.items():
        require(Path(name).is_file() and sha256(name) == digest, "Source changed or differs from original run: " + name)
    return root, c, paths, expected, pd.DataFrame(audit)


def record_expected(run, expected):
    for name, digest in expected.items():
        p = run.record(Path(name), "source_" + Path(name).name)
        recorded = next(x for x in run.inputs if x["path"] == str(p))
        require(recorded["sha256"] == digest, "Source changed between validation and read: " + name)


def unchanged(run):
    for item in run.inputs:
        require(Path(item["path"]).is_file() and sha256(item["path"]) == item["sha256"], "Input changed during analysis: " + item["label"])


def score_frame(run, path, label):
    d = read_csv(path, ["participant_id"])
    # Derived participant score exports may omit scope; their canonical path is explicit.
    for col in ["cohort", "research_stage", "cgm_research_stage"]:
        if col in d:
            values = norm(d[col])
            expected = run.c["cohort" if col == "cohort" else "research_stage"]
            require(values.notna().all() and values.eq(expected).all(), label + ": wrong/missing cohort or visit")
    d["participant_id"] = norm(d.participant_id)
    unique(d, ["participant_id"], label)
    explicit = "cohort" in d and any(x in d for x in ["research_stage", "cgm_research_stage"])
    run.table(pd.DataFrame([dict(source=label, rows=len(d), scope="explicit columns" if explicit
                                else "canonical baseline export; missing scope columns not independently verified")]), "00_" + label + "_scope.csv")
    return d.set_index("participant_id")


def load(run, paths):
    source = {k: score_frame(run, paths[k], k) for k in ["current4", "new3", "corrected_hpdi"]}
    score, mappings = {}, []
    for diet in DIETS:
        label, col = SCORE_SOURCE[diet]
        score[diet] = strict_number(source[label][col], diet)
        mappings.append(dict(diet=diet, source=label, column=col, scale="original Z unit; no re-standardization",
                             definition={"AHEI": "modified AHEI 7 components", "AMED": "adapted AMED 8 components",
                                         "rEDIH": "already reversed EDIH", "EAT13": "modified EAT-Lancet 13",
                                         "NOVA4": "personal ultra-processed energy proportion",
                                         "Carbohydrate_pct": "personal carbohydrate energy proportion"}.get(diet, "corrected hPDI")))
    people = pd.DataFrame(score)
    people["mean_daily_energy_kcal"] = strict_number(source["new3"].mean_daily_energy_kcal, "daily energy")
    require(people.mean_daily_energy_kcal.dropna().gt(0).all(), "Nonpositive daily energy")
    cov = select_participant_scope(run, read_csv(paths["covariates"]), "covariates")
    for col in CATEGORIES:
        cov[col] = norm(cov[col])
    for col in [x for x in BASE if x not in CATEGORIES] + ["exclude_diabetes_or_a10"]:
        cov[col] = strict_number(cov[col], col)
    for col in ["vitamin_use", "hormone_use", "exclude_diabetes_or_a10"]:
        require(cov[col].dropna().isin([0, 1]).all(), "Nonbinary covariate: " + col)
    for col in ["known_diabetes", "a10_medication_use"]:
        if col in cov:
            values = strict_number(cov[col], col)
            require(values.dropna().isin([0, 1]).all(), "Nonbinary " + col)
            require(not (values.eq(1) & ~cov.exclude_diabetes_or_a10.eq(1)).any(), "Diabetes/A10 flags contradict exclusion flag")
    alcohol = [x for x in ["alcohol_intake_g_day", "alcohol_g_day", "alcohol_intake"] if x in cov]
    require(alcohol, "Covariate alcohol column missing; hPDI requires its original alcohol adjustment")
    chosen = strict_number(cov[alcohol[0]], alcohol[0])
    for alias in alcohol[1:]:
        require(np.allclose(chosen, strict_number(cov[alias], alias), equal_nan=True), "Conflicting alcohol columns")
    cov["alcohol_intake_g_day"] = chosen
    flags = cov.exclude_diabetes_or_a10
    cov["primary_eligible"] = ~flags.eq(1) & cov[BASE].notna().all(axis=1)
    cov["strict_eligible"] = flags.eq(0) & cov[BASE].notna().all(axis=1)
    people = people.join(cov.set_index("participant_id")[BASE + ["alcohol_intake_g_day", "primary_eligible", "strict_eligible"]], how="outer")
    for col in ["primary_eligible", "strict_eligible"]:
        people[col] = people[col].eq(True)
    people.index.name = "participant_id"
    d = read_csv(paths["meals"], MEAL_COLS, usecols=MEAL_COLS)
    for key in KEYS + ["meal_id", "local_date"]:
        d[key] = norm(d[key])
    unique(d, ["meal_id"], "meal table")
    require(d[KEYS].notna().all().all() and d.cohort.eq(run.c["cohort"]).all()
            and d.research_stage.eq(run.c["research_stage"]).all(), "Meal scope mismatch")
    require(d.analysis_ready.str.lower().isin(["true", "false"]).all(), "Invalid meal eligibility flags")
    d = d.loc[d.analysis_ready.str.lower().eq("true")].copy()
    for col in MEAL_NUM:
        d[col] = strict_number(d[col], col)
    d["mean_increment"] = d.iauc_net_0_120 / 120
    d["mean_glucose"] = d.mean_increment + d.baseline_glucose
    require(pd.to_datetime(d.local_date, format="%Y-%m-%d", errors="coerce").notna().all(), "Invalid meal dates")
    require(d.carbohydrate_g.dropna().ge(0).all() and d.calories_kcal.dropna().gt(0).all(), "Invalid meal amounts")
    run.table(pd.DataFrame(mappings), "00_score_mapping.csv")
    run.table(pd.DataFrame([dict(field="alcohol_intake_g_day", source_column=alcohol[0])]), "00_covariate_mapping.csv")
    ready_ids = pd.Index(d.participant_id.unique())
    coverage = []
    for diet in DIETS:
        label, _ = SCORE_SOURCE[diet]
        present = ready_ids.isin(source[label].index)
        known_score = score[diet].reindex(ready_ids).notna().to_numpy()
        coverage.append(dict(diet=diet, source=label, source_people=len(source[label]),
                             source_people_with_score=int(score[diet].notna().sum()),
                             ready_meal_people=len(ready_ids), ready_meal_people_absent_from_source=int((~present).sum()),
                             ready_meal_people_present_but_score_missing=int((present & ~known_score).sum()),
                             ready_meal_people_with_score=int(known_score.sum()),
                             ready_meals_with_score=int(d.participant_id.isin(ready_ids[known_score]).sum()),
                             scope="analysis_ready meals before covariate and repeated-support selection"))
    run.table(pd.DataFrame(coverage), "00_score_coverage.csv")
    return d.reset_index(drop=True), people
