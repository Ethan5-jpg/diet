#!/usr/bin/env python3
"""Build the participant-level HPP diet-CGM covariate master table.

The population table defines the participant universe.  Every derived source
is left-joined to that universe, so missing source data remains missing and no
participant is removed at this stage.  Model-specific complete-case flags are
reported separately for later analyses.
"""

from __future__ import print_function

import argparse
import ast
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["participant_id", "cohort"]
DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"

MODEL2_COMMON = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "cgm_device_type",
]

CORE_COVARIATES = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "cgm_device_type",
    "alcohol_intake_g_day",
    "bmi",
    "nsaid_aspirin_use",
    "family_history_diabetes",
    "family_history_cvd",
]

EXCLUSION_COLUMNS = [
    "a10_medication_use",
    "known_diabetes",
    "exclude_diabetes_or_a10",
]

MODEL_FLAG_COLUMNS = [
    "amed_model2_covariates_complete",
    "hpdi_model2_covariates_complete",
    "amed_model3_covariates_complete",
    "hpdi_model3_covariates_complete",
    "amed_model4_covariates_complete",
    "hpdi_model4_covariates_complete",
]


def require_columns(frame, columns, label):
    """Raise a readable error when required columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("{} is missing required columns: {}".format(label, missing))


def _normalize_id(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    match = re.match(r"^([0-9]+)\.0$", text)
    return match.group(1) if match else text


def normalize_identity(frame, label):
    """Return a copy with stable participant and cohort string identifiers."""
    require_columns(frame, KEYS, label)
    output = frame.copy()
    output["participant_id"] = output["participant_id"].map(_normalize_id)
    output["cohort"] = output["cohort"].map(
        lambda value: np.nan if pd.isna(value) else str(value).strip()
    )
    if output[KEYS].isna().any().any():
        raise ValueError("{} contains missing participant identity fields".format(label))
    return output


def filter_baseline(frame, label, cohort, research_stage):
    """Restrict a visit-level source to the requested cohort and baseline visit."""
    output = normalize_identity(frame, label)
    output = output.loc[output["cohort"].eq(str(cohort))].copy()
    if "research_stage" in output.columns:
        output["research_stage"] = output["research_stage"].map(
            lambda value: np.nan if pd.isna(value) else str(value).strip()
        )
        output = output.loc[
            output["research_stage"].eq(str(research_stage))
        ].copy()
    return output


def _numeric(series, minimum=None, maximum=None):
    values = pd.to_numeric(series, errors="coerce")
    if minimum is not None:
        values = values.mask(values.lt(minimum))
    if maximum is not None:
        values = values.mask(values.gt(maximum))
    return values.astype(float)


def _parse_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def select_baseline_row(frame, reference_dates, fields, label, cohort, research_stage):
    """Select one deterministic baseline row per participant.

    When a CGM reference date exists, the latest row on or before that date is
    preferred; otherwise the earliest later row is used.  Participants without
    a CGM date use their earliest dated row.  ``array_index`` resolves ties.
    """
    required = KEYS + list(fields) + ["collection_timestamp"]
    require_columns(frame, required, label)
    work = filter_baseline(frame, label, cohort, research_stage)
    if work.empty:
        return pd.DataFrame(columns=required)

    reference = reference_dates[KEYS + ["reference_date"]].copy()
    reference["reference_date"] = _parse_datetime(reference["reference_date"])
    work["_visit_date"] = _parse_datetime(work["collection_timestamp"])
    work = work.merge(reference, on=KEYS, how="left", validate="many_to_one")

    has_visit = work["_visit_date"].notna()
    has_reference = work["reference_date"].notna()
    before = has_visit & has_reference & work["_visit_date"].le(work["reference_date"])
    after = has_visit & has_reference & ~before
    no_reference = has_visit & ~has_reference

    work["_side"] = 2
    work.loc[before, "_side"] = 0
    work.loc[after | no_reference, "_side"] = 1

    timestamp_number = work["_visit_date"].astype("int64").astype(float)
    timestamp_number = timestamp_number.mask(~has_visit, np.inf)
    work["_time_order"] = timestamp_number
    work.loc[before, "_time_order"] = -timestamp_number.loc[before]
    if "array_index" in work.columns:
        work["_array_order"] = pd.to_numeric(
            work["array_index"], errors="coerce"
        ).fillna(np.inf)
    else:
        work["_array_order"] = np.inf

    work = work.sort_values(
        KEYS + ["_side", "_time_order", "_array_order"],
        kind="mergesort",
    )
    selected = work.drop_duplicates(KEYS, keep="first")
    return selected[required].copy()


def _normalize_sex(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"0", "0.0"}:
        return "female"
    if text in {"1", "1.0"}:
        return "male"
    if text in {"female", "f", "woman", "women"}:
        return "female"
    if text in {"male", "m", "man", "men"}:
        return "male"
    if text in {"", "nan", "none", "-1", "-3"}:
        return np.nan
    return "sex_code_{}".format(text.replace(" ", "_"))


def _normalize_education(value):
    if pd.isna(value):
        return np.nan
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if float(numeric) < 0:
            return np.nan
        if float(numeric).is_integer():
            return "education_code_{}".format(int(numeric))
        return "education_code_{}".format(numeric)
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "missing", "unknown"}:
        return np.nan
    return text


def derive_smoking_status(current, past):
    """Derive current/former/never using the HPP questionnaire codes."""
    current_number = pd.to_numeric(current, errors="coerce")
    past_number = pd.to_numeric(past, errors="coerce")
    output = pd.Series(np.nan, index=current.index, dtype=object)
    output.loc[current_number.isin([1, 2])] = "current"
    not_current = current_number.eq(0)
    output.loc[not_current & past_number.isin([1, 2])] = "former"
    output.loc[not_current & past_number.isin([3, 4])] = "never"
    return output


def derive_physical_activity(lifestyle):
    """Return legacy UK Biobank-style summed activity in MET-hours/week.

    The HPP lifestyle fields mirror the UK Biobank walking/moderate/vigorous
    activity questions.  The legacy UK Biobank derivation handles an incomplete
    activity domain by setting BOTH fields of that domain to zero when the
    other two domains are complete.  If two or more domains are incomplete,
    the total remains missing.

    Daily duration is truncated at 180 min/day before MET weighting.

    Note
    ----
    The reference Diet-Gut-Liver paper labels physical activity as MET-h/day,
    but the replicated distribution matches the legacy UK Biobank quantity
    after converting summed MET-min/week to MET-h/week (divide by 60 only).
    We therefore preserve the mathematically correct unit here.
    """
    columns = {
        "walking": (
            "activity_walking_10min_days_weekly",
            "activity_walking_minutes_daily",
            3.3,
        ),
        "moderate": (
            "activity_moderate_days_weekly",
            "activity_moderate_minutes_daily",
            4.0,
        ),
        "vigorous": (
            "activity_vigorous_days_weekly",
            "activity_vigorous_minutes_daily",
            8.0,
        ),
    }

    work = pd.DataFrame(index=lifestyle.index)

    # Clean raw questionnaire values.
    for _, (days_column, minutes_column, _) in columns.items():
        work[days_column] = _numeric(
            lifestyle[days_column], minimum=0, maximum=7
        )
        work[minutes_column] = _numeric(
            lifestyle[minutes_column], minimum=0, maximum=1440
        )

    pair_complete = {
        name: work[days_column].notna() & work[minutes_column].notna()
        for name, (days_column, minutes_column, _) in columns.items()
    }

    legacy = work.copy()

    # Documented historical UK Biobank rule:
    # if one domain is incomplete while the other two are complete,
    # set both fields of the incomplete domain to zero.
    for name, (days_column, minutes_column, _) in columns.items():
        other_names = [other for other in columns if other != name]
        salvage = (
            ~pair_complete[name]
            & pair_complete[other_names[0]]
            & pair_complete[other_names[1]]
        )
        legacy.loc[salvage, [days_column, minutes_column]] = 0.0

    complete = legacy.notna().all(axis=1)

    met_min_week = pd.Series(0.0, index=lifestyle.index, dtype=float)

    for _, (days_column, minutes_column, met_weight) in columns.items():
        minutes = legacy[minutes_column].clip(upper=180)
        met_min_week = (
            met_min_week
            + met_weight * legacy[days_column] * minutes
        )

    met_h_week = met_min_week / 60.0
    return met_h_week.where(complete)


def _parse_code_cell(value):
    if pd.isna(value):
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = re.split(r"[,;|]", text)
    if isinstance(parsed, (list, tuple, set, np.ndarray)):
        values = list(parsed)
    else:
        values = [parsed]
    output = []
    for item in values:
        if pd.isna(item):
            continue
        code = re.sub(r"[^A-Z0-9]", "", str(item).upper())
        if code:
            output.append(code)
    return output


def aggregate_medications(frame, cohort, research_stage):
    """Aggregate ATC-prefix medication indicators with three-state coverage."""
    require_columns(frame, KEYS + ["atc3", "atc4", "atc5"], "medications")
    work = filter_baseline(frame, "medications", cohort, research_stage)
    columns = KEYS + [
        "vitamin_use",
        "hormone_use",
        "nsaid_aspirin_use",
        "a10_medication_use",
    ]
    rows = []
    for identity, group in work.groupby(KEYS, sort=False):
        codes = []
        for column in ["atc3", "atc4", "atc5"]:
            for value in group[column].tolist():
                codes.extend(_parse_code_cell(value))
        rows.append(
            {
                "participant_id": identity[0],
                "cohort": identity[1],
                "vitamin_use": float(any(code.startswith("A11") for code in codes)),
                "hormone_use": float(
                    any(
                        code.startswith(prefix)
                        for code in codes
                        for prefix in ["G03", "H01", "H02", "H03", "H04", "H05"]
                    )
                ),
                "nsaid_aspirin_use": float(
                    any(
                        code.startswith(prefix)
                        for code in codes
                        for prefix in ["B01", "M01", "N02"]
                    )
                ),
                "a10_medication_use": float(
                    any(code.startswith("A10") for code in codes)
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _family_count_flag(frame, columns):
    """Derive a family-history flag without replacing unknowns with zero."""
    values = pd.concat(
        [_numeric(frame[column], minimum=0) for column in columns], axis=1
    )
    values.columns = columns
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    output.loc[values.gt(0).any(axis=1)] = 1.0
    output.loc[values.notna().all(axis=1) & values.eq(0).all(axis=1)] = 0.0
    return output


def _family_history_diabetes_from_parent(series):
    """Derive diabetes family history from the HPP parent multi-select question.

    The current HPP dataset stores the parent response as strings such as
    ``[1 2 4 10]`` rather than comma-separated Python lists.  Audit against the
    child follow-up questions showed that parent code 3 corresponds to type 1
    diabetes and code 4 corresponds to type 2 diabetes.

    Rules:
      * parent question missing -> missing
      * answered and contains code 3 or 4 -> 1
      * answered and contains neither code 3 nor 4 -> 0
    """
    output = pd.Series(np.nan, index=series.index, dtype=float)

    for index, value in series.items():
        if pd.isna(value):
            continue

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "missing"}:
            continue

        codes = {
            int(match)
            for match in re.findall(r"-?\d+", text)
        }

        # An explicitly answered parent item with no diabetes code is a No.
        output.loc[index] = 1.0 if (3 in codes or 4 in codes) else 0.0

    return output


def _binary_flag(series):
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[numeric.eq(0)] = 0.0
    output.loc[numeric.gt(0)] = 1.0
    text = series.astype(str).str.strip().str.lower()
    output.loc[text.isin(["no", "false", "n"])] = 0.0
    output.loc[text.isin(["yes", "true", "y"])] = 1.0
    return output


def _is_known_diabetes(row):
    code = re.sub(r"[^A-Z0-9]", "", str(row.get("icd11_code", "")).upper())
    condition = str(row.get("medical_condition", "")).strip().lower()
    if "gestational" in condition:
        return False
    if code.startswith("5A1"):
        return True
    return bool(
        re.search(
            r"(type\s*[12].*diabet|diabet.*type\s*[12]|diabetes\s+mellitus)",
            condition,
        )
    )


def aggregate_conditions(frame, cohort, research_stage):
    """Aggregate known type 1/type 2 diabetes while preserving source absence."""
    require_columns(
        frame,
        KEYS + ["medical_condition", "icd11_code"],
        "medical_conditions",
    )
    work = filter_baseline(frame, "medical_conditions", cohort, research_stage)
    rows = []
    for identity, group in work.groupby(KEYS, sort=False):
        rows.append(
            {
                "participant_id": identity[0],
                "cohort": identity[1],
                "known_diabetes": float(
                    any(_is_known_diabetes(row) for _, row in group.iterrows())
                ),
            }
        )
    return pd.DataFrame(rows, columns=KEYS + ["known_diabetes"])


def prepare_alcohol(frame, cohort, research_stage):
    require_columns(
        frame,
        KEYS + ["alcohol_complete", "mean_daily_alcohol_g"],
        "AMED alcohol source",
    )
    work = filter_baseline(frame, "AMED alcohol source", cohort, research_stage)
    if work.duplicated(KEYS).any():
        raise ValueError("AMED alcohol source contains duplicate participant rows")
    complete_text = work["alcohol_complete"].astype(str).str.strip().str.lower()
    complete = complete_text.isin(["true", "1", "yes", "y"])
    alcohol = _numeric(work["mean_daily_alcohol_g"], minimum=0)
    work["alcohol_intake_g_day"] = alcohol.where(complete)
    return work[KEYS + ["alcohol_intake_g_day"]].copy()


def _derive_age(master):
    month = _numeric(master["month_of_birth_raw"], minimum=1, maximum=12)
    year = _numeric(master["year_of_birth_raw"], minimum=1900, maximum=2100)
    birth = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 15}),
        errors="coerce",
        utc=True,
    )
    reference = _parse_datetime(master["reference_date"])
    age = (reference - birth).dt.total_seconds() / (365.2425 * 24 * 60 * 60)
    return age.where(age.between(18, 110))


def _three_state_union(first, second):
    output = pd.Series(np.nan, index=first.index, dtype=float)
    output.loc[first.eq(1) | second.eq(1)] = 1.0
    output.loc[first.eq(0) & second.eq(0)] = 0.0
    return output


def _source_coverage_row(label, input_frame, eligible_frame, derived_frame, master):
    input_normalized = normalize_identity(input_frame, label)
    eligible_normalized = normalize_identity(eligible_frame, label)
    derived_keys = derived_frame[KEYS].drop_duplicates() if not derived_frame.empty else derived_frame
    matched = master[KEYS].merge(derived_keys, on=KEYS, how="inner")
    return {
        "source": label,
        "input_rows": int(len(input_frame)),
        "eligible_rows": int(len(eligible_frame)),
        "input_unique_participants": int(input_normalized["participant_id"].nunique()),
        "eligible_unique_participants": int(eligible_normalized["participant_id"].nunique()),
        "derived_unique_participants": int(len(derived_keys)),
        "matched_master_participants": int(len(matched)),
    }


def _build_reports(master, source_rows):
    completeness_rows = []
    for column in CORE_COVARIATES + EXCLUSION_COLUMNS + MODEL_FLAG_COLUMNS:
        if column in MODEL_FLAG_COLUMNS:
            nonmissing = int(master[column].eq(True).sum())
            missing = int(len(master) - nonmissing)
        else:
            nonmissing = int(master[column].notna().sum())
            missing = int(master[column].isna().sum())
        completeness_rows.append(
            {
                "variable": column,
                "participant_count": int(len(master)),
                "nonmissing_or_complete_count": nonmissing,
                "missing_or_incomplete_count": missing,
                "nonmissing_or_complete_fraction": (
                    float(nonmissing) / len(master) if len(master) else np.nan
                ),
            }
        )

    categorical_columns = [
        "sex",
        "education_level",
        "smoking_status",
        "cgm_device_type",
        "vitamin_use",
        "hormone_use",
        "nsaid_aspirin_use",
        "a10_medication_use",
        "family_history_diabetes",
        "family_history_cvd",
        "known_diabetes",
        "exclude_diabetes_or_a10",
    ] + MODEL_FLAG_COLUMNS
    categorical_rows = []
    for column in categorical_columns:
        values = master[column].astype(object).where(master[column].notna(), "<missing>")
        counts = values.value_counts(dropna=False)
        for value, count in counts.items():
            categorical_rows.append(
                {
                    "variable": column,
                    "value": str(value),
                    "count": int(count),
                    "fraction": float(count) / len(master) if len(master) else np.nan,
                }
            )

    return {
        "completeness": pd.DataFrame(completeness_rows),
        "source_coverage": pd.DataFrame(source_rows),
        "categorical_values": pd.DataFrame(categorical_rows),
    }


def build_covariate_master(
    population,
    cgm,
    sociodemographics,
    lifestyle,
    medications,
    anthropometrics,
    family_history,
    medical_conditions,
    alcohol,
    cohort=DEFAULT_COHORT,
    research_stage=DEFAULT_RESEARCH_STAGE,
):
    """Return the covariate master, audit reports, and scalar summary."""
    require_columns(
        population,
        KEYS + ["month_of_birth", "year_of_birth", "sex"],
        "population",
    )
    population_clean = normalize_identity(population, "population")
    population_clean = population_clean.loc[
        population_clean["cohort"].eq(str(cohort))
    ].copy()
    if population_clean.duplicated(KEYS).any():
        raise ValueError("population contains duplicate participant rows")
    population_clean = population_clean.sort_values(KEYS, kind="mergesort")

    master = population_clean[
        KEYS + ["month_of_birth", "year_of_birth", "sex"]
    ].rename(
        columns={
            "month_of_birth": "month_of_birth_raw",
            "year_of_birth": "year_of_birth_raw",
            "sex": "sex_raw",
        }
    )

    require_columns(
        cgm,
        KEYS
        + [
            "research_stage",
            "connection_id",
            "collection_timestamp",
            "cgm_device_type",
        ],
        "baseline CGM",
    )
    cgm_clean = filter_baseline(cgm, "baseline CGM", cohort, research_stage)
    if cgm_clean.duplicated(KEYS).any():
        raise ValueError("baseline CGM contains duplicate participant rows")
    cgm_piece = cgm_clean[
        KEYS
        + [
            "research_stage",
            "connection_id",
            "collection_timestamp",
            "cgm_device_type",
        ]
    ].rename(
        columns={
            "research_stage": "cgm_research_stage",
            "collection_timestamp": "cgm_collection_timestamp",
        }
    )
    cgm_piece["reference_date"] = _parse_datetime(
        cgm_piece["cgm_collection_timestamp"]
    )
    cgm_piece["reference_date_source"] = "baseline_cgm"
    cgm_piece["has_baseline_cgm"] = True
    master = master.merge(cgm_piece, on=KEYS, how="left", validate="one_to_one")
    master["has_baseline_cgm"] = master["has_baseline_cgm"].fillna(False).astype(bool)

    initial_reference = master[KEYS + ["reference_date"]].copy()

    socio_fields = ["education"]
    socio_selected = select_baseline_row(
        sociodemographics,
        initial_reference,
        socio_fields,
        "sociodemographics",
        cohort,
        research_stage,
    )
    socio_piece = socio_selected[KEYS + ["education", "collection_timestamp"]].rename(
        columns={
            "education": "education_raw",
            "collection_timestamp": "sociodemographics_collection_timestamp",
        }
    )

    lifestyle_fields = [
        "smoking_current_status",
        "smoking_past_frequency",
        "sleep_hours_daily",
        "activity_walking_10min_days_weekly",
        "activity_walking_minutes_daily",
        "activity_moderate_days_weekly",
        "activity_moderate_minutes_daily",
        "activity_vigorous_days_weekly",
        "activity_vigorous_minutes_daily",
    ]
    lifestyle_selected = select_baseline_row(
        lifestyle,
        initial_reference,
        lifestyle_fields,
        "lifestyle",
        cohort,
        research_stage,
    )
    lifestyle_piece = lifestyle_selected[KEYS + lifestyle_fields + ["collection_timestamp"]].copy()
    lifestyle_piece = lifestyle_piece.rename(
        columns={"collection_timestamp": "lifestyle_collection_timestamp"}
    )
    lifestyle_piece["smoking_status"] = derive_smoking_status(
        lifestyle_piece["smoking_current_status"],
        lifestyle_piece["smoking_past_frequency"],
    )
    lifestyle_piece["sleep_duration_hours_day"] = _numeric(
        lifestyle_piece["sleep_hours_daily"], minimum=0.01, maximum=24
    )
    lifestyle_piece["physical_activity_met_h_week"] = derive_physical_activity(
        lifestyle_piece
    )

    anthropometric_fields = ["bmi"]
    anthropometric_selected = select_baseline_row(
        anthropometrics,
        initial_reference,
        anthropometric_fields,
        "anthropometrics",
        cohort,
        research_stage,
    )
    anthropometric_piece = anthropometric_selected[
        KEYS + ["bmi", "collection_timestamp"]
    ].rename(columns={"collection_timestamp": "anthropometrics_collection_timestamp"})
    anthropometric_piece["bmi"] = _numeric(
        anthropometric_piece["bmi"], minimum=10, maximum=80
    )

    family_fields = [
        "family_history",
        "type_1_diabetes_family_number",
        "type_2_diabetes_family_number",
        "sudden_death_family",
    ]
    family_selected = select_baseline_row(
        family_history,
        initial_reference,
        family_fields,
        "family_history",
        cohort,
        research_stage,
    )
    family_piece = family_selected[KEYS + family_fields + ["collection_timestamp"]].copy()
    family_piece = family_piece.rename(
        columns={"collection_timestamp": "family_history_collection_timestamp"}
    )
    family_piece["family_history_diabetes"] = _family_history_diabetes_from_parent(
        family_piece["family_history"]
    )
    family_piece["family_history_cvd"] = _binary_flag(
        family_piece["sudden_death_family"]
    )

    medication_piece = aggregate_medications(medications, cohort, research_stage)
    condition_piece = aggregate_conditions(medical_conditions, cohort, research_stage)
    alcohol_piece = prepare_alcohol(alcohol, cohort, research_stage)

    pieces = [
        socio_piece,
        lifestyle_piece,
        anthropometric_piece,
        family_piece,
        medication_piece,
        condition_piece,
        alcohol_piece,
    ]
    for piece in pieces:
        master = master.merge(piece, on=KEYS, how="left", validate="one_to_one")

    fallback_sources = [
        ("sociodemographics", "sociodemographics_collection_timestamp"),
        ("lifestyle", "lifestyle_collection_timestamp"),
        ("anthropometrics", "anthropometrics_collection_timestamp"),
        ("family_history", "family_history_collection_timestamp"),
    ]
    fallback_frames = []
    for priority, (source, column) in enumerate(fallback_sources):
        candidate = master[KEYS + [column]].rename(columns={column: "fallback_date"})
        candidate["fallback_date"] = _parse_datetime(candidate["fallback_date"])
        candidate["fallback_source"] = source
        candidate["fallback_priority"] = priority
        fallback_frames.append(candidate.loc[candidate["fallback_date"].notna()])
    if fallback_frames:
        fallback = pd.concat(fallback_frames, ignore_index=True)
        fallback = fallback.sort_values(
            KEYS + ["fallback_date", "fallback_priority"], kind="mergesort"
        ).drop_duplicates(KEYS)
        master = master.merge(
            fallback[KEYS + ["fallback_date", "fallback_source"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        missing_reference = master["reference_date"].isna()
        master.loc[missing_reference, "reference_date"] = master.loc[
            missing_reference, "fallback_date"
        ]
        master.loc[missing_reference, "reference_date_source"] = master.loc[
            missing_reference, "fallback_source"
        ]
        master = master.drop(columns=["fallback_date", "fallback_source"])

    master["age_years"] = _derive_age(master)
    master["sex"] = master["sex_raw"].map(_normalize_sex)
    master["education_level"] = master["education_raw"].map(_normalize_education)
    master["exclude_diabetes_or_a10"] = _three_state_union(
        master["known_diabetes"], master["a10_medication_use"]
    )

    model_definitions = {
        "amed_model2_covariates_complete": MODEL2_COMMON,
        "hpdi_model2_covariates_complete": MODEL2_COMMON + ["alcohol_intake_g_day"],
        "amed_model3_covariates_complete": MODEL2_COMMON + ["bmi"],
        "hpdi_model3_covariates_complete": MODEL2_COMMON + ["bmi", "alcohol_intake_g_day"],
        "amed_model4_covariates_complete": MODEL2_COMMON
        + ["nsaid_aspirin_use", "family_history_diabetes", "family_history_cvd"],
        "hpdi_model4_covariates_complete": MODEL2_COMMON
        + [
            "nsaid_aspirin_use",
            "family_history_diabetes",
            "family_history_cvd",
            "alcohol_intake_g_day",
        ],
    }
    for flag, columns in model_definitions.items():
        master[flag] = master[columns].notna().all(axis=1).astype(bool)

    source_rows = []
    source_specs = [
        ("population", population, population_clean, master[KEYS]),
        ("selected_baseline_cgm", cgm, cgm_clean, cgm_piece),
        (
            "sociodemographics",
            sociodemographics,
            filter_baseline(sociodemographics, "sociodemographics", cohort, research_stage),
            socio_selected,
        ),
        (
            "lifestyle",
            lifestyle,
            filter_baseline(lifestyle, "lifestyle", cohort, research_stage),
            lifestyle_selected,
        ),
        (
            "medications",
            medications,
            filter_baseline(medications, "medications", cohort, research_stage),
            medication_piece,
        ),
        (
            "anthropometrics",
            anthropometrics,
            filter_baseline(anthropometrics, "anthropometrics", cohort, research_stage),
            anthropometric_selected,
        ),
        (
            "family_history",
            family_history,
            filter_baseline(family_history, "family_history", cohort, research_stage),
            family_selected,
        ),
        (
            "medical_conditions",
            medical_conditions,
            filter_baseline(medical_conditions, "medical_conditions", cohort, research_stage),
            condition_piece,
        ),
        (
            "alcohol",
            alcohol,
            filter_baseline(alcohol, "alcohol", cohort, research_stage),
            alcohol_piece,
        ),
    ]
    for label, input_frame, eligible_frame, derived_frame in source_specs:
        source_rows.append(
            _source_coverage_row(label, input_frame, eligible_frame, derived_frame, master)
        )

    preferred_order = (
        KEYS
        + [
            "reference_date",
            "reference_date_source",
            "has_baseline_cgm",
            "cgm_research_stage",
            "connection_id",
            "cgm_collection_timestamp",
            "cgm_device_type",
        ]
        + CORE_COVARIATES[:8]
        + CORE_COVARIATES[9:]
        + EXCLUSION_COLUMNS
        + MODEL_FLAG_COLUMNS
    )
    preferred_order = list(dict.fromkeys(preferred_order))
    remaining = [column for column in master.columns if column not in preferred_order]
    master = master[preferred_order + remaining].sort_values(KEYS, kind="mergesort")
    master = master.reset_index(drop=True)

    reports = _build_reports(master, source_rows)
    summary = {
        "participants_in_master": int(len(master)),
        "participants_with_baseline_cgm": int(master["has_baseline_cgm"].sum()),
        "participants_known_diabetes": int(master["known_diabetes"].eq(1).sum()),
        "participants_a10_medication_use": int(master["a10_medication_use"].eq(1).sum()),
        "participants_exclude_diabetes_or_a10": int(
            master["exclude_diabetes_or_a10"].eq(1).sum()
        ),
    }
    for flag in MODEL_FLAG_COLUMNS:
        summary[flag] = int(master[flag].sum())
    return master, reports, summary


def atomic_write_csv(frame, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target_path.name),
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(str(temporary_path), index=False)
        os.replace(str(temporary_path), str(target_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_outputs(master, reports, summary, output_dir):
    """Atomically write the master table and four audit reports."""
    output_dir = Path(output_dir)
    paths = {
        "master": output_dir / "data" / "02_covariate_master.csv",
        "completeness": output_dir / "reports" / "02_covariate_completeness.csv",
        "source_coverage": output_dir / "reports" / "02_source_coverage.csv",
        "categorical_values": output_dir / "reports" / "02_categorical_values.csv",
        "summary": output_dir / "reports" / "02_build_summary.csv",
    }
    atomic_write_csv(master, paths["master"])
    atomic_write_csv(reports["completeness"], paths["completeness"])
    atomic_write_csv(reports["source_coverage"], paths["source_coverage"])
    atomic_write_csv(reports["categorical_values"], paths["categorical_values"])
    summary_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    atomic_write_csv(summary_frame, paths["summary"])
    return {key: str(path) for key, path in paths.items()}


def read_csv(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(label, path))
    return pd.read_csv(
        str(path),
        dtype={
            "participant_id": str,
            "cohort": str,
            "research_stage": str,
            "connection_id": str,
        },
        low_memory=False,
    )


def parse_args(argv=None):
    script_path = Path(__file__).resolve()
    default_project_root = script_path.parents[1]
    default_hpp_root = default_project_root.parent.parent
    parser = argparse.ArgumentParser(
        description="Build one-row-per-participant HPP covariate master table."
    )
    parser.add_argument("--project-root", default=str(default_project_root))
    parser.add_argument("--csv-dir", default=str(default_project_root / "csv"))
    parser.add_argument(
        "--baseline-cgm-csv",
        default=str(
            default_hpp_root
            / "Data"
            / "cgm_deal"
            / "outputs"
            / "data"
            / "01_baseline_cgm_connections.csv"
        ),
    )
    parser.add_argument(
        "--alcohol-csv",
        default=str(
            default_hpp_root
            / "Data"
            / "diet_deal"
            / "outputs"
            / "05_diet_scores"
            / "amed"
            / "amed_participant_scores.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(default_project_root / "outputs")
    )
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_dir = Path(args.csv_dir)
    inputs = {
        "population": read_csv(csv_dir / "population.csv", "population CSV"),
        "cgm": read_csv(args.baseline_cgm_csv, "selected baseline CGM CSV"),
        "sociodemographics": read_csv(
            csv_dir / "sociodemographics_initial_medical.csv",
            "sociodemographics CSV",
        ),
        "lifestyle": read_csv(
            csv_dir / "lifestyle_and_environment.csv", "lifestyle CSV"
        ),
        "medications": read_csv(csv_dir / "medications.csv", "medications CSV"),
        "anthropometrics": read_csv(
            csv_dir / "anthropometrics.csv", "anthropometrics CSV"
        ),
        "family_history": read_csv(
            csv_dir / "family_history_initial_medical.csv", "family history CSV"
        ),
        "medical_conditions": read_csv(
            csv_dir / "medical_conditions.csv", "medical conditions CSV"
        ),
        "alcohol": read_csv(args.alcohol_csv, "AMED participant score CSV"),
    }

    print("=== 02 PARTICIPANT COVARIATE MASTER ===")
    print("PROJECT_ROOT={}".format(Path(args.project_root)))
    print("CSV_DIR={}".format(csv_dir))
    print("BASELINE_CGM_CSV={}".format(args.baseline_cgm_csv))
    print("ALCOHOL_CSV={}".format(args.alcohol_csv))
    print("OUTPUT_DIR={}".format(args.output_dir))

    master, reports, summary = build_covariate_master(
        cohort=args.cohort,
        research_stage=args.research_stage,
        **inputs
    )
    paths = write_outputs(master, reports, summary, args.output_dir)

    for key, value in summary.items():
        print("{}={}".format(key.upper(), value))
    print("MASTER_FILE={}".format(paths["master"]))
    print("COMPLETENESS_REPORT={}".format(paths["completeness"]))
    print("SOURCE_COVERAGE_REPORT={}".format(paths["source_coverage"]))
    print("CATEGORICAL_VALUES_REPORT={}".format(paths["categorical_values"]))
    print("SUMMARY_REPORT={}".format(paths["summary"]))
    print("COVARIATE_MASTER_COMPLETED=True")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR={}".format(error), file=sys.stderr)
        raise
