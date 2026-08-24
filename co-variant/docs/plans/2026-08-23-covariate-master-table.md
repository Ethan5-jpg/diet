# HPP Participant Covariate Master Table Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build one participant-level HPP table containing the 14 diet-CGM covariates, CGM baseline identity, diabetes/A10 exclusion indicators, and model-specific completeness flags without deleting or imputing participants.

**Architecture:** Use `population.csv` as the participant universe for the 10k cohort. Select baseline questionnaire or measurement rows deterministically, aggregate repeated medication and diagnosis rows, derive the paper-defined covariates, and left-join every result to the universe. Use the already selected baseline CGM connection from `cgm_deal` so the device and date match the CGM phenotype pipeline.

**Tech Stack:** Python 3.7-compatible pandas/NumPy, Bash, unittest with synthetic CSV fixtures.

---

### Task 1: Add derivation tests

**Files:**
- Create: `Data/co-variant/tests/test_build_covariate_master.py`

**Step 1:** Create synthetic population, baseline CGM, questionnaire, medication, anthropometric, family-history, condition, and alcohol inputs.

**Step 2:** Test age, smoking, physical-activity, ATC-prefix, family-history, diabetes, missing-data, and model-completeness derivations.

**Step 3:** Run the new test and confirm it fails before the implementation exists.

### Task 2: Implement participant-level derivation

**Files:**
- Create: `Data/co-variant/scripts/02_build_covariate_master.py`

**Step 1:** Add validated CSV loading, ID normalization, and deterministic baseline-row selection.

**Step 2:** Derive the 14 covariates using the paper definitions, including IPAQ-style MET-h/day and ATC-prefix medication flags.

**Step 3:** Derive A10 and known-diabetes exclusion indicators with three-state missingness rather than treating absent source records as negative.

**Step 4:** Left-join all results to the population universe and add Model 2, Model 3, and Model 4 completeness flags for AMED and hPDI.

**Step 5:** Atomically write the master table plus source-coverage, completeness, and categorical-distribution reports.

### Task 3: Add the server runner and documentation

**Files:**
- Create: `Data/co-variant/run_build_covariate_master.sh`
- Modify: `Data/co-variant/README.md`

**Step 1:** Add server defaults for the eight converted CSVs, selected baseline CGM connection, AMED alcohol source, and outputs directory.

**Step 2:** Document the output columns, the fact that no participant is removed, and the terminal success markers.

### Task 4: Verify compatibility and behavior

**Files:**
- Test: `Data/co-variant/tests/test_build_covariate_master.py`
- Test: `Data/co-variant/tests/test_prepare_covariate_sources.py`

**Step 1:** Run both unittest modules in the local `pheno` environment.

**Step 2:** Parse the Python source using the Python 3.7 grammar and run `bash -n` on both runners.

**Step 3:** Inspect produced synthetic outputs for one row per participant, preserved missingness, and expected completeness counts.
