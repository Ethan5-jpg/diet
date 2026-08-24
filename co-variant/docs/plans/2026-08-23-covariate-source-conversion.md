# Covariate Source Conversion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Copy the eight required HPP parquet sources, including a fresh CGM copy, into `co-variant/raw` and convert each to a CSV in `co-variant/csv` without losing parquet index fields.

**Architecture:** A Python 3.7-compatible command-line program uses a fixed manifest of approved source files. It validates every source before writing anything, atomically refreshes only the eight managed raw and CSV targets, restores pandas index levels as ordinary CSV columns, and verifies output row counts and required identity fields. A small shell launcher supplies the server defaults.

**Tech Stack:** Python 3.7, pandas, pyarrow, pathlib, unittest, Bash.

---

### Task 1: Define and test the conversion manifest

**Files:**
- Create: `Data/co-variant/tests/test_prepare_covariate_sources.py`
- Create: `Data/co-variant/scripts/01_prepare_covariate_sources.py`

**Step 1:** Write a test fixture containing eight small parquet files with the same index layouts as the HPP files.

**Step 2:** Run the test and confirm it fails because the conversion module does not yet exist.

**Step 3:** Add a Python 3.7-compatible manifest covering population, sociodemographics, lifestyle, medications, anthropometrics, family history, medical conditions, and CGM.

**Step 4:** Run the manifest test and confirm all eight expected specifications are present.

### Task 2: Copy sources and recover parquet indexes

**Files:**
- Modify: `Data/co-variant/scripts/01_prepare_covariate_sources.py`
- Modify: `Data/co-variant/tests/test_prepare_covariate_sources.py`

**Step 1:** Test that all source paths are checked before any output is written.

**Step 2:** Implement preflight validation and atomic copies into source-specific subdirectories below `raw`.

**Step 3:** Test that source and raw SHA-256 checksums match.

**Step 4:** Implement parquet loading and `reset_index()` recovery for named index levels.

### Task 3: Convert and verify CSV outputs

**Files:**
- Modify: `Data/co-variant/scripts/01_prepare_covariate_sources.py`
- Modify: `Data/co-variant/tests/test_prepare_covariate_sources.py`

**Step 1:** Test that CSV row counts match parquet row counts and that `participant_id`, `cohort`, and source-specific index fields are present.

**Step 2:** Implement atomic CSV output with `index=False` after index recovery.

**Step 3:** Test list-valued medication columns and deterministic reprocessing of existing managed outputs.

**Step 4:** Run the complete unittest suite.

### Task 4: Add the server launcher and handoff documentation

**Files:**
- Create: `Data/co-variant/run_prepare_covariate_sources.sh`
- Create: `Data/co-variant/README.md`

**Step 1:** Add a launcher that uses the server defaults while allowing explicit path overrides.

**Step 2:** Document the eight inputs, raw and CSV outputs, exclusions, execution command, and expected completion message.

**Step 3:** Run shell syntax validation, Python compilation, and the complete tests.

