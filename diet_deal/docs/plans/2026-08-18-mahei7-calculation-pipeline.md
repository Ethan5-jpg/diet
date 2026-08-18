# HPP mAHEI-7 Calculation Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible Python 3.7-compatible pipeline that converts HPP baseline diet logs into the approved seven-component modified AHEI score, with explicit mapping, completeness, energy-adjustment, and QC outputs.

**Architecture:** Reuse the already-reviewed hPDI food mapping for fruit, vegetables, whole grains, nuts, legumes, fruit juice, and sugar-sweetened beverages, and the AMED mapping for red/processed meat. Aggregate event-level energy or weight proxy inputs over the preserved participant-day index, reuse AMED's resolved alcohol and total-energy fields, then calculate component scores and the common winsorization/residual-adjustment/Z-score/quintile post-processing. Never treat excluded sodium, trans fat, or PUFA as zero and never present the result as the complete 11-component AHEI.

**Tech Stack:** Python 3.7-compatible syntax, pandas, unittest, CSV outputs.

---

### Task 1: Restore the read-only input audit

**Files:**
- Create: `Data/diet_deal/scripts/ahei/00_audit_ahei_inputs.py`
- Create: `Data/diet_deal/scripts/ahei/00b_scan_ahei_lookup_sources.py`
- Test: `Data/diet_deal/tests/ahei/test_audit_ahei_inputs.py`

**Steps:**
1. Implement the frozen `modified_AHEI_2010_7_component` constants and protocol metadata.
2. Audit event-field coverage after excluding only all-label-missing food IDs while preserving participants and original logging days.
3. Audit the AMED shared alcohol/energy table and population sex coverage.
4. Keep lookup scanning read-only and mark whole-grain strict values as optional and the three excluded nutrients as non-blocking.
5. Run `python -m unittest Data.diet_deal.tests.ahei.test_audit_ahei_inputs -v`; expect all tests to pass.

### Task 2: Build the AHEI food mapping

**Files:**
- Create: `Data/diet_deal/scripts/ahei/03_prepare_ahei_mapping.py`
- Create: `Data/diet_deal/tests/ahei/test_prepare_ahei_mapping.py`

**Steps:**
1. Write tests for the exact hPDI/AMED-to-AHEI component translations, fruit-juice separation, red-meat source, and missing-label handling.
2. Join the food dictionary one-to-one to the reviewed hPDI and AMED mappings.
3. Map hPDI `fruits`, `vegetables`, `whole_grains`, `nuts`, `legumes`, `fruit_juice`, and `sugar_sweetened_beverages`; map AMED `red_processed_meat`.
4. Combine nuts and legumes and combine fruit juice and sugar-sweetened beverages.
5. Emit `ahei_component_definitions.csv`, `ahei_food_id_mapping.csv`, and `ahei_mapping_qc.csv`.

### Task 3: Build daily and participant mAHEI-7 inputs

**Files:**
- Create: `Data/diet_deal/scripts/ahei/04_prepare_ahei_score_intake.py`
- Create: `Data/diet_deal/tests/ahei/test_prepare_ahei_score_intake.py`

**Steps:**
1. Write tests that preserve zero-intake days, exclude all-label-missing events without dropping their participant-day keys, and reject unresolved mapped weights.
2. Apply approved hPDI event-level weight overrides where available.
3. Resolve missing/invalid event energy with the approved lower-bound Atwater rule used by AMED.
4. Aggregate vegetable, fruit, SSB/fruit-juice, and red/processed-meat energy; aggregate whole-grain and nuts/legumes weight.
5. Convert energy proxies to servings with 30, 70, 70, and 220 kcal/serving, respectively; use 28.35 g/serving for nuts/legumes and retain sex-specific whole-grain grams for scoring.
6. Merge AMED's complete alcohol (g/day) and energy (kcal/day) fields and emit daily, participant, and QC CSVs.

### Task 4: Calculate raw and analysis-ready scores

**Files:**
- Create: `Data/diet_deal/scripts/ahei/05_calculate_ahei_scores.py`
- Create: `Data/diet_deal/tests/ahei/test_calculate_ahei_scores.py`

**Steps:**
1. Write endpoint and interpolation tests for all seven components, including sex-specific whole grains and alcohol.
2. Score beneficial components linearly upward and adverse components linearly downward, clipping each to 0--10.
3. Require all seven components and valid sex for `mAHEI7_raw_0_70`; keep excluded components out of numerator and denominator.
4. Winsorize the raw score at P0.5/P99.5, residual-adjust for mean daily energy, calculate sample-SD Z scores, and assign tie-safe quintiles.
5. Emit component scores, participant scores, adjustment parameters, and QC outputs.

### Task 5: Verify and document the complete pipeline

**Files:**
- Modify: `Data/diet_deal/scripts/README.md`

**Steps:**
1. Run all AHEI unit tests and then all existing diet-deal tests.
2. Run `py_compile` on the five AHEI scripts.
3. Parse every AHEI script with `ast.parse(..., feature_version=(3, 7))` and reject Python 3.9+ built-in generic annotations.
4. Run mapping, intake, and scoring end-to-end on the local snapshot.
5. Inspect output schemas, component ranges, raw 0--70 bounds, Z-score moments, and QC blocker rows.
6. Provide copy-paste server commands rooted at `/home/ec2-user/Desktop/HPP/Data`.

**Repository note:** `/Users/blizzardxu/Desktop/HPP` is not a Git worktree, so this implementation cannot create a dedicated worktree or make the plan's usual incremental commits.
