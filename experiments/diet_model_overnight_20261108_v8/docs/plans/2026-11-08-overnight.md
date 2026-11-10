# V8 overnight modeling implementation plan

**Goal:** Prepare a broader, reproducible discovery-only batch for manual server execution, with no promised R2 improvement.

**Architecture:** Keep the frozen V4 preprocessing and V6 data/checkpoint adapter. Build training-role representations and model families in the new V8 directory. Select settings and fusion weights inside household inner CV; use old V6/V7 predictions only after fitting for evaluation.

**Tech stack:** Python 3.11, pinned NumPy/SciPy/pandas/Arrow/scikit-learn/CatBoost/LightGBM, unittest and Ruff. All server code/output is ASCII. No server data are copied locally.

## 1. Frozen question and comparisons

- Primary: participant-level fiber_density -> D_Glu, mg/dL, original eligibility and 2 x 5 household outer CV / 3 household inner folds.
- Keep native outcomes and rows unchanged. UPF-background pipelines add only the two documented dietary bounds, imputed from training rows; report separately from original-input models.
- Include matched main-effect nulls for sparse and NMF diet-by-microbiome corrections. Report pooled OOF R2 and matched delta_R2, both repeats, negative results and warnings.
- Existing V6 fusion and V7 ridge stack are read-only evaluation comparators. They never enter a training design or meta learner.

## 2. Implement training-role feature methods

Files: features.py, tests/test_features.py.

Test frozen projections before implementing NMF co-abundance factors, CLR-PC plus screened-taxon hybrid features, host-residual PLS factors, interaction-aware screening and documented UPF bounds. Fit and transform training rows through identical operations. Fit no scaler, selector or factorization on validation rows.

## 3. Implement model families

Files: estimators.py, models.py, registry.py, config.json, tests/test_models.py.

Prioritize existing strong HGB/CatBoost anchors and new NMF, hybrid and PLS tree learners. Add supervised leaf-rule regression, piecewise-linear boosting, robust-loss CatBoost, random rotations, host-residual neighborhood smoothing, guarded FM and additive main-effect models. Sparse group corrections use a convex proximal optimizer with a computable stopping residual; FM reports optimizer success and gradient norm. Do not call the sparse correction a hierarchical-lasso implementation.

## 4. Implement honest nested evaluation and recovery

Files: bridge.py, engine.py, runtime.py, run.py, launch.py, report.py, check.py, setup.sh.

Reuse close-before-rename/checksum checkpoints. Register all source/input/reference versions, reject changed resume identities, retain crash locks, print progress to the terminal, and suppress pooled scores for partial batches. Fit all meta models from the current outer-training inner OOF predictions. Compare no-FM and with-FM fusion with the same core members. Propagate contributing fit warnings to meta results.

## 5. Verify and deliver

Files: tests/, README.md, summary.md, research_records/validation.json, upload_manifest.json.

- Run all configured atomic cases on sufficiently wide synthetic profiles; prediction and serialization must round-trip.
- Check the sparse proximal objective and null/zero correction identities, FM gradients, native-unit contrasts and known synthetic heterogeneity.
- Mutate held-out outcomes/features and verify choices, transformations and fusion weights remain training-only.
- Run the actual runtime in a temporary synthetic handoff, stop/resume, re-read reports and detect corrupted checkpoints.
- Check Python syntax, Ruff, shell syntax, ASCII executable files, dependency pins and upload file hashes.
- Verify old V6/V7 source hashes stay unchanged. Publish a detailed Chinese local summary and short English server reports. Actual research performance remains pending until the user runs the uploaded batch.
