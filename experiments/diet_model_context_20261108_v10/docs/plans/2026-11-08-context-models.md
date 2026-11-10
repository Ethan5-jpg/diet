# V10 context models implementation plan

**Goal:** Test whether newly audited macronutrient and lifestyle information improves discovery prediction, then measure matched fiber-by-microbiome gains.

**Architecture:** Keep the frozen V6 discovery loader and V8 role projections. Build an isolated V10 package with explicit filtered source reads, person-level outcome-free aggregation, training-role context encoding, nested household selection and current-inner-OOF fusion. Historical predictions are evaluation-only.

**Tech stack:** Python 3.11, the existing seven pinned V8 packages, NumPy/SciPy/pandas/PyArrow/scikit-learn/CatBoost/LightGBM. No remote data or pretrained weights.

## Design decisions

- Target remains fiber_density -> D_Glu (mg/dL), the original two repeats of household five-fold CV and three inner folds. No new-feature complete-case selection.
- Primary input ladder: original, macros, macros+sodium, lifestyle only, macros+sodium+lifestyle. Five independent sensitivity blocks: macro+lifestyle without sodium, joint-macro days, fiber-valid macro days, nutrient distributions, and macro log-ratios.
- Read complete nutrient day sums, never event partial sums. Validate current-day energy and fiber anchors. Filter composite discovery identities before materializing mixed-role registries. Unknown categorical values stay missing/unknown.
- Use matched HGB, CatBoost, block-penalized ridge and host-only controls across the input ladder. Add PCA/splines, RBF SVR, extra trees, additive kernels and zero-aware microbial representations on the enriched background.
- Explicit corrections use a separately stored train-fitted oral/gut PC or nutrient modifier, not every context column. Include an exact-zero candidate. Distinguish additive-ridge nulls from tree backgrounds that already allow implicit interactions.
- Fusion fits only current inner OOF predictions; retrain every member. No previous outer predictions as training features.
- Do not use alcohol grams, sparse vigorous activity, microbiome collection lags, BMI or other metabolic outcomes as default predictors.

## Implementation steps

1. Copy only reusable runtime/engine/checkpoint modules into this new folder; preserve prior sources and record required imported hashes.
2. Implement nutrient and lifestyle ingestion with projected filtered reads, strict keys/QC anchors, missing-preserving aggregation and audit counts. Test bad flags, duplicate days, anchor conflicts, cross-role identities, sparse people and zero values.
3. Implement replayable training-role feature encoding and block partitions. Test held-out changes cannot alter imputation, categories, screening or projections; test serialized replay.
4. Implement block ridge, spline ridge, SVR, extra trees and additive kernels alongside the established tree learners. Test kernel/primal equivalence, all default candidates, finite predictions, zero corrections and native-unit fiber contrasts.
5. Freeze the registry and paired input comparisons; implement current OOF convex/ridge pools, compact aggregate reports and diagnostic views.
6. Exercise the actual worker on synthetic two-repeat nested CV, partial completion/resume, result publication recovery, label/feature isolation and read-only reporting. Preserve negative scores and conditional-interval limits.
7. Reuse a verified native server environment; foreground launch only. Generate upload checksums last and reverify after final changes. Deliver English server files plus a detailed Chinese local summary.

## Acceptance evidence

- Local tests are synthetic and use the actual frozen preprocessor, not server data.
- The Linux preflight runs before any native fit. No local test establishes scientific performance or predicts elapsed server time.
- R2 >= 0.20/0.30 is checked for the same pipeline in both repeats at full precision, never promised.
- Server execution and screenshots remain manual. Confirmation data remain unopened.
