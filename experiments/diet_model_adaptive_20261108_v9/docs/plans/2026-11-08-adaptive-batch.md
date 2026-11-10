# V9 adaptive microbiome modeling implementation plan

**Goal:** Deliver a larger, locally verified, uploadable discovery-only experiment batch with new compositional, zero-aware, neural and adaptive-fusion methods.

**Architecture:** Reuse the frozen V6 data adapter and V8 role utilities without changing them. Fit all new representations and model choices inside the current training role. Recompute current inner OOF predictions for every fusion learner; use completed V6/V7/V8 predictions only for final matched evaluation. Keep the original fiber_density -> D_Glu target, native mg/dL, eligibility and 2 x 5 household outer folds with 3 inner folds.

**Tech stack:** Python 3.11; the same pinned NumPy/SciPy/pandas/Arrow/scikit-learn/CatBoost/LightGBM versions as V8. NumPy implementations avoid new binary/GPU/model-weight dependencies. Server scripts and output are ASCII; the local detailed summary is Chinese.

## Design and scope

- Plan 24 atomic families x 2 site settings, four matched explicit interactions and 12 meta learners: 64 pipelines, 164 inner candidate settings.
- New input geometry: hurdle presence/positive abundance, log1p abundance, training ECDF ranks, stable-reference log ratios, supervised multiscale balances, masked rCLR low-rank completion, ecological summaries and compact hybrid features.
- New learners/objectives: ordered and leafwise CatBoost, Tweedie boosting, training-only Gaussian target transforms, residual two-layer neural networks and deep random vector functional links.
- New fusion: regularized patient-specific softmax gates, nonlinear tree meta regression and polynomial stacking alongside fixed convex/ridge controls. Gate covariates are deterministic ecological/host summaries with training-fitted scaling; no cross-fold PCA axes are mixed.
- New dietary-input branch: day-level fiber/energy variability from the documented FOOD-ONLY daily table. Read explicit permitted columns, preserve composite participant keys, never read CGM/day Y, and impute within training roles. Label these pipelines D; do not change primary exposure or evaluation rows.
- Explicit interactions use the same selected balance/eco ridge main-effect null, zero-weight option, and a weaker penalty path. Preserve negative and null outcomes.
- Hold all model choices fixed before running. No target R2 is promised; all rankings remain exploratory because the discovery cohort has been used repeatedly.

## Implementation steps

1. `bridge.py`, `daily.py`, `tests/test_data.py`: read historical comparators with receipt/row/target guards; aggregate documented food-only records. Test shuffled identity/date mapping and prohibited-column projection before integration.
2. `features.py`, `tests/test_features.py`: implement train-only dictionaries, zero handling, rank/reference/balance/rCLR/ecological features and replay. Test closure invariance, masked projections, saved warnings and validation-feature isolation.
3. `estimators.py`, `models.py`, `tests/test_models.py`: implement compact CPU estimators, target inverses and matched interactions. Check neural and gate gradients independently, log-balance constraints, zero correction identity, native contrast equality and serialization.
4. `fusion.py`, `engine.py`: fit every meta learner on current inner OOF predictions. Keep original-input and dietary pools separate. Propagate input-member warnings; save weight semantics.
5. `runtime.py`, `run.py`, `launch.py`, `report.py`, `setup.sh`: reuse immutable run identities, retained locks and close-before-rename checkpoints. Foreground terminal output only; native-filesystem venv; no persistent FUSE log.
6. Verify all default candidate cases on sparse synthetic profiles and the actual two-repeat runtime with smaller synthetic budgets. Test label/feature isolation, cache resume, corrupted receipts, report precision, partial result recovery, ASCII, syntax, pinned dependencies and old-source hashes. Publish exact checks and a checksum upload manifest; do not create an automatic ZIP or start server work.

## Reporting

Default output: baseline and leading R2 pairs, gains versus original-input V8 ridge_wide, matched null/interaction gains, dietary-input label, fit warnings and same-pipeline two-repeat target counts. Keep it below 40 lines. Retain complete aggregates and diagnostics on the server. No raw outcomes, participant predictions or taxon weights are requested locally.

## Method references and deliberate limits

Masked rCLR completion is a small inductive iterative-SVD implementation inspired by [Martino et al.](https://doi.org/10.1128/mSystems.00016-19), not DEICODE/OptSpace. Supervised balances are a custom multiscale construction inspired by [Rivera-Pinto et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6050633/), not the selbal algorithm. Reference log ratios follow the compositional log-contrast motivation in [Shi et al.](https://arxiv.org/abs/1603.00974).

TabPFN was considered through its [official repository](https://github.com/PriorLabs/TabPFN), but model access/license acceptance, checkpoint downloads and CPU resource requirements are not established for this isolated server. It is not inserted as an unverified default or an API data-transfer path.
