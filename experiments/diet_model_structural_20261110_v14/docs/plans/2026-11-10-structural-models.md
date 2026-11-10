# V14 Structural Models Implementation Plan

**Goal:** Prepare one uploadable batch of 66 prediction workflows, preserving the fixed population, target and nested household validation, and investigate complementary information beyond V11/V13.

**Architecture:** Reuse the checked V6/V8/V10/V11 input/context stack and V13 closed INNER caches. Add 40 base workflows, 20 linear fusions and six nonlinear stacks; all fitting uses current outer-training data. Historical OUTER payloads are deserialized only after candidate choices and meta-models are fixed. Replay saved V11/V13 meta-models rather than solving their historical optimization again.

**Tech Stack:** Existing pinned NumPy, SciPy, sklearn, CatBoost and pandas; native V8 Python environment; no new installation. Separate local synthetic verification and native aggregate results.

## Design decisions

1. Keep SIR as a matched control. Add random-Fourier SIR and pilot-gradient projection. Do not extend the consistently weaker SAVE/distribution-objective grid.
2. Add local ridge, split-sample forest-kernel local regression, normalized ReLU NTK Nystrom regression, inductive graph residual smoothing, SIR-space SVR and sparse Bayesian RBF bases. These are finite-budget implementations, not claims to reproduce complete research software or its inference guarantees.
3. Add regularized oral/gut shared/difference plus private coordinates, with a paired-PC control of the same output width.
4. Fuse oral, balanced, non-tree, new, historical-bank and combined pools. Compare six nonlinear stacks to matched nonnegative stacks on the same bank/combined pool. Meta-CV remains an internal heuristic; only outer scores are reported as generalization estimates.

## Task 1: Freeze registry and configuration

Create `registry.py`, `config.json`; check 40 unique base names, 80 candidate settings, 26 meta workflows, 128 historical bank members and 168 combined members. Save complete grids in `docs/MODEL_GRID.md`. Preserve all older runtime-file hashes.

## Task 2: Implement role-fitted representations

Create `projections.py` and `features.py`. Reuse exact frozen CLR-to-raw indices and training prevalence selection. Fit nuisance regression, PCA, Fourier map, SIR, gradient pilot, CCA and scalers only on the training role. Verify known linear/nonlinear directions, CCA recovery, gradient finite differences, replay, serialization, sparse/constant inputs and held-out feature perturbations.

## Task 3: Implement regressors and nonlinear stacks

Create `estimators.py`, `models.py`; extend `fusion.py`. Verify local weighted regression against independent normal equations, NTK PSD and exact full-center KRR limit, graph smoothing against dense linear equations, forest partition/estimation household separation, constant-response behavior and RBF basis replay. Run every default candidate on wide synthetic CLR inputs.

## Task 4: Add strict V13-bank adapter

Extend `bridge.py`; create `legacy.py`. Check source/data identity and closed receipts, use saved V13 prior INNER/selected-member caches, verify replay member order and inner selections, retain all required hashes. Test actual composite synthetic cache trees, corruption, missing units and source/input drift. Verify INNER access never opens OUTER payloads for model fitting and baseline solvers are never refit.

## Task 5: Complete runtime/report interfaces

Adapt `engine.py`, `runtime.py`, `report.py` and shell entrypoints. Keep exclusive locks, exact resume, completed-unit reuse, partial publication recovery and no redirected file logs. Log hashing progress. Default short report and one diagnostic screen; detailed scores remain native. No causal or effect-modification claim.

## Task 6: End-to-end verification and delivery

Run the actual worker over two repeats and ten reduced synthetic outer folds, stop/resume without cached refits, corrupt or change identity and verify refusal, perturb outer labels/features and verify learned choices/meta-functions stay fixed. Run lint, formatting, shell syntax, ASCII and upload-manifest checks. Write detailed Chinese `summary.md` and English README/console; record exact checks and platform limits. No native HPP fit or promised target R2.

The user has already authorized implementation and local checks. Work stays in this dedicated experiment folder; this workspace is not a Git checkout, so no worktree/commit is required. No additional approval checkpoint or subagent delegation is used.
