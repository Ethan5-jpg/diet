# V13 Complementary Prediction Models Implementation Plan

**Goal:** Test genuinely different predictive representations, spline/kernel models and response-distribution objectives after the documented V6-V11 results, retaining the same endpoint, participants and household validation.

**Architecture:** An isolated sibling experiment reuses immutable V6/V8/V10/V11 input/role helpers. New fold-fitted sufficient-dimension-reduction representations feed matched predictors. A provenance-checked V10/V11 INNER cache bank preserves the V11 strong ensemble; no historical OUTER predictions enter training. V12 remains the separate effect-modification analysis and is not modified here.

**Tech Stack:** Python 3.11, NumPy/SciPy/pandas and existing pinned scikit-learn/CatBoost APIs. Reuse the existing native V8 environment; no download, package install, cloud model, new endpoint or GPU. Default two threads supports a separate V12 terminal when resources permit. No Git repository/worktree is available or required.

## Evidence, alternatives and limits

- V6-V11 cumulative summary is complete at `../post_v5_experiment_summary_20261109/summary.md`. Stronger full prediction chiefly came from nutrition context and fusion, not stable explicit microbial modification. V11 positive_combined is 0.130555/0.139855, only +0.004328/+0.002589 over V10. No target R2 is promised.
- V8/V11 already tried ordinary linear-leaf trees, nonlinear screening and several PC/PLS/NMF variants. Rebranding another boosted-tree grid as new would add little information.
- Selected approach combines residualized PC controls with SIR/SAVE-inspired supervised inverse-regression directions; MARS-style paired hinge regression; FITC sparse Gaussian processes; quantile-bin probability regression; multi-quantile regression; and graph-smoothed coefficients.
- Full dense GPs/BART or externally pretrained tabular models were considered but not selected: the former increase CPU/implementation cost, while the latter would require new dependency/weight access and raise data-boundary issues. Existing exact endpoint and original population stay fixed.
- This is supplementary prediction exploration, not a statistical test of microbial modification. Newly used algorithms are predictive implementations; no central-subspace consistency, causal interpretation, pathway identity or calibrated GP/quantile uncertainty is claimed.

## Frozen batch specification

Thirty-eight base workflows:

1. Residualized PC, SIR and SAVE representations x HGB/CatBoost x oral/balanced (12).
2. Those same representations x paired-hinge regression/FITC GP x oral/balanced (12).
3. PC/SAVE x quantile-bin CatBoost/quantile-bin HGB/multi-quantile CatBoost x oral/balanced (12).
4. Graph-smoothed CLR ridge x oral/balanced (2), including a zero graph-penalty option.

Five pools (matched oral, matched balanced, non-tree regressors, all new, V11 bank plus new) get convex/ridge/nonnegative fusion (15), with sparse new/combined fusion (2). Total 55 new workflows. The frozen base candidate count is 84 and all 84 were fitted, predicted and serialized in local checks. Fusion penalties are additional internal training choices.

## Task 1: Input/role bridge and representations

Create `bridge.py`, `context.py`, `food_features.py`, `features.py`, and a frozen config/registry.

- Reuse current energy/fiber and complete-nutrient alignment checks; fixed discovery evaluation rows, no daily Y or confirmation input. Diet context is the V11 joint-day macro background with sodium/lifestyle/alcohol.
- For each site, select variable CLR columns using training-only raw prevalence, with deterministic ordering; use the same source-column policy for PC/SIR/SAVE.
- Regress selected CLR on the role's host/context nuisance design. Transform train/validation with that same coefficient matrix.
- PC uses the leading covariance directions. SIR/SAVE slice the training nuisance-residualized outcome, use regularized whitening, and learn mean/covariance-change directions. Handle ties, rank drop and constant targets explicitly. Do not select features or slices from validation labels.
- Persist all projection/scaling state. Identical train/replay operations and outer-label/feature perturbation checks are mandatory.

## Task 2: New regressors

Create `estimators.py` and `models.py`.

- Paired hinges: deterministic training quantile knots, bounded degree/term count, greedy paired-hinge forward search with stable orthogonalized loss calculations, followed by documented GCV-style backward pruning and ridge coefficients. Call it MARS-style, not a full reproduction of a mature MARS library. Test known piecewise/additive/interaction functions and serialization.
- FITC GP: training-only inducing centers; Matérn kernels; diagonal correction `diag(Kff-Qff)` plus noise; stable Cholesky solves. Use all training outcomes. Verify against an independent dense corrected-prior solve, exact-GP limit and validation-only changes. No uncertainty interval is advertised.
- Quantile-bin regressors: training quantile cuts, per-observed-class training means and predicted class probabilities. Empty/tied/one-class targets must be valid. Return mean predictions in the original unit; this is not a proportional-odds ordinal model.
- MultiQuantile CatBoost: fixed evenly spaced midpoint quantiles, average to approximate the mean; verify pinned 1.2.8 prediction shape and one-target fallbacks. Quantile output is not proof of individual effect heterogeneity.
- Graph-smoothed ridge: positive training CLR correlations, bounded nearest-neighbor graph, normalized within-site Laplacian and SPD coefficient prior. Include graph weight zero. The graph is statistical, not a validated biological network. Verify PSD and independent normal equations.

## Task 3: V11 bank provenance and numerical reconstruction

Create `legacy.py`.

- Verify completed V11 scope, data/source hashes and closed receipts, plus the V10 bank that V11 actually used. Unrelated display helpers are not mandatory dependencies.
- For each current outer fold, rebuild V10's 28 selected INNER columns and V11's 62 selected INNER columns against identical current inner target rows. Cache these current-outer-training matrices only.
- Reconstruct V11 positive_combined with its original household meta-CV penalty recipe. Store the trained replay model before any old outer payload is deserialized.
- After every new model choice/meta fit is fixed, read matching V10/V11 OUTER arrays for inference. Check rows, processed targets and selected settings, and require V11 replay maximum error <=1e-8 mg/dL. Do not fit or tune using these old outer arrays.
- Tests cover missing/corrupt cache units, source/data drift, row/target order, read-only behavior and no outer-deserialization during training. Synthetic integration must exercise the actual bank adapter, not solely a fake reference.

## Task 4: Nested engine, reporting and operation

Create `engine.py`, `fusion.py`, `runtime.py`, `run.py`, `launch.py`, setup/run/report helpers.

- Same two outer household repeats x five folds, three inner household folds. Candidate selection uses pooled inner MSE; all new learning stays in the corresponding training role.
- Outer predictions are aligned to the original frozen V4 processed Y/row indices. No new complete-case cohort or altered outcome.
- Meta tuning is an outer-training heuristic on inner OOF features, not an independent validation score. Preserve signed/negative gains and contributing fit warnings.
- Compact ASCII report, matched representation/objective comparisons, native-unit errors and historical benchmark. Full aggregates/checkpoints remain native; no individual rows printed. Report not-yet-started/partial status without fitting.
- Foreground TTY, closed-file checkpoints, strict run identity, no continuous FUSE log, no silent lock removal. Two-thread default; separate from V12. A read-only resource helper reports affinity/quota and memory before users choose concurrency.

## Task 5: Local verification and handoff

Create meaningful tests in `tests/`, English README/METHODS and Chinese `summary.md`.

- Every default candidate fits, predicts and serializes on wide sparse synthetic profiles under exact installed dependency pins.
- Numerical identities and known nonlinear signals for all custom estimators; constant targets/columns, incomplete context, rank drop, ties, probabilities and kernel stability.
- Actual reduced-budget two-repeat worker, partial stop/resume without refitting, report publication recovery, fixed population, household isolation, negative R2 and unchanged choices under outer-label/feature changes.
- Execute shell dispatch with spaces/arguments, exact environment check and setup smoke; no installation in native scripts.
- Capture source/old-dependency hashes, platform and test logs honestly. Local macOS LightGBM differs from native Linux; native preflight remains required. Produce no ZIP and change no prior experiment source or V12 file.
- After checks pass, build upload manifest, remove local caches/interpreter path and deliver the folder plus exact native commands. Scientific results remain pending user-run screenshots.

## Method/API sources to verify

- Li (1991), Sliced Inverse Regression, DOI 10.1080/01621459.1991.10475035; Cook/Weisberg SAVE formulation and regularized predictive adaptations.
- Friedman (1991), Multivariate Adaptive Regression Splines, DOI 10.1214/aos/1176347963.
- Quinonero-Candela and Rasmussen (2005), A Unifying View of Sparse Approximate Gaussian Process Regression, JMLR 6:1939-1959.
- CatBoost official regression/classification objective documentation; actual pinned APIs tested locally.

No prior exploratory score is reclassified as prospective confirmation, and no weak/negative interaction result is hidden to justify the new batch.

## Completion record

Implemented tasks 1-5. The 44-test local suite passed in 98.362 seconds, including actual reduced two-repeat worker and historical-bank integration. Numerical-reference, label/feature perturbation, resume/publication, 84 default-case API, source-integrity and shell checks passed. Native real-data execution/results remain pending. See research_records/local_verification.json.
