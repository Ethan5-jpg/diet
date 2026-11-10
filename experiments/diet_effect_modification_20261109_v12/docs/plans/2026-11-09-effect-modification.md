# Diet–microbiome effect modification implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare a discovery-only server batch that separates microbial main-effect prediction from modification of the fiber-density–D_Glu association.

**Architecture:** Keep the existing V4 fold-local preprocessing, V6 discovery adapter and V10 outcome-free context reader. Add an isolated V12 directory containing an outcome-blind ecological interaction target, fold-stratified finite-basis partial regression with household leverage adjustment, and twelve matched additive-versus-interaction prediction comparisons. Native execution remains gated until V11 is complete and the V6–V11 summary has been reviewed; synthetic verification is permitted now.

**Tech Stack:** Python 3.11, existing pinned NumPy/SciPy/pandas/pyarrow/scikit-learn, Bash. No new packages, SSH, participant downloads, browser control, API calls or confirmation-outcome access.

---

## Scientific choices fixed before implementation

- Population: exactly the same 3,343 discovery participants in both repeats. No outcome, visit, exclusion or meal-response target is added.
- Primary target: global zero coefficients for (fiber density − 10)/5 multiplied by oral Shannon entropy, oral maximum relative abundance, gut Shannon entropy and gut maximum relative abundance. Raw ecological definitions are identical across folds; no outcome-dependent target selection or median split. Use a conservative four-coordinate Bonferroni-union test; retain the original multivariate score/Wald test only diagnostically because pre-native simulation showed mild size inflation.
- Primary inference: repeat 0, four-dimensional ordinary partial-regression working-model target with fold-specific background coefficients. Feature maps are fitted only in training households; inference background coefficients are estimated in held-out folds, separately from prediction evaluation. Household leverage-adjusted sandwich and score-multiplier inference are exploratory and model-dependent. Repeat 1 tests split sensitivity, not independent replication. Raw-outcome results on the same people are a planned sensitivity, not a second primary opportunity.
- Nuisance/additive background: frozen host/diet terms, full V10 nutrition/lifestyle context, documented UPF bounds, continuous ecological main effects, 30 unsupervised PCs per site, extra univariate main-effect curvature, and diet-by-background terms that do not involve microbes. No diet-by-microbial term in the matched null.
- Prediction registry: ecology/composition × linear/spline diet × oral/gut/both = 12 augmented models. Same additive background and selected main penalty; center products and orthogonalize against the matched main design in training rows only so they cannot simply duplicate penalized main directions. Interaction candidates include an exact additive zero option. All 12 are reported, including negative increments.
- Fixed display contrast: 10→15 g/1,000 kcal, mg/dL. Keep total contrast, interaction component and difference from matched null separate. Report local exposure-range support; never call model contrasts observed personal effects.
- Uncertainty: household score multipliers for low-dimensional coefficients; paired household bootstrap of fixed OOF predictions for ΔR² only. The latter is explicitly conditional, not a full refitting interval and not corrected for historical adaptive searches.
- Interpretation: ecological markers and compositional PCs are association modifiers, not proven mechanisms. PC loadings remain fold-specific; cross-repeat stability uses predicted contrasts, not a pooled PC coefficient.

## Task 1: source boundary and runtime

Create `bridge.py`, `runtime.py`, `config.json`, `setup.sh`, `server_run.sh`, `server_report.sh` and `upload_check.py`. Read only the established discovery inputs. Require V11's native completed summary and an explicit acknowledgement of the prior experiment-summary review before fitting. Check dependencies, source and input hashes, retained-on-failure run locks, closed-file checksummed checkpoints and exact resume identity.

Verify: blocked prerequisites, changed-source rejection, invalid run IDs, completed/corrupt checkpoint behavior, no real-data read in synthetic checks.

## Task 2: frozen, replayable feature construction

Create `features.py` and `synthetic.py`. Reuse frozen host and CLR preprocessing; fit PCA, context encoders, spline knots, scaling and rank handling on each training role. Store replayable main and interaction designs, raw ecological target coordinates, training-defined display IQRs, and outcome-free local support diagnostics. Recompute every diet-dependent column for fixed contrasts.

Verify: held-out changes cannot change fitted transforms; fixed-target coordinates have stable units; replay matches training/validation predictions; main-effect background contains both microbial sites for every comparison; support failure does not change the prediction population.

## Task 3: finite-basis interaction inference

Create `inference.py`. Freeze training-fitted feature maps; use held-out-fold QR partial regression to fit the ordinary model with separate nuisance coefficients for each fold and common target coefficients. Keep this inference use of held-out outcomes separate from OOF prediction. Account for nuisance and target leverage in household scores. Require full target rank and enough households. Save joint/site/coordinate tests and IQR-scaled interaction differences with intervals. Apply Holm within the declared follow-up families.

Verify: known interaction recovery; strong nonlinear microbial main effects without interaction; correlated diet/microbes; household dependence; rank deficiency becomes unavailable rather than significance; repeats are never stacked as extra people.

## Task 4: nested matched prediction

Create `models.py` and `engine.py`. Three household inner folds select the common additive penalty and each augmented model's interaction penalty. Fit selected settings on the full outer training households; evaluate untouched outer households. Store all predictions and models only under `TRE_ONLY`. Preserve actual negative ΔR², zero-option selections, loadings, warnings and full/interaction-only contrasts.

Verify: no outer outcome enters selection; exact zero option reproduces the additive model; all diet columns change correctly; negative and zero results survive reporting; resume skips completed units.

## Task 5: aggregate reporting and release records

Create `run.py`, `report.py`, `README.md`, Chinese `summary.md`, `METHODS.md`, `RESULTS_SUMMARY.md` and `research_records/verification/`. Reports use English and at most three compact screens. Full inference details and aggregate tables stay on the server; no participant export helper is provided. Preparation documents must say no native scientific result exists.

Verify: synthetic end-to-end nested run, numerical identity after save/replay/resume, manifest completeness, syntax and meaningful tests. Record synthetic calibration with its simulation assumptions and Monte Carlo uncertainty; do not label it empirical evidence or a proof of universal test validity.

## Handoff sequence

1. V11 finishes in the original chat.
2. Original chat summarizes V6–V11.
3. Reconcile that summary with this prepared design and freeze the final configuration.
4. User uploads the one new experiment directory and runs the supplied server commands.
5. User supplies compact result screenshots; archive them inside this module without downloading server data.

No git repository is present at this experiment root, so there is no commit/worktree step. The new directory isolates all edits from V11 and historical sources.
