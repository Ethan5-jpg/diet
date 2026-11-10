# V9 adaptive microbiome batch

64 model pipelines, 164 inner candidate settings. Real-data performance is pending manual server execution. Local fits use synthetic inputs only.

## Upload and start

Upload the entire directory next to V6/V7/V8 under:

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_adaptive_20261108_v9
```

Run in the server terminal:

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_adaptive_20261108_v9 &&
bash setup.sh &&
/home/ec2-user/.venvs/hpp_v9_20261108/bin/python launch.py --console --run-id adaptive_v9_20261108 --threads 4
```

The existing V8 environment can also run this batch if its pinned versions remain unchanged:

```bash
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python launch.py --console --run-id adaptive_v9_20261108 --threads 4
```

Use one launch path. The installer verifies uploaded files before creating a native-filesystem venv outside the FUSE project mount. It pins the same seven numerical packages used in V8, including Arrow 15.0.2. No torch/GPU framework, pretrained checkpoint, credential or external prediction API is needed.

Launch performs synthetic checks and a project-filesystem probe before replacing itself with the foreground worker. Several minutes of preflight output are expected. A browser disconnection is distinct from terminating the VM terminal process. The worker uses terminal output, not a persistent FUSE file log. `CONSOLE_WORKER`, `LOAD`, `FOOD_CONTEXT`, `HASH`, then `START r0_f0 ...` identify the launch stages.

V6/V7/V8 directories, completed references and the original handoff inputs must remain available. The loader reuses V6 preprocessing and V8 role utilities without modifying them. Research performance and total duration cannot be inferred from small synthetic tests; inspect stage and outer-fold completion messages for progress.

## Read the result

After `BATCH_COMPLETE`, run in this directory:

```bash
/home/ec2-user/.venvs/hpp_v9_20261108/bin/python report.py --run-id adaptive_v9_20261108
```

The default aggregate report stays below 40 lines. Additional read-only views:

```bash
/home/ec2-user/.venvs/hpp_v9_20261108/bin/python report.py --run-id adaptive_v9_20261108 --diagnostics
/home/ec2-user/.venvs/hpp_v9_20261108/bin/python report.py --run-id adaptive_v9_20261108 --all
/home/ec2-user/.venvs/hpp_v9_20261108/bin/python report.py --run-id adaptive_v9_20261108 --progress
```

`--all` is the longer complete table. `--progress` prints checkpoint counts and lock presence; it does not assert process liveness. Reports never fit a model or print individual outcomes/predictions. The completed report recomputes aggregates from checksum-verified outer checkpoints and compares them with `summary.json`.

`D` marks the extra FOOD-day variability branch. `W` marks a fit warning, including selected-inner and contributing-member warnings. Negative scores and failed convergence diagnostics remain visible; absence of W is not an independent validation result.

## Resume

If the worker has actually exited and no lock remains:

```bash
/home/ec2-user/.venvs/hpp_v9_20261108/bin/python launch.py --console --run-id adaptive_v9_20261108 --threads 4 --resume
```

Sources, config, input/reference hashes, environment, interpreter and thread count must match. Existing locks are never silently removed. Completed candidates, selected nulls, meta fits and outer refits are reused. Partial batches have no pooled R2 report. Incomplete final summary files are retained and recovered; conflicting completed results are not overwritten.

## What is new

| Group | Atomic pipelines | Main change |
|---|---:|---|
| HGB/CatBoost anchors | 4 | Current inner-OOF controls and fusion members |
| Hurdle/log1p | 6 | Presence and conditional positive abundance; lower training prevalence threshold |
| Training rank features | 4 | Frozen empirical normal scores with ridge or RBF SVR |
| Reference log ratios | 4 | Training-selected stable reference, elastic net or CatBoost |
| Supervised balances | 4 | Multiscale signed log-ratio groups, ridge or HGB |
| Masked rCLR completion | 4 | Fixed-budget low-rank completion with inductive masked projection |
| Ecological summaries/hybrids | 6 | Richness, diversity, dominance, exact-ID cross-site similarity and local CLR features |
| Ordered/leafwise CatBoost | 4 | Different boosting/tree construction schemes |
| Alternative target objectives | 4 | Tweedie boosting or training Gaussian target transform; native-unit scoring |
| Residual neural/deep random-feature networks | 4 | Two learned tanh layers with ridge shortcut; or random tanh layers with ridge heads |
| FOOD-day variability | 4 | Added energy/fiber distribution features with training-role imputation |

These 48 atomic pipelines are augmented by four matched explicit interaction procedures and 12 meta learners, giving 64 in total. Atomic grids contribute 144 settings; the four interactions contribute 20 including their zero-weight cases, giving 164 inner settings. Meta presets are separately fixed, not secretly tuned against outer results.

The 12 meta learners comprise convex/ridge fusion on core, wider and daily-input pools; three softmax conditional gates; a tree meta learner; polynomial stacking; and an inner-selected original-input pipeline. The gate combines current inner-OOF predictions using host/ecological covariates. Its prior mixes 80% current global fusion weights with 20% uniform mass so a globally weak expert can still receive subgroup weight. Covariate scaling and gate parameters use outer-training data only. No PCA axes from different folds are treated as aligned gate coordinates.

## Input and evaluation boundaries

- Primary target: original participant-level `fiber_density -> D_Glu`, mg/dL. Keep frozen V4 eligibility, host/diet basis, phenotype handling and 2 x 5 household partitions; use 3 household inner folds.
- All outer rows and processed targets must exactly match the existing V4 reference. Prior evaluation size was 3343 per repeat. No other metabolic outcome or confirmation cohort is read.
- Old V6 fusion, V7 ridge stack and V8 ridge_wide/ridge_joint predictions are evaluation comparators only, read after current choices and predictions. Main deltas use original-input V8 ridge_wide.
- Settings use pooled inner MSE; each repeat's R2 uses pooled outer predictions. The same pipeline must pass both repeats, at unrounded precision, for the >=0.20/0.30 counters.
- All new screening, ranks, pseudo-counts, reference selection, balance weights, low-rank bases, scalers, imputers and target transforms fit only on the current training role. Training and validation use the same frozen projection operation.
- Gate features are deterministic raw host/ecological summaries. Held-out features may change a fitted gate's predictions and average weights; they must not change its learned parameters. Tests check this distinction explicitly.
- Discovery has already been used for several batches. Nested tuning within V9 does not erase cross-batch adaptive-search uncertainty. Shared participants across repeats are not independent replication.

## FOOD-only extension

Only this documented additional table is read:

```text
inputs/daily/food_days_discovery_TRE_ONLY.parquet
```

Projection is restricted to `participant_id`, `cohort`, `research_stage`, `date`, `energy_kcal`, `fiber_g`, `energy_eligible`, `fiber_eligible`. It never reads a daily CGM table, `Y_value`, scores or predictions. Identity/date keys are checked for uniqueness and matched by all three identity fields.

For people with at least three valid fiber days, 13 features describe fiber-density dispersion, quantiles, relative variability, nonlinear averages, high-density-day fraction, energy dispersion, fiber-energy correlation and weekday/weekend differences. Day counts and the difference between daily and participant mean density are audit quantities, not added model predictors. Missing people/features remain in the original evaluation set and are imputed using training medians with missingness indicators.

The primary exposure is not replaced. Added dietary context is a separate predictive-input question, marked D. `daily_cat` and `ecohybrid_cat` have identical settings for the paired added-context comparison. Other D comparisons are broader predictive comparisons. Coverage and aggregate alignment checks are stored in `registration.json` under `daily_extension`, and coverage is printed at load time. These features describe recorded visit diet, not an intervention or future dietary response.

## Interpretability and method limits

The 1% prevalence rule and source caps are training-only; frozen eligibility still uses the original supported profiles. Hurdle features are two-part input features, not a hurdle probability model for glucose. Reference log ratios and balance columns preserve log-contrast structure. Supervised balances use training host-residual associations; they are not a reimplementation of selbal.

Masked rCLR uses nonzero log centering, eight iterative-SVD completion steps by default, then the same masked ridge projection for training and validation. It is an inductive approximation inspired by robust Aitchison methods, not DEICODE/OptSpace. Fixed iteration budgets are not described as optimizer convergence. Neural models also use fixed epoch budgets, two seeds and training-only normalization; they have no validation-target early stopping.

Four explicit interactions compare balance/eco ridge nulls with only an orthogonalized diet-product correction. The penalty path is weaker than V8 and includes a zero option. The same selected main effects, rows and target are kept. Native-unit 10 -> 15 g/1000 kcal correction contrasts and matched delta_R2 are reported. Softmax/tree/polynomial fusion may contain implicit interactions; their total R2 alone is not evidence of a biological diet modifier or a causal treatment response.

Conditional household intervals hold learned predictions fixed and exclude adaptive-search uncertainty. Ridge raw coefficients are not percentages. Conditional convex weights are averages over the current outer validation covariates. Nonlinear fusion conservatively propagates warnings from its input members.

## Verification and saved artifacts

`python check.py` uses synthetic data only. It checks all 164 default cases, known nonlinear/subgroup signal, analytic neural/gate gradients, sparse projections, closure invariance, composite key alignment, prohibited food-column access, null identities, native contrasts, serialization, household isolation, held-out label/feature perturbations, the actual two-repeat runtime, resume, partial publication and corrupted receipts.

The small full-default smoke tests and reduced-budget full-runtime tests have different scopes; neither is a real HPP performance result. `research_records/validation.json` records the observed local checks and environment. Local testing is macOS ARM64, with the previously built LightGBM 3.3.5 library; the server's official Linux wheel must pass the same preflight. No claim of a larger R2 is made before server results.

The run directory stores `registration.json`, `summary.json`, `SUMMARY.txt` and private `TRE_ONLY/` checkpoints. Models, participant predictions and taxon weights remain on the server. Meta fits are checkpointed separately so evaluation payloads stay free of custom model objects. Local `summary.md` is the detailed Chinese design/result narrative. `upload_manifest.json` verifies the uploaded release; no automatic ZIP is created.

## Sources

- [Martino et al., sparse compositional dimensionality reduction](https://doi.org/10.1128/mSystems.00016-19)
- [Rivera-Pinto et al., microbial balances](https://pmc.ncbi.nlm.nih.gov/articles/PMC6050633/)
- [Shi et al., compositional regression](https://arxiv.org/abs/1603.00974)
- [CatBoost official training parameters](https://catboost.ai/docs/en/references/training-parameters/common)
- [Random vector functional link ensemble deep learning](https://arxiv.org/abs/1907.00350)
- [PriorLabs official TabPFN repository](https://github.com/PriorLabs/TabPFN): considered but not included as an unverified default; local model access, checkpoint availability and CPU budget have not been established.

These references motivate method families and do not establish an achievable R2 for this cohort.
