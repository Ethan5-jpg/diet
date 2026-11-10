# V8 overnight discovery batch

55 model pipelines, 138 candidate settings. No real-data performance has been measured locally. All local fits use synthetic profiles.

## Upload and run

Upload this whole directory next to V6/V7 under:

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_overnight_20261108_v8
```

In the server terminal:

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_overnight_20261108_v8
bash setup.sh && /home/ec2-user/.venvs/hpp_v8_20261108/bin/python launch.py --console --run-id overnight_v8_20261108 --threads 4
```

The environment is created on the server native filesystem, outside the FUSE project mount. Seven numerical dependencies are pinned, including Arrow 15.0.2. No project-local virtualenv, symlink workaround or global environment upgrade is used. `HPP_BASE_PYTHON` and `HPP_VENV_DIR` may override the installer defaults.

An already working V7 environment can also be used if its dependency versions still match:

```bash
/home/ec2-user/.venvs/hpp_v7_20261107/bin/python launch.py --console --run-id overnight_v8_20261108 --threads 4
```

Do not start both commands for the same run. Launch verifies upload hashes, runs synthetic checks, then replaces itself with the foreground worker. Expect the full-grid synthetic preflight to take a few minutes. Keep the terminal process alive; browser disconnection behavior depends on the hosting session. No file log or automatic background service is created. The research run begins at `CONSOLE_WORKER`, `LOAD`, then `START r0_f0 ...`.

## Screenshot reports

After `BATCH_COMPLETE`, print the short report:

```bash
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python report.py --run-id overnight_v8_20261108
```

For the full ranking or warning summary:

```bash
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python report.py --run-id overnight_v8_20261108 --all
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python report.py --run-id overnight_v8_20261108 --diagnostics
```

For aggregate checkpoint counts while running, use another terminal:

```bash
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python report.py --run-id overnight_v8_20261108 --progress
```

These commands never fit a model or print individual records. Reports verify completed checkpoint hashes before recomputing aggregates. `--all` can span more than one screenshot. The default report stays below 40 lines. `J` marks added dietary context; `W` marks a fit warning, including contributing fusion members.

## Resume

If an interrupted worker has actually exited and its lock is absent:

```bash
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python launch.py --console --run-id overnight_v8_20261108 --threads 4 --resume
```

Code, configuration, dependencies, input/reference hashes, interpreter and thread count must match. A retained lock is never silently removed. Inspect its recorded PID first; do not delete it merely because a terminal appears quiet. Completed inner candidates and outer refits are reused. Partial batches have no pooled R2 or final summary. Incomplete final summary writes are preserved and recovered after successful completion.

## Frozen question and method breadth

The primary question remains participant-level `fiber_density -> D_Glu`, original raw mg/dL and original eligibility. The two household outer partitions and all training-only phenotype/CLR/diet transformations are unchanged. Expected real evaluation size from prior runs is 3343 per repeat; exact equality to the frozen V4 row/target reference is enforced at every outer fold. No confirmation data, additional outcome, CGM-derived covariate or participant-level export is used.

| Family | Pipelines | Purpose |
|---|---:|---|
| CLR HGB/CatBoost and Hellinger HGB anchors | 6 | Recompute current inner OOF features for honest fusion |
| NMF plus screened taxa, HGB/CatBoost | 4 | Nonnegative co-abundance factors plus local taxon information |
| CLR PCs plus screened taxa, HGB/CatBoost | 4 | Preserve global structure and selected local signals |
| Host-residual supervised PLS, HGB/CatBoost | 4 | Outcome-aware low-dimensional factors with nonlinear downstream fitting |
| Interaction-aware screening, HGB/CatBoost | 4 | Retain taxa whose main-effect correlation can be small |
| Joint-diet HGB/CatBoost | 4 | Add only documented UPF bounds and their missingness flags |
| Supervised leaf-rule elastic net | 2 | Sparse aggregation of learned tree partitions |
| Piecewise-linear LightGBM | 2 | Linear leaf models on compact hybrid features |
| Huber-loss CatBoost | 2 | Reduce sensitivity of fitting to extreme training residuals |
| Rotated-PC HGB ensembles | 2 | Combine three training-frozen coordinate rotations |
| Host-residual neighborhood smoothing | 2 | Local microbial similarity around the host prediction |
| Guarded factorization machines | 2 | More optimization budget and explicit convergence diagnostics |
| Additive NMF main-effect nulls | 2 | Host/diet plus separate oral/gut functions; no diet-by-microbe products |
| Screened ridge main-effect nulls | 2 | Matched background for group-sparse varying coefficients |
| Matched sparse/NMF diet interactions | 4 | Test explicit extra heterogeneity above the same selected null |
| Convex/ridge/selection meta learners | 9 | Core, wider, added-UPF and controlled with-FM comparisons |

NMF is fit on training Hellinger profiles and augmented with 32 training-screened CLR taxa per site. These statistical factors are not annotated functional pathways. PLS uses training outcomes only. PCA/NMF/PLS training and validation scores all use the same saved `transform` operation, avoiding fit-transform projection mismatches.

Original-input and joint-diet meta pools remain distinct. The joint-diet branch adds `UPF_lower_equalday_fraction` and `UPF_upper_equalday_fraction`, fractions in [0,1]. Missing bounds are training-median imputed with explicit missingness indicators; rows are not discarded. It does not treat an unknown FoodID as non-UPF or reconstruct an unverified food dictionary. It does not isolate the causal effect of fiber.

Each original strong anchor and its joint-diet counterpart use identical candidate grids. Added-context comparisons therefore include nested re-selection. The explicit interaction models are stricter: they use exactly the selected null and its saved features, then fit only an orthogonalized diet-product correction. The zero-weight candidate is numerically identical to its null. Group sparsity is applied to the two diet-basis coefficients per microbial feature; this is a custom disjoint group-lasso correction, not an implementation of glinternet.

## Evaluation and artifacts

- 2 x 5 outer household folds, with 3 household inner folds per outer training set.
- Setting selection uses pooled inner MSE. Each repeat's R2 uses pooled outer held-out predictions, not mean fold R2.
- No outer labels, old V6/V7 outer predictions or outer features fit a scaler, representation, selector or fusion weight. Historical predictions enter evaluation only, after all choices.
- The same core fusion is fit without FM and with a newly optimized FM member. Direct fusion gains are reported; a small old FM weight is not assumed to imply a specific effect.
- A single pipeline must pass both repeats at full precision for the >=0.20/0.30 counters. Negative scores remain visible. Warnings propagate from selected inner fits and contributing fusion members.
- Cached factorization warnings are retained with the saved representation. Ridge fusion records raw coefficients separately from standardized contributions; raw coefficients are not percentages. Warning propagation uses standardized contributions so prediction scaling cannot hide an affected member.
- Fixed fiber contrast: 10 -> 15 g/1000 kcal. Interaction SD is the model's centered correction contrast in mg/dL, not an individual causal treatment response.
- Conditional household intervals use paired outer losses, cluster sums and a normal approximation. They condition on learned predictions and do not account for adaptive discovery searches, representation selection, repeated participants or multiplicity.

`runs/<run-id>/registration.json` records source/input/reference hashes and environment. `TRE_ONLY/` contains private checkpoints, fitted representations/models and predictions; keep it on the server. `summary.json` contains complete aggregates, settings, fusion weights and fit diagnostics. `SUMMARY.txt` is the compact English output. Local `summary.md` explains the design and validation in Chinese; it is not a real-data results report.

The batch makes 4140 inner candidate evaluations, 460 selected outer evaluations and additional selected-null refits. Zero-interaction cases reuse their null and representations are cached. Runtime cannot be inferred precisely from synthetic checks or last night's smaller batch. Progress is printed at each model stage and outer completion.

## Verification and sources

Run `python check.py` for synthetic checks only. `research_records/validation.json` records the actual local verification; `upload_manifest.json` detects incomplete/changed uploads. The local numerical environment matches the pins on macOS ARM64. Linux/Amazon Linux runtime and actual predictive performance must be checked after upload; no local synthetic score is an HPP result.

The second review passes 29 checks, including constant/rank-dropped covariates, saved-factor warning retention, and scale-invariant fusion warning propagation. Details and the previous source snapshot are in `research_records/recheck_20261108/`. Use the updated whole V8 directory when uploading.

- [Scikit-learn NMF API, version 1.4](https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.NMF.html)
- [LightGBM 3.3.5 linear-tree parameters](https://lightgbm.readthedocs.io/en/v3.3.5/Parameters.html#linear_tree)
- [Friedman and Popescu, Predictive learning via rule ensembles](https://arxiv.org/abs/0811.1679) (inspiration for leaf-rule aggregation; this code is a smaller custom estimator)
- [Lim and Hastie, Learning interactions via hierarchical group-lasso regularization](https://pmc.ncbi.nlm.nih.gov/articles/PMC4706754/) (related motivation; the implemented correction does not impose their strong-hierarchy constraints)

These methods motivate alternatives; none of these references establishes an R2 expectation for the present cohort.
