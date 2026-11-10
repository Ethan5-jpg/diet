# V11 distribution and residual models implementation plan

**Goal:** Test new person-level dietary representations and complementary regressors while retaining the completed V10 prediction bank as a strong, fold-matched comparison.

**Architecture:** Reuse frozen discovery preprocessing and V10 source validation. Fit all new dietary encoders and microbial projections inside each current training role. Reuse ONLY V10 inner prediction caches from the identical outer training households for fusion training; read historical outer predictions only after all choices and meta fits are fixed. Preserve V6-V10 sources and results.

**Tech stack:** Python 3.11 and the existing seven pinned V8 dependencies. CPU only, four threads, no new native installation or pretrained weights.

## Evidence and alternatives

- V10 finished 10/10 outer folds for the same 3,343 people. The balanced winner was ridge_diverse, R2 0.126227/0.137266. It did not reach 0.20.
- Macro additions were positive in HGB, CatBoost and ridge fusion; joint macro-day means had small positive gains in both tree families/repeats. Additional gut input and explicit interactions were weaker.
- Repeating the same average-input model grid offers limited new information. The selected direction combines richer food distributions, train-role empirical shrinkage and staged residual prediction.
- New endpoints, complete-case population changes, BMI/other concurrent outcomes, date-lag proxies, GPUs, outside prediction services and gated pretrained weights are out of scope.

## Implementation sequence

1. Create an isolated experiment folder and record hashes of every actually imported old source. Set up a persistent local synthetic-check environment outside the upload folder.
2. Reuse current-day energy/fiber alignment checks and primary exposure/day-count validation. Add complete-joint-day arrays, person counts, variances and covariance summaries with no labels. Test nullable fields, composite scope, missing people and exact row mapping.
3. Implement fold-fitted dietary encoders: full/joint controls, UPF context, two empirical-shrinkage variants, quality controls, joint covariance, random Fourier mean embeddings, prototype occupancy and random-projection quantiles. Give each person equal total weight when fitting daily bases; keep people with sparse new records.
4. Add conditional supervised PCs and nonlinear microbial screening. New learners include linear-leaf/DART boosting, posterior/robust CatBoost, Bayesian/ARD regression and reweighted ElasticNet, all tested under the pinned APIs.
5. Implement grouped cross-fit residual prediction conditional on the current role-fitted design. Include exact-zero correction options and explicit matched nulls. Do not call this causal orthogonalization or independent inner validation.
6. Read and checksum the completed V10 registration, configuration, inner receipts and prediction arrays. Verify source/data/fold identity. Reconstruct inner selections using current inner targets; never select using old outer predictions. Outer legacy predictions enter inference only after new choices/meta fitting; validate row, target and choice matches.
7. Refit convex/ridge/nonnegative/sparse fusions on current outer-training OOF matrices. Include matched oral/balanced pools, new/combined pools and exact V10 replay. Any meta tuning remains internal to outer training and is not reported as validation performance.
8. Exercise every default candidate on wide synthetic profiles, fixed native-unit contrasts, serialization, zero options, known distribution signals, provenance rejection, external-label/feature isolation, full two-repeat worker execution and resume.
9. Deliver concise ASCII server reports, detailed Chinese local summary and a rebuilt upload manifest. Native execution stays manual; no server data transfer. State the actual local test scope and do not promise a target R2.

## Acceptance criteria

- Fixed original exposure/outcome, population and household folds; training-only imputation, screening, projection, shrinkage, food bases and model selection.
- Distribution features have identical train/replay operations; no daily glucose outcomes are read.
- V10 cached predictors reproduce the registered fold-specific inner choices and outer row/target alignment; its outer scores are never fusion training inputs.
- Report total R2, paired input/pool gains and matched explicit interaction increments separately. Preserve negative outcomes and fit warnings.
- Same pipeline must cross R2 0.20/0.30 in both repeats at full precision. Repeats share people; search uncertainty is not removed by conditional household intervals.
