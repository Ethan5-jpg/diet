# Methods and frozen comparisons

## Motivation from completed V10

The screenshot-derived V10 winner was `ridge_diverse`, R2 0.126227/0.137266, versus V8 `ridge_wide` 0.120396/0.122745. Macro additions helped HGB oral (+0.008130/+0.007903) and Cat balanced (+0.003567/+0.004918). Joint macro-day means had smaller positive increments in both families/repeats. Sodium/lifestyle additions were less consistent; explicit microbial modifiers did not produce a large stable gain. These observations guide an exploratory batch, not independent hypothesis confirmation.

V11 emphasizes information representations and complementary regressors instead of another broad depth/learning-rate sweep. It preserves the V10 bank so that a new model can contribute without replacing a strong existing workflow.

## Fixed scope and preprocessing

- Same discovery rows and frozen V4 outer assignments; original processed `D_Glu` targets in mg/dL. Two repeats share the same 3,343 evaluated people.
- All outer/inner partitions separate `split_unit_id` households. Frozen CLR, host design, outcome processing and row support are fitted within the current training role, as in V6-V10.
- Diet encoders, missing-value medians/flags, alcohol categories, microbial screens and projections are fitted using role training rows only. Validation records are transformed using those fitted objects.
- Shared input readers project permitted food/lifestyle columns and filter participant/cohort/research-stage keys before materializing values. No daily glucose outcomes, additional clinical outcomes or confirmation rows are read.
- Original energy/fiber day alignment, primary fiber density and record count must agree. Nutrient features use complete recorded-event sums; missingness is not replaced with zero intake. New features requiring three days remain missing when insufficient, with training imputation and flags; people are retained.

## Eleven food profiles

All profiles retain host covariates and the existing sodium/lifestyle/alcohol context. `full` is the old separate-day-mean control. The other profiles start from `joint`: carbohydrate/fat/protein density means computed on the same days with all three complete, nonnegative values; the original sodium/lifestyle context stays fixed. A correlation with a zero-variance margin is encoded as zero, while its SD is retained.

| Profile | Added or changed information |
| --- | --- |
| full | V10 separate-day macro mean control |
| joint | Same-day three-macro means |
| joint_upf | Original UPF lower and upper fractions, with missing flags |
| shrink_global | Empirical shrinkage of three joint means using pooled within-person day variance |
| shrink_local | Half individual and half pooled within-person variance for shrinkage |
| shrink_uncertainty | Local shrinkage plus posterior-SD summaries and log record count |
| quality | Estimated within-person mean SEs and log record count |
| covariance | Three daily density SDs and three pairwise macro correlations |
| rff | Mean of 64 random Fourier features of each person's four-dimensional daily bag |
| prototype | Soft occupancy of 12 training-learned daily prototypes |
| sliced | Five quantiles along each of 16 fixed random daily projection directions |

Daily bags contain `[carbohydrate_g/1000kcal, fat_g/1000kcal, protein_g/1000kcal, log(energy_kcal)]`. This is recorded food composition, not verified ingestion timing. The upstream joint admission count is retained, but days are exchangeable within a person; no time-response model is claimed.

For daily basis fitting, each day receives weight `1/n_person`, so each training person contributes total weight one to the daily scaler and prototype fitting. RFF directions are seeded, not label-selected. Prototype membership is softmax of negative squared standardized distance (gamma 0.25). Projection quantiles summarize each person's own daily distribution. These are finite feature approximations inspired by distribution regression, not the estimators or guarantees in the cited papers.

For each macro, the empirical prior has training-person mean `mu` and `tau2=max(var(person means)-mean(within variance/n),1e-8)`. Pooled noise weights within-person variances by `n-1`. A mean becomes `mu + tau2/(tau2+noise/n)*(observed_mean-mu)`. The local variant averages individual and pooled noise. These are predictive regularizers; neither correction for validated measurement reliability nor an increase in the R2 denominator is claimed.

## Fifty base workflows

- HGB oral and Cat balanced for all eleven profiles (22), plus HGB/Cat host-only joint controls (2).
- Linear-leaf LightGBM, DART LightGBM, posterior-sampling CatBoost, Huber CatBoost, Bayesian ridge, ARD and coefficient-reweighted ElasticNet, each with oral/balanced inputs (14).
- Supervised-screen-then-PC HGB/Cat and quadratic-screen HGB/ridge, each oral/balanced (8).
- HGB balanced joint, Cat oral joint, and block ridge with oral/balanced PCs (4).

Linear-leaf LightGBM receives host/context and PCs, standardized in training; it uses the pinned CPU API. Posterior CatBoost uses the library's own posterior-sampling defaults without manually conflicting Langevin parameters. Huber delta is 1.0 on the training-standardized target. These settings explore complementary predictions; stochastic/posterior labels do not provide validated uncertainty intervals.

Supervised PCs select CLR columns by correlation with training outcome residuals conditional on host/context, then fit PCA and replay `transform` identically for train and validation. Quadratic screening scores linear, squared and positive-part CLR transformations, then retains selected linear/squared terms. Coefficient-reweighted ElasticNet rescales columns using a training ridge fit; this is not a claim of adaptive-lasso selection consistency. The same family/grid defines the matched oral and balanced pools.

## Eight residual workflows and four explicit fiber corrections

Selected background workflows are fit first. Within a current role, three-fold grouped cross-fitting refits the background regressor conditional on that role's frozen design. The residual stage learns `training Y - cross-fitted prediction`. It uses a fresh feature-design cache and residual-based microbial screening. Candidate stage weights are 0, 0.25, 0.5 and 1; exact zero returns the selected background unchanged. This tests remaining predictive structure after host or oral modeling.

This is **conditional-design cross-fitting**: the role's feature screen/encoder and chosen background setting are not independently reselected inside those extra folds. Those residual targets and meta CV scores are training tools, not independent validation or causal DML scores. Outer validation households remain entirely excluded from all fitting and selection.

Explicit corrections use gut, oral, both-site RCS-by-PC, or macro-by-fiber products. A training ridge projection removes the part predictable from the selected background design; a ridge correction learns the cross-fitted background residual. All have an exact-zero option. Their reported increment is versus their same selected background, which can itself contain tree interactions. It is not versus a universally additive biological null.

The contrast SD describes the **added correction's** varying fiber 10-to-15 g/1,000 kcal contrast, in native mg/dL. It is not the full model's contrast, a treatment effect, or proof that microbial states caused the observed heterogeneity. Finite-difference tests cover nonlinear fiber terms and constant-sex rank drop.

## Existing V10 bank and seventeen new fusions

The bank contains the 28 registered members of V10's `diverse` pool. Every source/input fingerprint needed to interpret these caches is checked against the completed V10 registration, including frozen V6/V8 helpers, V10 modeling sources and current data inputs. Receipts must identify one intact closed payload. No old files are edited; missing/incompatible caches stop execution.

For each current outer training split, V11 reads only that split's original three **inner** prediction matrices and reconstructs each old member's selected candidate using the current matching inner targets and pooled MSE. Those matrices enter `old__` columns. Old outer result payloads are byte-hashed for integrity but remain undeserialized during training. After all new candidates and fusion fits are fixed, V11 reads the old outer predictions for the matching held-out rows and checks the original processed target/row arrays and selected candidate indices. They supply held-out predictor values for combined fusion inference and historical evaluation, never fusion training.

A separate fixed StandardScaler + Ridge(alpha=n*0.1) replay on the 28 old inner columns must reproduce completed V10 `ridge_diverse` outer predictions to maximum absolute error at most `1e-8` mg/dL. It is a provenance/numerical reconstruction check, not a newly discovered model.

Five pools (matched 9-member oral/balanced, 62-member new, 18-member distribution, 90-member combined) each receive convex, ridge and nonnegative-ridge fusion (15). Sparse ElasticNet fusion adds new/combined variants (2). Ridge/nonnegative penalties are `[0.003,0.01,0.1,1]`; sparse penalties `[0.001,0.01,0.1]`. Household meta CV on the current training OOF matrix chooses penalties; its scores are not reported as independent performance. No old outer prediction is a training feature. Prior bank columns may include gut information: combined/new comparisons are bank additions, not matched tests of gut benefit.

## Selection, evaluation and reporting

Each atomic/correction workflow chooses its candidate by pooled current inner MSE. The selected model refits on the current full outer training role. Pooled outer MSE/R2 use identical processed validation rows; negative R2 is preserved. Fitted-model warnings, projection warnings, selected-inner warnings and contributing-fusion warnings are retained. Bayesian iteration caps are flagged, even if the library emits no warning.

Default output ranks by the lower repeat R2 and compares with V10 `ridge_diverse`. Diagnostic comparisons hold algorithm family/grid and participant population fixed while separately reselecting candidate settings. Household intervals use cluster sums of paired squared-error differences and condition on learned predictions; they exclude model-selection/adaptive-search uncertainty. No multiplicity-adjusted discovery or causal claim is made.

R2 0.20/0.30 threshold checks require the same workflow to pass both repeats at unrounded precision, with flagged and unflagged counts separated. Repeated outer partitions do not create a second independent sample. An independent confirmation stage would require a separately authorized, frozen analysis.

## Primary method/API sources

- Szabo et al., [Learning Theory for Distribution Regression](https://www.jmlr.org/beta/papers/v17/14-510.html), JMLR 2016: motivation for regression on sample distributions.
- Meunier et al., [Sliced Wasserstein Kernels for Probability Distributions](https://proceedings.mlr.press/v162/meunier22b.html), ICML 2022: motivation for projection-based distribution summaries; V11 uses an explicit quantile representation.
- [LightGBM 3.3.5 parameters](https://lightgbm.readthedocs.io/en/v3.3.5/Parameters.html): pinned linear-tree, linear-lambda and DART options.
- [CatBoost common training parameters](https://catboost.ai/docs/en/references/training-parameters/common): CPU posterior sampling behavior and parameter compatibility.
- [scikit-learn 1.4 StandardScaler](https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.StandardScaler.html) and [MiniBatchKMeans](https://scikit-learn.org/1.4/modules/generated/sklearn.cluster.MiniBatchKMeans.html): sample-weighted fitting interfaces. ARD/Bayesian iteration APIs are exercised with the pinned installed version.

Implementation details and grids are in the small registry/config modules. Local synthetic tests verify numerical behavior and isolation; native real-data results are pending manual upload and execution.
