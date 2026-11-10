# Frozen V10 model bank and interpretation

## Input blocks

The original host design, fiber linear/RCS basis and existing host-by-fiber terms
remain unchanged. Microbial CLR preprocessing uses the frozen loader. The new
nutrition source is `analysis/shared/food_day_qc_TRE_ONLY.parquet`, relative to the
project above the handoff. Only current energy-eligible day keys are admitted.
The lifestyle source is `analysis/shared/covariate_registry_TRE_ONLY.parquet`.

| Block | Additional numeric inputs | Categorical input |
|---|---|---|
| none | None | None |
| macro | Carb, fat, protein mean density | None |
| sodium | Macro plus sodium mean density | None |
| life | Walking days/week, moderate-activity days/week, sleep hours/day | Alcohol frequency |
| full | Macro, sodium and life numeric inputs | Alcohol frequency |
| macro_life | Full with sodium omitted | Alcohol frequency |
| joint | Full, replacing three macro means by means on joint-complete macro days | Alcohol frequency |
| fiber_days | Full, replacing three macro means by means on current fiber-eligible complete nutrient days | Alcohol frequency |
| distribution | Full plus macro SD, P10 and P90 (nine inputs) | Alcohol frequency |
| ratios | Full, replacing three macro means by two joint-day log-ratio coordinates | Alcohol frequency |

Density units are g/1000 kcal for macros and mg/1000 kcal for sodium. At least
three valid days are required per feature; population membership never depends on
the new fields. Joint/fiber-day sensitivity blocks hold sodium and lifestyle fixed.
They do not redefine the primary fiber exposure or establish exact CGM-day overlap.

For joint-day mean densities C, F, P, ratios are
`sqrt(2/3)*log(C/sqrt(F*P))` and `log(F/P)/sqrt(2)`. The fixed floor is
0.1 g/1000 kcal. Missing values stay missing. These are predictive representations,
not causal isocaloric substitutions. Complete sums cover recorded events; they do
not prove complete daily intake. No partial nutrient sums are used.

## Pipelines

| Group | Number | Purpose |
|---|---:|---|
| Five input levels x five matched learners/site choices | 25 | none, macro, sodium, life, full; HGB oral, CatBoost oral+gut, block ridge oral+gut, HGB host, CatBoost host |
| Full-context site comparisons | 4 | HGB oral+gut, CatBoost oral, block ridge oral/gut |
| Five sensitivity blocks x two anchor choices | 10 | Same HGB oral and CatBoost oral+gut grids |
| Spline ridge, RBF SVR, ExtraTrees, additive kernels x two sites | 8 | Full context with train-fitted microbial PCs |
| Zero-aware hurdle learners | 3 | Presence plus positive abundance, HGB oral/oral+gut and CatBoost oral+gut |
| CLR plus ecology CatBoost | 1 | Richness, entropy, Simpson diversity, dominance and evenness summaries |
| PC block-ridge null | 1 | Low-dimensional additive microbial background for explicit corrections |
| Matched fiber corrections | 6 | Below |
| Current-inner-OOF convex/ridge ensembles | 10 | Below |
| Total | **68** | **181 candidate settings before ensembles** |

Block ridge standardizes columns within the current training role and scales
each host/context/site block by the square root of its relative weight divided by
its feature count. Oral and gut contributions can receive different weights.
Spline ridge expands nonconstant continuous columns with quadratic B-splines,
four knots over the training range and linear extrapolation. It remains additive
between microbial blocks and host/context inputs.

The additive kernel is `RBF(host/context) + w*K_oral + w*K_gut` (only included
sites), with dimension-normalized linear kernels on standardized microbial PCs.
It has no direct fiber-by-microbe product. Ridge penalties are proportional to the
training sample size. All selection uses native-target inner squared error;
there is no outer-target early stopping or inverse-transformed target objective.

Spline and kernel implementation references: scikit-learn's
[spline/ridge example](https://scikit-learn.org/1.4/auto_examples/linear_model/plot_polynomial_interpolation.html)
and [kernel ridge documentation](https://scikit-learn.org/1.4/modules/generated/sklearn.kernel_ridge.KernelRidge.html).
The explicit block weighting and correction design here are project choices,
not a claim that these methods improve this dataset.

## Matched explicit corrections

| Correction | Selected background | Modifier |
|---|---|---|
| fiber_gut_linear | block_pc_balanced_full | Gut PC x linear fiber |
| fiber_oral_linear | block_pc_balanced_full | Oral PC x linear fiber |
| fiber_both_rcs | block_pc_balanced_full | Oral/gut PCs x linear and RCS fiber |
| fiber_gut_tree | cat_balanced_full | Gut PC x linear fiber |
| fiber_macro | block_pc_balanced_full | Three macro means x linear fiber; nonmicrobial comparison |
| fiber_gut_kernel | kernel_balanced_full | Gut PC x linear fiber |

Every workflow includes an exact-zero option. Microbial settings request PC
counts 5/10/20 per site with prespecified ridge penalties. Macro correction uses
three macro means, not PCs; its distinct penalties are 0.001/0.01/0.1, plus the
zero option. Identical macro configurations are evaluated only once. Diagnostics
record requested components and the actual number of modifier columns. Candidate
counts do not represent independent scientific tests.

For a selected background f0 with target scale s, interaction products U are
residualized against the SAME main design X using a training-only ridge projection
A (penalty n*0.01). Let Q=(U-XA)/column_SD. Fit a ridge correction to
`(Y-f0(X))/s`, then predict `f0(X)+s*Q*b`. This regularized residualization is
well-defined for wide backgrounds; it is not exact orthogonality or a causal
residualization guarantee. The null is held fixed; zero reproduces it exactly.

The CatBoost background already permits implicit interactions. Its correction
gain measures additional predictive correction; it does not compare every
possible interaction against a strictly interaction-free tree. The block-PC and
additive-kernel backgrounds are additive between microbial and host/context blocks.

Modifier SD is the added correction's contrast as fiber density moves from
10 to 15 g/1000 kcal, in mg/dL. It includes the corresponding change in projected
host fiber terms, is centered by the training modifier mean and is not the total
model's fiber contrast. SD is descriptive, not evidence of mechanism.

## Selection, fusion and reporting

Every outer fold repeats three household inner folds. Pool inner squared errors
to select a candidate, refit it on outer training households, then predict the
outer fold. PCA, imputation, categories, scaling and selection are role-fitted.

Five ensembles pools each use convex and ridge fusion: base (five original-input
anchors), macro (the same five choices with macros), full (same with all context),
diverse (all full-context atomic learners and corrections), wide (all new atomic
learners and corrections). Fusions use only current inner OOF predictions and
targets. Historical outer predictions enter evaluation after all new choices and
predictions are fixed. A fusion's total gain is not an explicit-interaction gain.

Paired input additions isolate macro, sodium and lifestyle changes at the matched
learner/grid level. Full vs sodium is the conditional lifestyle comparison.
Macro_life vs full is a sodium-omission sensitivity. Full vs macro ensemble gains
combine sodium and lifestyle additions. Matched host-vs-microbial and oral-vs-both
comparisons are also recorded separately. Conditional household intervals in
JSON treat fitted predictions as fixed and exclude adaptive model-search and
multiple-comparison uncertainty.

Both repeats share participants. R2 is pooled across OOF people per repeat; it is
not squared correlation or training R2. Negative scores are retained. A target
R2 >= 0.20/0.30 requires the SAME pipeline to cross the threshold in BOTH repeats
at full precision. There is no guarantee of reaching these values.

Recorded microbiome dates often follow CGM dates in the existing scout. Until
measurement timing is verified, results remain visit-level associational
prediction. New context or missingness may capture recording processes. Larger
total R2 alone cannot establish microbiome-specific dietary response, causal
effects, clinical usefulness or an independent confirmation result.
