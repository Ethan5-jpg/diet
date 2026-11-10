# Diet–microbiome experiments

The V6–V14 work extends the existing V1–V5 experiments. Start with the
[detailed Chinese cumulative summary](post_v5_experiment_summary_20261109/summary.md),
[428-row predictive workflow inventory](post_v5_experiment_summary_20261109/model_inventory.md),
and [effect-modification results](post_v5_experiment_summary_20261109/modification_inventory.md).
The 428 count covers eight predictive batches; V12 inference and its three-route
functional follow-up are recorded separately, not counted as unique algorithms.

## New experimental modules

| Module | Main focus | Final aggregate observation |
|---|---|---|
| [V6](diet_model_batch_20261107_v6/) | Boosting, kernels, explicit interactions and fusion | Fusion R2 0.114660 / 0.115187 |
| [V7](diet_model_expansion_20261107_v7/) | Additive models, sparse representations, mixture and factorization | Ridge stack 0.112019 / 0.119128 |
| [V8](diet_model_overnight_20261108_v8/) | NMF, supervised representations and complementary fusion | Ridge joint 0.120803 / 0.121851 |
| [V9](diet_model_adaptive_20261108_v9/) | Compositional representations, conditional fusion and daily diet summaries | Convex wide 0.119491 / 0.126979 |
| [V10](diet_model_context_20261108_v10/) | Macro-nutrient/lifestyle context and matched input additions | Ridge diverse 0.126227 / 0.137266 |
| [V11](diet_model_distribution_20261109_v11/) | Daily diet distributions, shrinkage and historical/new pools | Positive combined 0.130555 / 0.139855 |
| [V12](diet_effect_modification_20261109_v12/) | Prespecified ecological effect-modification analysis | Primary global P 0.3769; split sensitivity 0.1278 |
| [V12 functional follow-up](diet_effect_modification_20261109_v12/functional_followup_20261110_v1/) | Three fixed oral-pathway interaction encodings | No route improved prediction in both splits |
| [V13](diet_model_complementary_20261109_v13/) | SIR/SAVE, sparse GP, hinge bases and distributional objectives | Positive combined 0.129425 / 0.139372 |
| [V14](diet_model_structural_20261110_v14/) | Fourier/gradient directions, CCA, local/kernel regressors and nonlinear stacks | Positive combined 0.131369 / 0.139524 |
| [Data scout](data_scout_20261108/) | Read-only schema, record coverage and nutrient-day checks | Aggregate context only; no new outcome fitting |

## Interpretation and execution

Most predictive comparisons use the same 3343 discovery participants and two
household-grouped five-fold splits, with three inner folds. Repeats share people.
R2 is the pooled held-out score for fiber_density -> D_Glu (mg/dL), not an average
of all original exposure/outcome combinations. The functional follow-up uses 3337
people and must not be ranked directly against the 3343-person historical scores.
Prediction improvements do not establish microbial effect modification or causality.

Each module retains its original code, configuration, tests, negative findings,
failed-run notes and server commands. Real execution requires the prepared native
server handoff, its existing dependencies and private historical caches. This GitHub
source archive does not contain participant inputs, fitted models or predictions.
Do not batch-launch every historical script or overwrite a completed native run.

## Publication contents

This update adds code and text records only. Original screenshots remain local;
historical image filenames, links and hashes refer to that local evidence archive.
They are not claims that the images are hosted here. The V13/V14 repository-copy
upload manifests exclude those screenshots so their file checks remain usable;
original local manifest hashes and exact changes are in
[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json).
Runtime source/configuration bytes and the existing V1–V5 directories are unchanged.
