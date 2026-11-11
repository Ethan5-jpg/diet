# UPF–glucose modification with shared food classification uncertainty

This discovery comparison uses the frozen person-level Experiment 2 cohort, restricted to people with at least three eligible energy days and valid UPF bounds. The endpoint is raw visit-level mean CGM glucose in mg/dL. The frozen cohort, two household partitions and food catalogue row alignment are verified before fitting.

## Exposure assumptions

We reuse the W20 grouped Bayesian-bootstrap recipe with fixed category shrinkage. Each outer training fold supplies its own reference FoodIDs, with a single evidence weight per provenance group. The label for an unknown FoodID is drawn once per catalogue draw and shared across all people. The design integrates the linear and quadratic dose terms separately. A model-expectation plug-in and the recorded lower and upper bounds are comparison specifications. The bound fits represent two extreme assignments, and their coefficients do not provide identification bounds for the association. Thirty-two shared catalogue scenarios show sensitivity of the primary contrast to the label draws. This spread describes exposure-model sensitivity and is separate from the conditional confidence interval.

The exposure is equal-day average UPF energy fraction. A symmetric 10-percentage-point contrast around the training calibrated median gives the effect scale. This is an observational contrast with fixed covariates.

## Models

Both backgrounds include age, sex, smoking, mean recorded energy, interval width, dose terms, age/sex by dose, and oral and gut CLR/PC and ecology main effects. The joint background additionally includes fiber, macronutrient and sodium densities, activity and sleep, using training-only imputation. Each candidate adds a matched microbiome-by-dose block. Gut and oral Shannon, CLR PCs, and combined-site PCs are evaluated with fixed linear and selected quadratic interaction bases.

The primary direct contrast is the difference in the 10-point UPF association between the 75th and 25th percentiles of gut Shannon using the integrated joint background. The 64 direct tests use household CR1 standard errors and Holm adjustment. Pointwise intervals condition on the fitted exposure design. Each of the 128 prediction specifications has a matched no-interaction baseline. Microbiome preprocessing and catalogue calibration use training households. No route is selected by held-out performance. The two repeats reuse the same participants.

A nutritionally matched crossover found time-dependent glucose differences between processed-food meals. It motivates evaluating timing and exposure definitions separately in HPP. [Metabolic and neural responses to ultraprocessed foods](https://www.nature.com/articles/s42255-026-01619-4)

## Run inside TRE

```bash
h=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1
p="$(cat "$h/experiments/diet_effect_modification_20261109_v12/.python_path")"
cd "$h/experiments/diet_upf_modification_20261112_v1"
"$p" -B -m unittest -q test_upf
"$p" -B upf.py --package "$h" --commit FULL_COMMIT --run smoke-1 --smoke
"$p" -B upf.py --package "$h" --commit FULL_COMMIT --run discovery-1
```

Verify the release hashes before use. Store logs and the full summary in the study. Completion requires the ten household-fold receipts and the verified summary marker. Resume only the matching unfinished run after checking its process and lock.
