# Meal-log-anchored CGM reconstruction

This stage constructs the meal-level discovery input for testing whether the within-person carbohydrate–glucose association differs with microbiome profiles. The [HPP CGM prediction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13446328/) motivates measuring a two-hour response separately from the visit mean. The source timestamps record food logging. Individual ingestion times and logging delays have no independent verification in these tables.

Food timestamps must agree with their local wall-time fields to within one minute. Group events up to 30 minutes from an episode's first timestamp. Use only episodes on eligible energy days, with full nutrient data, no repeated-food/time or duplicate flags, and at least 135 minutes of isolation from the preceding and following episode. Match only the visit's already selected W14 CGM connection. Each raw point file is checked against the W09 fragment manifest before processing. Apply the same verified CGM window implementation used in the earlier HPP meal pipeline, including a pre-window median and conservative interpolation and gap criteria. Store net and positive incremental areas divided by 120, peak rise, baseline and pre-window slope. Compute the same windows at the registered log time and at ±15 minutes to assess clock sensitivity. Time shifts use the same detected episodes. The later analysis can compare all three clocks on exactly the same meals.

The frozen Experiment 2 discovery roster is the only roster used. Admission requires five qualifying meals across at least three calendar dates. The model stage uses the same meals with complete windows at all three clock offsets and re-applies the five-meal and three-day minimums on that subset.

## Within-person models

The primary test is the two-hour time-averaged net increment at the logged timestamp, with a 45% to 55% carbohydrate share contrast at fixed protein share and recorded energy. The modifier is the fixed seven-taxon abundance sum from the earlier Westlake adaptation. A second exposure uses a 10 g carbohydrate difference at fixed protein grams and recorded energy. The model includes participant fixed effects with equal total weight per participant, dietary main terms, local time, weekday, prior episode gap, and dose by age and sex. Post-window models additionally adjust for pre-window glucose and slope. Pre-window glucose is examined as a temporal check of residual diet selection or timing.

Compare peak rise, net increment at both alternative clocks, and oral, gut and combined PCs and Shannon representations. Report household CR1 omnibus tests with Holm adjustment for all 60 models, signed quartile contrasts for scalar modifiers, and matched prediction results from two five-fold household partitions. Microbial transformations and coefficients use training households. The fold-out target is the within-person deviation in the recorded-window outcome.

## TRE deployment

Deploy this directory and the committed `meal_cgm_exploration/hppmeal/cgm.py` from the same release archive. Put that helper in this experiment as `pinned_cgm.py`; its Git-blob SHA-256 is pinned in the configuration. Run the synthetic checks before the 48-participant smoke run and then the full preparation.

```bash
h=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1
p="$(cat "$h/experiments/diet_effect_modification_20261109_v12/.python_path")"
cd "$h/experiments/meal_carbohydrate_modification_20261112_v1"
"$p" -B -m unittest -q test_prepare
"$p" -B prepare.py --package "$h" --commit FULL_COMMIT --run smoke-1 --smoke
"$p" -B prepare.py --package "$h" --commit FULL_COMMIT --run discovery-1
"$p" -B -m unittest -q test_analyze
"$p" -B analyze.py --package "$h" --commit MODEL_COMMIT --prepared discovery-1 --run models-smoke --smoke
"$p" -B analyze.py --package "$h" --commit MODEL_COMMIT --prepared discovery-1 --run models-1
```

The full window table, batch checkpoints and logs stay in the study. A completed run has a marker with hashes for the registration, summary and private window table. Use `--resume` after checking the previous process and lock; registered source bytes must match.

## Transfer across recording days

This discovery sensitivity focuses on the selected 10 g carbohydrate contrast at
fixed recorded energy and protein. Split each participant's sorted dates at the
median date rank and require at least four dates overall and four meals in each
period. Keep dates intact. Evaluate the recorded-clock net response and peak
rise, with pre-meal glucose as a temporal check. Recenter all model terms within
participant and period to keep the within-person deviation target.

Use the archived model, main terms, fixed interaction penalty and two household
partitions. Within each fold fit on the early dates of training households and
score the late dates of held-out households, and vice versa. Fit each microbial
representation on training households. Compare the joint and single-site
interaction blocks against the same additive background. Report 1,999 paired
household bootstrap draws per repeat and direction, conditional on the fitted
predictions. These are robustness checks in the selected discovery population.

Run `test_time_transfer` and then `time_transfer.py` with the package path, the
full release commit and a new run identifier. Use `--smoke` with a separate
identifier to check both time directions in the first fold before the full run.
