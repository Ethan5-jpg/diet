# Meal-log-anchored CGM reconstruction

This stage constructs the meal-level discovery input for testing whether the within-person carbohydrate–glucose association differs with microbiome profiles. The [HPP CGM prediction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13446328/) motivates measuring a two-hour response separately from the visit mean. The source timestamps record food logging. Individual ingestion times and logging delays have no independent verification in these tables.

Food timestamps must agree with their local wall-time fields to within one minute. Group events up to 30 minutes from an episode's first timestamp. Use only episodes on eligible energy days, with full nutrient data, no repeated-food/time or duplicate flags, and at least 135 minutes of isolation from the preceding and following episode. Match only the visit's already selected W14 CGM connection. Each raw point file is checked against the W09 fragment manifest before processing. Apply the same verified CGM window implementation used in the earlier HPP meal pipeline, including a pre-window median and conservative interpolation and gap criteria. Store net and positive incremental areas divided by 120, peak rise, baseline and pre-window slope. Compute the same windows at the registered log time and at ±15 minutes to assess clock sensitivity. Time shifts use the same detected episodes. The later analysis can compare all three clocks on exactly the same meals.

The frozen Experiment 2 discovery roster is the only roster used. Admission requires five qualifying meals across at least three calendar dates. This release prepares inputs and audits window coverage. The model specification will report within-person carbohydrate contrasts, separate main effects from microbiome modification, and include pre-window outcomes as a temporal check.

## TRE deployment

Deploy this directory and the committed `meal_cgm_exploration/hppmeal/cgm.py` from the same release archive. Put that helper in this experiment as `pinned_cgm.py`; its Git-blob SHA-256 is pinned in the configuration. Run the synthetic checks before the 48-participant smoke run and then the full preparation.

```bash
h=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1
p="$(cat "$h/experiments/diet_effect_modification_20261109_v12/.python_path")"
cd "$h/experiments/meal_carbohydrate_modification_20261112_v1"
"$p" -B -m unittest -q test_prepare
"$p" -B prepare.py --package "$h" --commit FULL_COMMIT --run smoke-1 --smoke
"$p" -B prepare.py --package "$h" --commit FULL_COMMIT --run discovery-1
```

The full window table, batch checkpoints and logs stay in the study. A completed run has a marker with hashes for the registration, summary and private window table. Use `--resume` after checking the previous process and lock; registered source bytes must match.
