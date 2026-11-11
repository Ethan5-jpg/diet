# Literature-anchored gut carbohydrate sensitivity score

[Fu, Gou et al.](https://doi.org/10.1016/j.ebiom.2024.105483) summarized ten Fig. 5b taxa by summing their relative abundance. Seven map uniquely to the HPP species dictionary. The names and mapping are frozen in `config.json`. The mapping of Clostridium bolteae to Enterocloster bolteae follows the published reclassification. Escherichia unclassified and the two unmatched strain labels are absent from the HPP sum. This is a seven-taxon adaptation of the published score. [Haas and Blanchard](https://pubmed.ncbi.nlm.nih.gov/31782700/)

The primary estimate is the difference in the 45% to 55% carbohydrate-for-fat association between the 75th and 25th percentiles of the seven-taxon score in a participant fixed-effects model. This uses the same discovery cohort, CGM mean, common days, covariates and carbohydrate and fiber definitions as the pinned daily analysis. The between-person analysis uses person averages from those exact days and includes the score and its square as main effects. It answers a separate observational question. Sensitivities use a log-transformed sum and a sum without E. coli. The three encodings are tested against carbohydrate, fiber and both together on both time scales.

The direct family comprises 18 household-clustered tests with Holm adjustment. Report signed contrasts with pointwise 95% intervals. Two family holdout partitions compare matched regressions with and without the interaction block at a fixed penalty. Score scaling uses training households. The within-person denominator is the held-out centered outcome sum of squares. For between-person evaluation the reference is the corresponding training-person mean.

Run inside the HPP study alongside the exact pinned daily release:

```bash
h=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1
p="$(cat "$h/experiments/diet_effect_modification_20261109_v12/.python_path")"
cd "$h/experiments/diet_carb_external_score_20261112_v1"
"$p" -B -m unittest -q test_external
"$p" -B external.py --package "$h" --commit FULL_COMMIT --run smoke-1 --smoke
"$p" -B external.py --package "$h" --commit FULL_COMMIT --run discovery-1
```

Verify the release and pinned dependencies before running. Keep logs and the full summary in the study. Completion requires all ten holdout receipts and the summary marker. Use `--resume` only for the same unfinished run and after checking its recorded process and lock.
