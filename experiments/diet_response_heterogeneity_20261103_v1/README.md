# Fiber–D_Glu discovery interaction pilot

Scientific target: association modification for a common fiber contrast by oral and gut microbiome features. This is an observational development experiment, not a causal treatment-effect estimator.

## Fixed interface and comparator

Use the existing collaborator_handoff_20261102_v1 fold loader. Keep current 70/30 roles, five household folds, fold-local preprocessing, phenotype rules and dual-site eligibility. Expected CV: 3343 people / 3059 households; fold validation counts 661, 677, 668, 665, 672. Confirmation outcomes are never accessed.

All candidate models retain the same host design, five unpenalized PCs per site and full dual-site CLR main-effect ridge (mean-loss lambda=1). The no-interaction prediction and joint CLR interaction at lambda=10 must reproduce archived v2 fold predictions to 2e-8 before interpretation.

## Prespecified candidate families

For oral-only, gut-only and joint interaction blocks, compare ridge with CLR, PC5 and PC10; elastic-net CLR interaction; and low-dimensional nonlinear interaction using linear and restricted cubic fiber bases times PC5. There are 15 model families and 84 settings per fold, plus one matched main-effect null. The bounded ridge grid is 0.01, 0.1, 1, 10, 100. Elastic-net alpha is 0.2 or 0.8; lambda/lambda_max is 1, 0.3, 0.1 or 0.03. The profiled background retains its ridge penalty when estimating sparse interactions.

Select within each family by minimum mean five-fold MSE, ties favor stronger shrinkage. These same-fold tuning results are development estimates and cannot establish independent performance. No confirmation subset is used for selection.

## Outputs and interpretation

Report R2/MSE versus the common null, household-clustered paired losses, original-unit centered microbiome contributions for the fixed 10→15 g/1000 kcal fiber contrast, and exploratory slope calibration. Contrast display requires marginal 5–95% training support; this does not prove joint positivity. Host conditions are held fixed in the microbiome contribution. Nonlinear score calibration is a linear projection diagnostic, not full nonlinear validation. Cluster intervals omit model-selection and cross-fold shared-training uncertainty.

Individual outputs, fitted coefficients and input receipts stay in TRE. The source repository contains code, configuration, synthetic checks and aggregate operational records only.

## Run in TRE

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python verify_synthetic.py
/home/ec2-user/miniconda/envs/pheno/bin/python run_pilot.py --package /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1 --output /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_response_heterogeneity_20261103_v1/results --code-commit DEPLOYED_COMMIT
```

The result directory must be new. Existing inputs, archived code and previous results are preserved. Writes use fresh sequential NPZ files compatible with the mounted filesystem. Synthetic tests cover independent joint matrix and constrained optimization solutions, held-out outcome invariance, zero interaction, rank-deficient host columns, contrast identity, and serialization/no-overwrite behavior.
