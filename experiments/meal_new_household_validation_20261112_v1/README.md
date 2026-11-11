# Held-household transfer of the meal carbohydrate interaction

This experiment tests the previously selected joint oral and gut microbiome
interaction on additional households in the frozen HPP discovery index. It
reuses the original 10 g carbohydrate contrast, diet and timing covariates,
and mean net 0–120 minute glucose response. Peak, offset-start net response
and pre-meal glucose provide fixed secondary checks.

The training-only stage fits the diet background and a linear carbohydrate
interaction with the first five CLR principal components of each site. It
uses the original weighting, residualization and ridge penalty of 0.03.
Microbiome reference features, centering, scaling, principal axes and
regression weights are estimated from archived training households. The
stage saves the frozen marker coordinates and coefficients inside TRE before
any additional participant's point-level CGM is opened.

New households are drawn from the existing discovery index, excluding all
households represented in the original eligibility roster. Preparation
reproduces the earlier outcome-blind cohort screen using original food
episodes, verified CGM time bounds and the minimum of five meals on three
dates. This screen selects people. Their meal outcomes then use the original
food-event linkage, selected CGM connection and verified point file, isolated
meal window, and quality rules at each of three time offsets. At least five
full-offset windows and three dates are required for analysis. Batch and
final files are verified by SHA256. The reserved confirmation cohort is
outside the scope of all three stages.

The primary validation is the paired change in weighted held-out (R^2)
after adding the frozen interaction to the frozen background. Intervals use
1,999 paired household draws. A separate calibration test uses the frozen
direction and refits only nuisance terms for a clustered association in the
new households. The five calibration tests form one Holm family. Meal
outcomes and features are centered within person, as in the original task.
The prediction target is a deviation from the participant's observed mean
over eligible meals. Generalizability is evaluated within the HPP discovery
registry in households outside the original eligibility roster.

Run from the installed experiment directory using a verified commit.

```sh
python -B -m unittest discover -s .
python -B frozen.py freeze --package /study/package --commit COMMIT --run frozen-1
python -B prepare_holdout.py --package /study/package --commit COMMIT --frozen frozen-1 --run prep-smoke-1 --smoke
python -B frozen.py check-prep --run prep-smoke-1
python -B prepare_holdout.py --package /study/package --commit COMMIT --frozen frozen-1 --run prep-discovery-1
python -B frozen.py evaluate --package /study/package --commit COMMIT --frozen frozen-1 --prepared prep-discovery-1 --run validation-1
python -B frozen.py verify --run validation-1
```

Use fresh run identifiers. A partially prepared run can resume with the same
registration and `--resume` after checking the prior process and its
checkpoints. Model, glucose, household losses and source records stay in TRE.
