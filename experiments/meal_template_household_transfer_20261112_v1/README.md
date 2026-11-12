# Frozen transfer of the repeated meal template interaction

Use the existing additional HPP household preparation. The archived
families are separate from the discovery training households. Their
outcomes were used in previous exploratory HPP analyses. The planned
comparison concerns transport across households within the same cohort.
Independent cohort prediction has been used in postprandial nutrition
research. [Zeevi et al.](https://pubmed.ncbi.nlm.nih.gov/26590418/)

Audit repeated FoodID combinations and carbohydrate variation first using
the previous outcome-blind template reconstruction. Save the selected meal
keys in TRE. Freeze the discovery template background, projection and
coefficients using the prior training-only microbial coordinates, then
verify the frozen receipt before opening the additional glucose outcomes.
Use the same template centering and energy- and protein-by-microbiome
controls on both sides of the matched prediction.

The primary evaluation is recorded-start two-hour mean net glucose.
Peak rise is the prespecified follow-up endpoint of interest, with two
timing shifts and the pre-meal value as diagnostics. Report absolute
R², paired household-bootstrap intervals and leave-one-household
sensitivity. The separately reported coefficient benchmark applies
the previous generic carbohydrate interaction under the same template
background and projection. For the frozen direction calibration,
refit nuisance terms only, and adjust across five endpoints.
The target is deviation from the person's observed template mean.

Run from the isolated TRE experiment directory with the pinned
interpreter and verified release.

```sh
python -B -m unittest discover -s .
python -B run_transfer.py --package STUDY_HANDOFF --commit COMMIT --stage audit --run support-1
python -B run_transfer.py --package STUDY_HANDOFF --commit COMMIT --stage audit --run support-1 --verify
python -B run_transfer.py --package STUDY_HANDOFF --commit COMMIT --stage freeze --run frozen-1
python -B run_transfer.py --package STUDY_HANDOFF --commit COMMIT --stage freeze --run frozen-1 --verify
python -B run_transfer.py --package STUDY_HANDOFF --commit COMMIT --stage score --run validation-1
python -B run_transfer.py --package STUDY_HANDOFF --commit COMMIT --stage score --run validation-1 --verify
```

All selections, coefficients, losses, and detailed results remain inside TRE.
