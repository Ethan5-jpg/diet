# Gut Shannon and fiber on matched days and people

Experiment 2 v2 measures modification of the visit-level fiber association
at fixed doses. Combine its gut Shannon contrast with the earlier matched
day design. Compare variation within a person against variation across
people using the same qualified person, visit and calendar days. Both
outcomes use untransformed matched-day mean CGM. The between-person
exposure is the arithmetic mean of those days' fiber densities. This
between estimate uses its own matched-day inputs and nuisance adjustment.
[Experiment 2 v2](../experiment2_Effect_Modification/v2/README.md)
[Matched day design](../diet_daily_modification_20261112_v1/README.md)
[Within- and between-person longitudinal models](https://pmc.ncbi.nlm.nih.gov/articles/PMC3059070/)

Use the archived discovery population and five-fold household assignments.
The original daily eligibility requires three or more matched days. Weight
people equally. Fit person-centered daily regressions and person-mean
regressions separately, with a common raw gut Shannon interquartile contrast
for interpretation. Retain the existing daily nuisance model and add dose
hinge main effects and host-by-dose terms. The person-mean model includes
age, sex, mean energy and macronutrient shares, the dose main effects and
age/sex interactions with them, both sites' ecology and five CLR PCs per
site. The mean outcome and the centered outcome have separate R² scales.

The designated comparison is the household-clustered difference between
within-person and between-person Shannon modification for 10 to 15 g
fiber per 1,000 kcal in the linear interaction model. The common nuisance
model allows a fixed main-effect hinge at 12.5 in all comparisons. A
secondary interaction adds the hinge and reports 10 to 12.5, 12.5 to 15,
their difference, and the full 10 to 15 contrast. Secondary intervals are
pointwise. Inference conditions on the fitted feature basis and uses
aligned household influence vectors for the difference. Report support in
both Shannon tails, including the number of people whose observed daily
doses span 10 and 15. These conditional associations may differ because
within- and between-person measurement and confounding differ.

Compare the two fixed interaction bases with the identical fitted
baseline in each household fold. Scale the marker in training and use
the original fixed ridge penalty of 0.03. Fit CLR and PC axes in the
training households. Report paired 95% intervals from 1,999 household
draws, conditional on the predictions. The within-person outcome uses
the person's observed mean glucose as its scoring reference. This is an
exploratory comparison in the discovery cohort.

```sh
python -B -m unittest -q test_estimate
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --verify
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1 --verify
```

Use the registered TRE interpreter. Retain real results and fold payloads in TRE.
