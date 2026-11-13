# Date-separated fiber and gut Shannon associations

Test whether the between-person fiber association in the matched-day
experiment is still modified by gut Shannon when diet and CGM outcomes
come from disjoint dates. Reuse the exact registered matched days,
discovery household partitions, equal person weighting, and the raw
Shannon Q75-minus-Q25 contrast for a 10 to 15 g/1,000 kcal difference.
[Prior matched-day experiment](../diet_fiber_within_between_20261113_v1/README.md)

For each person, require at least four eligible dates. Use the first
floor(n/2) dates as early and the remaining dates as late, in chronological
order. Both halves have at least two dates. Early-to-late uses early mean
fiber, energy and macronutrient shares and late mean CGM glucose. The
reverse uses the opposite means. Compute early-only, late-only and
all-date contemporaneous estimates in these same people for comparison.
Do not include outcome-half diet in a cross-date fit. The host covariates,
microbial ecology terms, five CLR PCs per site, fiber main-effect hinge,
and age/sex by dose terms follow the prior between-person model.

The designated effect uses a linear Shannon interaction, with
early-to-late as the primary direction and late-to-early as a temporal
sensitivity check. Report pointwise household CR1 intervals and a
two-direction Bonferroni family interval. Use aligned household score
vectors for the paired difference versus the all-date linear estimate.
The fixed 12.5 g/1,000 kcal interaction hinge is secondary and its
intervals are pointwise. Report source-period dose support in both
Shannon tails and correlations of fiber and glucose between periods.

For prediction, score both directional models on the same held-out
households in each archived partition. Fit CLR, PCs and marker scaling
in the training households. Use the prior fixed penalty of 0.03 and
1,999 paired household resamples. Associations use a pooled fit for
each pairing. Both directions use the same participants and discovery
cohort. Reversing the dates evaluates stability of a between-person
association. It does not test the effect of changing an individual's
diet or establish temporal causality.

```sh
python -B -m unittest -q test_date_split
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --verify
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
python -B run.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1 --verify
```

Run with the pinned TRE interpreter. Keep all real outputs inside TRE.
