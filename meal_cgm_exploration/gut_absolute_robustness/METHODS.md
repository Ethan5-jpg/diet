# Exploratory absolute glucose and meal-support sensitivity: methods

These extensions were specified after reviewing the original four-candidate findings. They preserve the original main analysis and are not described as preregistered primary analyses. All four previously selected microbial candidates and both original common samples are retained. Microbiome timing is treated as same-visit metadata and does not determine eligibility. No additional fixed 14-day completeness criterion is imposed.

## Frozen source and analysis units

The completed original gut-meal run supplies hash-verified participant-level summaries, meal membership, sample counts and all 48 original estimates. Participant and meal identifiers, strict-sample flags, meal counts and common-sample hashes are checked. Date counts are inherited from the original hash-verified participant summaries; the membership table does not contain per-meal dates, and this extension does not claim to recalculate them from that table. The original 48 coefficients, standard errors, pointwise confidence intervals, p values and q values must reproduce before extension results are released.

The existing eligible meals require the original local CGM and logged-intake separation criteria. Each original participant contributes at least five eligible meals on three distinct dates. Outcomes and meal-context variables were already averaged within participants; each participant receives equal regression weight. Exposure CLR-Z values and all original summaries retain their frozen scales.

## A: absolute 0–120-minute glucose under common adjustment

For the same eligible meals, the participant mean absolute post-meal glucose is defined as mean pre-meal baseline plus mean net incremental area divided by 120 minutes. The pre-meal baseline is the median of valid readings in the original 30-minute window. The absolute outcome is a time-averaged curve level, not the 120-minute measurement. Signed net areas, including negative values, are retained.

Pre-meal glucose, net increment and absolute post-meal mean are each regressed on one microbial candidate and the same covariates. The common adjustment set is the original pre-meal model: age, sex, education, smoking, usual sleep, usual physical activity, vitamin and hormone use, BMI, CGM device, corrected hPDI, participant NOVA4, daily energy and alcohol, mean meal carbohydrate, mean squared meal carbohydrate, mean meal energy, hour sine/cosine, previous-intake gap, observation day and duration. It excludes pre-meal glucose and pre-meal slope. The four taxa are not mutually adjusted. The eight pre-meal coefficients reproduce original references; the net-increment estimates have a different adjustment set from the original baseline-adjusted main model.

Each outcome is fitted directly using the original OLS implementation, HC3 standard errors and residual-degrees-of-freedom t intervals. The coefficient identity beta(absolute)=beta(pre)+beta(net) under identical X is an algebraic implementation check, not a mediation test, independent replication or an explained-fraction estimate. Confidence intervals are not added. An absolute-outcome model additionally adjusting for baseline would give the same focal coefficient as the corresponding adjusted-net model, so this mathematical equivalence is not counted as a further finding.

Within each sample, all twelve candidate-by-outcome tests form a single exploratory BH family, reported as q_panel12. This family differs from the original meal families. The original main net-increment conclusion remains unchanged.

## B: minimum meal/date support

Participants are evaluated at fixed joint thresholds of at least 5 meals/3 dates, 10 meals/5 dates and 20 meals/7 dates. Higher thresholds select participants only; all their original eligible meals, means and covariates remain unchanged. The original 5/3 results are retained as references. At each higher threshold, all four taxa are evaluated against the three original meal outcomes, using the original outcome-specific adjustments. Net increment and peak rise retain adjustment for pre-meal level and slope; the pre-meal outcome does not adjust for itself or slope.

Within each threshold and sample, BH correction retains four original net-increment tests and eight secondary pre-meal/peak tests. Supplementary q_meal12 covers all twelve meal tests. Samples overlap. These corrections do not claim global FDR control across all samples, thresholds and both analysis blocks. Retention counts and retained/excluded participant characteristics are reported, with standardized mean differences based on the square root of the average of group variances. Selection descriptions do not establish removal of bias. Effect/interval changes can reflect both participant selection and meal support; a difference in significance is not a formal effect-difference test.

When both retained and excluded groups are constant for a characteristic but have different values, the standardized mean difference is undefined because its denominator is zero. This is explicitly flagged as zero-variance separation and prioritized in the summary, rather than omitted from the comparison. Identical constants and inadequate group sizes have separate undefined-status labels. These descriptive labels do not change regression estimates.

All planned specifications are retained. Fewer than 50 participants, inadequate variation, singular designs or insufficient residual degrees of freedom produce NOT_ESTIMATED with a reason. Such rows remain in planned BH denominators using the original missing-p convention; their displayed effects and q values remain missing. No threshold is relaxed based on results. Near 0.05, displayed q values retain enough digits to preserve which side of the boundary they occupy.

The analyses describe selected-candidate, same-cohort conditional associations. They do not establish causal ordering, microbiome mediation or independent validation. Original full-precision tables remain on the server; terminal displays are rounded summaries.
