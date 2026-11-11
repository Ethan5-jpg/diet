# Meal context and microbiome modification

Test whether the recorded 10 g carbohydrate contrast at fixed energy and
protein varies with joint oral and gut microbiome PCs differently across
two eating contexts. PREDICT compared the response to an identical meal
at breakfast and lunch. A separate analysis observed carbohydrate
interaction with host insulin resistance in structured meal contexts.
[PREDICT](https://pmc.ncbi.nlm.nih.gov/articles/PMC8265154/)
[CGMacros secondary study](https://doi.org/10.3390/metabo16090671)

Use the archived qualified discovery meals. Morning meals are logged
from 05:00 to before 10:00 with a previous recorded gap of 480–960
minutes. Daytime meals are logged from 11:00 to before 21:00 with a gap
of 135–360 minutes. These labels combine recorded time and previous
logged gap. Fasting and actual ingestion time are unverified. Restrict
both to 15–60 g carbohydrate based on outcome-blind overlap. Require
at least two dates and a dose span of at least 10 g in each context
per person.

Center the original meal design within participant and context and
give each participant-context group equal weight. Fit context-specific
coefficients for the original main terms. Compare the same baseline,
one common joint-PC interaction, and an extension that allows the
interaction to differ between contexts. Use the fixed 0.03 penalty for
the microbial terms and training-only microbial preprocessing. The
primary contrast is the fold-out net response with the context-specific
model versus the common interaction, across both contexts. Also report
estimates by context, peak rise, and the pre-meal temporal control.
Pooled direct tests of any modification and the difference use a
household-clustered Wald test, with Holm adjustment over six tests
per panel.

Repeat the pooled comparison on the balanced date-transfer subset.
For temporal transfer require both contexts to have at least two dates
and a dose span of at least 5 g in each chronological half. Center
within person, context, and period and keep households separated.
Compare the two directions on the same people. The clinical sensitivity
compares unadjusted and adjusted pooled models on the same people with
prior HbA1c and BMI, using the previously audited matching rules.
The stricter date-transfer support is evaluated in the broad panel.
Intervals resample matched households conditional on the fitted models.

Run `python -B -m unittest -q test_context` using the verified V12
interpreter. Use `evaluate_context.py --package HANDOFF --panel broad
--run ID --commit COMMIT --smoke` and repeat with `--panel clinical`
and a distinct identifier. After verifying both receipts, run the
ten folds using new identifiers. Keep complete real artifacts in TRE.
