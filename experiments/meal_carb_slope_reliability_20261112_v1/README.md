# Repeatability and precision weighting of meal carbohydrate slopes

Check how much of the between-person variation in the observed carbohydrate
slope repeats on different dates. Compare a precision-weighted microbial
estimate with the same marker learned using equal person weights. Earlier
meal responses also supply an empirical-Bayes history predictor, with matched
priors that use either the training mean or the microbial profile.
Repeated controlled meals have shown within-person CGM variability.
[Duplicate-meal CGM study](https://pubmed.ncbi.nlm.nih.gov/39755436/)

Use the verified archived meal windows, main terms, household partitions, and
recorded 10 g carbohydrate contrast at fixed energy and protein. Keep the
chronological halves with at least four dates overall and four meals per half.
Require squared within-period centered dose information of at least 4 in
10 g units in each half. Admission uses food and dates. Project the centered
meal residuals onto centered dose for each person and period. Fit nuisance
main effects separately by period. The outer test households are excluded
from all nuisance fits. Use the other frozen fold labels within outer training
to cross-fit the slope targets used for training.

Estimate shared variance from the covariance of training slopes across
periods, bounded between zero and 95% of the smaller marginal variance.
Use the remaining variance and inverse dose information to model period
noise. Fix the marker ridge penalty at 0.03. For the history predictor use
the same individual shrinkage weight for both priors. Learn the marker from
source-period slopes. Compare its held-out predictive increment for the
opposite period's residual meal outcome and observed slope. Both partitions,
both directions, net response, peak, and pre-meal glucose are reported.
The pre-specified main comparison uses joint oral/gut PCs, precision weights,
earlier to later, two-hour net response, and the matched precision-weighted
intercept. The literature-panel representation is supplementary.

Report fold-out correlations and paired household bootstrap intervals
conditional on the fitted models. The observed residual slope can reflect
diet choice, measurement error and changing context. A positive correlation
captures repeatable conditional association, and the repeated discovery
partitions share the same underlying people. The reverse history direction
is a robustness comparison using the observed later-period predictor.

Run `python -B -m unittest -q test_slopes` and then
`evaluate.py --package HANDOFF --commit COMMIT --run ID --smoke` using the
verified V12 interpreter. Verify the one-fold smoke receipt and run the ten
folds with a fresh identifier. Keep complete real outputs in TRE.
