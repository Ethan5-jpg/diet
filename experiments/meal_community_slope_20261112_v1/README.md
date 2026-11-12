# Community-wide predictors of repeated meal carbohydrate slopes

PREDICT 1 used repeated meals to assess the consistency of postprandial
responses. MiRKAT shows how microbial composition can be represented by
community similarities. Here a community kernel predicts a within-person
carbohydrate slope measured on separate dates in held-out households.
[PREDICT 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8265154/)
[MiRKAT](https://pmc.ncbi.nlm.nih.gov/articles/PMC4570290/)

Reuse the archived meal windows, dose-information eligibility, household
partitions and early/late period assignments of the repeated slope analysis.
The contrast is 10 g of logged carbohydrate at fixed recorded energy and
protein, conditional on the archived model. Reuse the period-specific
background residuals and cross-fitting. In each outer fold, both period
nuisance fits exclude the held households. The microbial slope predictor
uses only the source-period slope and dose information from training
households. Evaluating the target-period residuals uses nuisance fits from
the corresponding target period in training households.

The baseline is a weighted intercept. The PC comparator uses the top five
training CLR components per site, standardized in training. The community
model equally averages four centered kernels: oral and gut CLR dot products
and oral and gut radial kernels of the squared CLR distance. Each radial
bandwidth is the median squared distance between distinct training rows.
Weights are average centered squared dose in a person-period, capped at
the source-training 95th percentile and normalized to sum to one. Each
kernel has weighted trace one after weighted centering. Set the PC and
community ridge penalties to 0.03 on this scale. Include community
sensitivity penalties 0.003 and 0.3. Report them as sensitivity estimates.

The primary comparison is the held household meal-level prediction gain
for community versus PC in the early-to-late direction for the recorded
start 0–120 minute average net glucose. Report the matching intercept
comparison, reverse transfer, slope prediction, peak, and pre-meal glucose.
Use paired household intervals with 1,999 bootstrap draws conditional on
the out-of-fold predictions. A separate diagnostic compares the observed
early/late slope covariance and the change in covariance after removing
the two period-specific held-out predictors. The covariance difference
can be negative. It describes recurrent conditional association under
these models and does not identify causal effects of a microbial profile.
The two partitions share the same people. This is a discovery experiment.

```sh
python -B -m unittest -q test_community
python -B run_community.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B run_community.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --verify
python -B run_community.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
python -B run_community.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1 --verify
```

Use the registered TRE interpreter. Store real scores, fold receipts and
results inside the study. Verification checks hashes and reproduces the
summary from the household losses.
