# Antecedent clinical state and meal carbohydrate modification

Compare the association of a 10 g within-person carbohydrate contrast
at fixed recorded energy and protein across host and microbial profiles.
Wu et al. observed variation in standardized carbohydrate meal responses
with metabolic physiology. HbA1c and BMI characterize different aspects
of the host and neither estimates insulin resistance directly.
[Wu et al.](https://pubmed.ncbi.nlm.nih.gov/40467897/)

Use the archived discovery meal population and the existing main terms
and household partitions. Select each person's latest valid HbA1c
collected 1–365 days before their first archived meal, matching participant
and cohort. Community blood draws are dated independently of clinical
visits. Select same-visit BMI from 1–90 days before the first meal.
Exclude a clinical variable if the latest valid date carries conflicting
values. Use complete panels and the same minimum chronological date
support as the earlier transfer run. The main panel includes HbA1c and BMI.
A BMI-only analysis uses its broader eligible population and provides a
separate benchmark. Report selection and lags with each panel.
[HPP blood-test dictionary](https://knowledgebase.pheno.ai/datasets/016-blood_tests.html)

Benchmark adding the host-by-carbohydrate block to the original model.
Compare the joint oral/gut PCs as diet modifiers with and without host
terms. Then test a fixed three-way interaction between carbohydrate,
the joint PCs, and the first clinical variable (HbA1c in the main panel,
BMI in the BMI-only panel), keeping both lower-order diet interactions.
The remaining static terms are absorbed by within-person centering.
Use the original 0.03 penalty for the microbial blocks. HbA1c and BMI
contrasts use 0.5 percentage points and 5 kg/m², respectively.

The primary outcome is the two-hour net response. Peak rise and pre-meal
glucose give additional comparisons. Report pooled household holdouts
and both date-transfer directions. All predictions use the frozen household
partitions and microbial preprocessing within training households. Use
matched household bootstrap intervals and nine pooled direct tests with
Holm adjustment separately per panel. These are exploratory associations
within the discovery cohort.

Run `python -B -m unittest -q test_host` using the verified V12 interpreter.
Run `evaluate.py --package HANDOFF --commit COMMIT --panel hba1c_bmi
--run ID --smoke` and repeat for panel `bmi` using a distinct ID. After
verifying each smoke receipt, use fresh IDs for the full ten folds.
Keep all real outputs, checkpoints and logs in TRE.
