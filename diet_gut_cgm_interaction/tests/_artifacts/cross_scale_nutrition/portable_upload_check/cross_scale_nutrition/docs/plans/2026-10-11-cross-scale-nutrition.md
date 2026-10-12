# Cross scale nutrition implementation plan

**Goal:** Build a standalone upload folder for common-definition T068/T096 localization and same-person-day nutrition comparisons, reusing the completed revised daily panel.

**Architecture:** Read and verify the exact alcohol_revision output, without importing mutable sibling code or reading raw CGM. Use equal-person within FE and one-row-per-person between regressions on the same days. Join participant estimating-equation contributions for paired coefficient contrasts. Save aggregate results and three screenshot screens; personal rows remain in private/ on the server.

**Tech stack:** Python 3.8-compatible syntax, numpy, pandas, scipy; statsmodels/pytest only for local independent numerical checks.

## Frozen scope before new results

- Version 2026-10-11.cross.1; exploratory continuation after prior results, not preregistered or independent validation.
- NOVA4 total-energy percentage / 10 is primary; classified-energy percentage / 10 is a paired definition sensitivity on exactly the same selected days.
- Two original CLR-Z taxa; daily SD primary, Mean explanatory. Day means use arithmetic equal-day aggregation. Between-person exposure is the average of eligible daily percentages, not a new estimate of long-term habitual intake.
- Full sample F: original eligibility/QC, observed-source alcohol policy, both taxa and both outcomes, baseline covariates, at least 5 days and total-NOVA within-person range >=10 pp.
- N is a strict subset of F with complete carb/protein/fiber densities, reapplying the same 5-day/range filters. N0 and N1 reuse byte-identical person-day membership; all within-person means recomputed on their actual days.
- Models: F_joint, F_T068, F_T096, F_classified_joint, N0_joint, N1_joint (each SD/Mean and between/within), plus F_mean_adjusted_joint SD between/within. Total 26 fits / 44 taxon interaction rows.
- Between regression: participant-average daily outcome on average NOVA, original centered microbes, NOVA x microbes, average daily controls and microbe interactions, original baseline covariates. One equal-weight observation per person; HC1 with t(G-1).
- Within regression: person-demeaned daily outcome and regressors, weight 1/eligible_days per person, participant cluster CR1 using N-G-K residual degrees of freedom; t(G-1). Baseline covariate main effects are absorbed. No claim to remove time-varying confounding.
- Day controls: energy, audited alcohol, weekday, observation day and microbial interactions. Nutrition adds carbohydrate/protein/fiber densities and their microbial interactions; no redundant full macronutrient composition.
- Conditional SD adds daily Mean and Mean squared plus microbial interactions, separately decomposed for each scale. This changes the estimand and is not a direct causal effect.
- Reuse original microbe units; centers and IQR anchors frozen from F participants for all comparisons. Joint conditional slopes hold the other microbe at its F mean; support checks acknowledge joint distribution.
- Direct comparisons: within-minus-between (4), joint-minus-single (8), classified-minus-total (8), N0-minus-F (8, sample/composition change), N1-minus-N0 (8, adjustment change), conditional-minus-base SD (4): 40 rows. No ratio attenuation or mediated proportion.
- Overlap covariance: participant score contributions aligned by identity, each block retaining its declared finite-sample correction. Sum coefficient influence differences before their outer product. Paired contrasts use t(min(G_left,G_right)-1); clearly document estimating-equation approximation, not independent SEs.
- BH: primary F_joint qSD4/qAll8; each supporting family qFamily (single fits pooled across both taxa); qAll44 across all interaction rows. Contrasts have separate frozen families plus qAll40. Failed rows remain in denominators with missing displayed q.
- Preserve old output and all null results. Do not promote new signals to original primary findings. No sample-date direction filter or timing warning: same research window per user's instruction.

## Implementation tasks

1. `crossscale/common.py`, `inputs.py`: strict CSV/identity/boolean validation, pinned reference manifest/code/config and saved output hashes, historical primary/nutrient model reconstruction before new fits. Reject synthetic/real mode mismatch.
2. `crossscale/estimators.py`: scaled-SVD OLS/FE, corrected participant scores, fitted coefficient covariance, paired contrasts, diagnostics and BH. Test against statsmodels explicit dummies and OLS robust covariance; test stacked covariance algebra.
3. `crossscale/analysis.py`: common sample selection, anchors, between/within designs, 26-model and 40-contrast registries, fixed multiple-testing families, slopes, support and sample audits. Test no-change paired designs, missing nutrition/range filtering, within means, two-taxon models and failed families.
4. `run_analysis.py`, `run_analysis.sh`, `show_results.py`, `crossscale/reporting.py`, `config/protocol.json`: protected new output, portable reference path, three screens, hashes and version records, read-only result display. Verify no import from siblings or original-input modification.
5. README/METHODS/VALIDATION and tests: full server command, independent numerical tests, synthetic end-to-end runs in existing parent test artifact area, tamper rejection, Python 3.8 syntax and shell checks, final clean upload folder.

## Delivery checks

Only cross_scale_nutrition is uploaded; reference is ../daily_within_person/alcohol_revision/outputs/revised_20261011T103556_258035Z_9583d5. No original-data rebuild, files to download, ZIP, or automatic installation. Tests do not constitute real cohort results. Review effect units, weighting, selection, covariance of dependent contrasts, multiplicity, and completion/failure reporting before delivery.
