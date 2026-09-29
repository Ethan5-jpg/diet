# Gut–meal bridge implementation plan

**Goal:** Prepare one uploadable `gut_meal_bridge/` package connecting four previously selected gut taxa with existing meal phenotypes, without rerunning mediation or modifying frozen research outputs.

**Architecture:** Read-only adapters validate the completed original meal run, the microbiome source recorded there, diet exports, and the finalized CGM input recorded by nutrition_bridge. A common participant/meal sample supplies three meal outcomes and three existing CGM reference outcomes. One-person-one-row models use HC3 and explicit fixed multiple-testing families. New outputs go to a unique run directory; only three compact pages are returned.

**Tech stack:** Python >=3.8, numpy/pandas/scipy; existing hppmeal and seven_diet_meals helpers, pinned by SHA256. No new server packages or network calls. Workspace is not a Git repository, so no worktree/commit is applicable.

## Approved scope and resolved choices

- Four prior candidates: T023 GGB6012_SGB8640, T046 Clostridium_phoceensis, T068 Eubacterium_rectale, T096 Ruthenibacterium_lactatiformans. Require a unique exact species token; retain full taxonomy and original CLR-Z values. No taxon substitution or new screening.
- Primary outcome: personal mean net iAUC/120 (mg/dL). Secondary outcomes: personal mean premeal baseline and mean peak rise, also mg/dL. Use existing analysis_ready meals, >=5 meals and >=3 dates.
- Reference outcomes: existing Mean, CV and SDw Z values. Require matching participant, cohort, visit and CGM connection. They are same-sample descriptive reference regressions, not reproductions of old mediation coefficients or independent validation.
- Main sample jointly complete for four taxa, three meal outcomes, three reference outcomes and fixed covariates. Strict sensitivity restricts the original exclusion flag to explicitly zero, using identical per-person meals. No imputation or significance-driven sample changes.
- All models: original BASE covariates, corrected hPDI, personal NOVA4, original mean daily energy and alcohol. These two dietary variables are a bounded conditional adjustment motivated by the old hPDI/NOVA4 examples; not a replacement for all seven old dietary analyses.
- Meal models additionally adjust mean meal carbohydrate, mean squared carbohydrate, energy, local-time sine/cosine, preceding-gap, observation-day and meal duration. Net/peak additionally adjust premeal baseline and premeal slope. Baseline outcome must never adjust itself or a response-derived measure. Whole-CGM references use only the person-level covariates above, explicitly a different estimand.
- Four primary tests; eight secondary tests; twelve CGM reference tests, separately within primary/strict settings. Also show q12 across all meal outcomes. NOT_ESTIMATED retains planned denominator. Pointwise 95% CIs, no ranking by significance.
- Optional date metadata defaults to the original configured date file, otherwise the canonical `Transfer/gut_microbiome/gut_microbiome.csv` if present. Missing/ambiguous date linkage is UNKNOWN/AMBIGUOUS and does not establish baseline timing. Duplicate biological sample identities fail. Dates are audited, not used for post-hoc exclusion. No collection order or causal claim.
- No interactions, serial mediation, new recovery-rate metrics, diet re-scoring, or real results generated locally. This follows the user's request for one small next step and at most three result screens.

## Implementation steps and verification

1. Add `gut_meal_bridge/CONTRACT.json`, `sources.json`, `PLAN.md`; freeze taxa, outcomes, model columns, support rules and FDR families before inspecting results.
2. Add `inputs.py`: original-run and dependency hashes, strict species mapping and numeric parsing, corrected diet sources, connection matching, optional date linkage. Test changed inputs, duplicates, mismatches and absent dates.
3. Add `analysis.py`: matched sample construction, person means, HC3 models, fixed FDR denominator, strict subset and readable empty outputs. Compare regression to an independent matrix reference and test baseline not self-adjusted, common membership and original units.
4. Add `reporting.py`, `run_analysis.py`, launcher and manifest: three pages, all detailed tables server-side, explicit failed/check-only states, latest never silently falling back, output integrity validation.
5. Run synthetic end-to-end, clean upload simulation, failed-input/readback tests and Python 3.8 syntax validation. Check source hashes remain unchanged. Record exact tests and limitations in `REVIEW.md`.
6. Deliver a single folder path, one run command and one three-page view command. No server execution or transfers are performed by this task.
