# Local verification record — 2026-11-18

No native HPP analysis or new empirical effect estimate has been run locally.
Generated fixtures test software behavior and are not HPP results.

## Environment and checks

- Python 3.11.17; NumPy 1.26.4, SciPy 1.11.4, pandas 2.1.0, pyarrow 15.0.2,
  scikit-learn 1.4.1.post1, threadpoolctl 3.3.0. The last dependency was installed
  into a separate local target directory for testing; the old environment was
  not modified. The server uses its existing V12 interpreter and exact pins.
- Expanded default suite: 48 tests, 46 passed and 2 optional slow tests skipped
  by default. The two optional tests were also explicitly executed locally and
  passed, as described below. Expanded default suite time approximately 6–16 seconds,
  depending on other generated checks running concurrently.
- Independent ridge normal-equation and paired bootstrap oracles agree;
  negative/zero R2, near-zero denominators, constant outcomes, household IDs,
  exact-null scores and inner zero-choice tie behavior are covered.
- All interaction families use the same fitted coefficients for held-out
  correction and dose contrast. Both-site columns are preserved. Validation
  predictors do not fit the representation. Zero dose and zero residual
  behavior is checked. Elastic Net warnings are exposed and fail the run.
- RFF baseline features and correction mappings match element for element.
  Baseline double differences are zero when diet and microbiome are swapped
  while shared age/sex remain fixed.
- Changing only outer validation outcomes by +1000 changes neither predictions,
  penalties, baseline choices nor stack/selector choices, across all 204
  settings and 78 output routes.
- Five aggregate-only audit tests verify completed-log checks, duplicate logs,
  summary hashes, legacy MSE reconstruction and preservation of negative gains.
  Shipped-history audit verifies 182 pairs, missing groups and task scope;
  its restricted fiber/person-D_Glu exploratory historical maximum is the
  archived legacy gut PC10 ratio 6.4568921959%, not a new V6 result.
- Checkpoint round trip, tamper rejection, interruption after two folds and
  recovery of eight remaining folds, completed replay without refitting,
  changed registration refusal and existing-lock refusal pass. Expanded checks
  also reject incomplete COMPLETE receipts, version/cohort/route/schema changes
  and changed runtime versions or requested threads on resume.
- Shell syntax is valid. A forced preflight failure test confirms the wrapper
  stops before audit or training and returns the failing status through tee.

## Longer generated integration checks executed

1. 180 generated participants, 90 households; complete two-by-five-fold nested
   evaluation with all 204 settings/78 routes and small tree settings; paired
   summaries, COMPLETE hashes and completed resume all passed for this expanded
   delivery (82 seconds while another generated interface check ran concurrently).
   The nested selector recovered a deliberately planted interaction in both
   repeats. These simulated effect sizes are not empirical HPP estimates.
2. Generated full archive: real source adapter, source registration and hashes,
   4,427 index rows, 3,343 eligible participants and 3,059 households. One outer
   fold passed with all settings. Then the formal 100/160-tree baseline settings
   also passed on this generated archive: 676 evaluated people in the checked
   fold, 989 shared-baseline columns in the expanded release, all 204 settings
   and 78 predictions finite and source hashes unchanged (57 seconds with two
   generated test processes running concurrently). This verifies the native interface using generated files, not
   the real HPP data or all ten native folds.

## Review and practical limits

Independent review checked additive baseline constraints, main-effect hierarchy,
inner/outer household separation, both-site PCs, PLS intercept handling, stack
selection, bootstrap arithmetic and storage. A stale-lock auto-removal race was
removed; the runtime now stops on an existing lock. Ranking now distinguishes
point-positive exploratory maxima and denominator-stable maxima, and historical
heterogeneous endpoints are labeled separately from the fiber/person-D_Glu set.

The requested literature-based expansion adds 30 single-interaction families.
Independent oracles additionally check tensor normal equations, a closed-form
balanced two-leaf tree-boost example, zero/extreme-small dose, nonlinear modifier
recovery and exact compositional balance replay. Selected-taxon prevalence,
zero/floor fractions and the rank bound of the balance bank are recorded.
PCs are present in the nonlinear baseline for tree modifiers; exact learned
tree leaves are not extra main effects, and main-effect approximation remains
an explicit limitation. The final summaries also compare expanded selection
against original-family selection on the identical expanded baseline.

Final independent review found that a centered tree direction could require a
negative line-search coefficient, while the initial early-stop condition rejected
all negative numerators. This was corrected to test the absolute numerator.
An independent one-step oracle verifies that the signed step reduces SSE in a
constructed heteroskedastic example and retains the zero-mean modifier.
After that correction, the entire default suite passed again (47 tests, two
optional skips), and both optional full-lifecycle/native-layout tests were
explicitly rerun and passed together in 94.5 seconds.

A further pre-upload review reproduced a fresh-process CLI thread-limit bug:
the limiter was entered before numerical libraries loaded, so requested two
threads could leave newly loaded BLAS/OpenMP pools at their default count.
The numerical stack now loads before the limiter is entered. An independent
subprocess test starts with no numerical libraries or pools, requests two
threads against defaults of eleven, and verifies every loaded numerical pool
is limited to at most two. The expanded default suite now contains 48 tests.
The two optional generated full-lifecycle/native-layout checks also passed
again during this review (80.7 seconds); the numerical model code is unchanged.

The delivered upload is hash-checked and contains code, tests, methods and saved
aggregate history only. Actual availability of archived server sources and any
V6 improvement still require the server run. Ratios and intervals cannot prove
a causal fraction of the diet effect or independently confirm an exploratory
maximum selected from many models.
