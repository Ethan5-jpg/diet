# V12: microbial modification of the fiber–D_Glu association

**Reviewed after V11 completion and the V6–V11 cumulative summary on 2026-11-09. No native V12 scientific result exists.** Use this updated directory, not an earlier prepared copy. The local review and necessary corrections are recorded in [POST_V11_REVIEW.md](POST_V11_REVIEW.md). No prior experiment source or V13 code is modified by this folder.

This is an exploratory association analysis of the same 3,343 discovery people, person-average fiber density and person-level D_Glu (mg/dL). It does not estimate observed personal dietary responses, read confirmation outcomes, add endpoints or seek an R² threshold.

## What is measured

1. **Primary, repeat 0:** global null of zero linear fiber interaction with oral Shannon entropy, oral dominance (maximum relative abundance), gut Shannon entropy and gut dominance. The four coordinates are fixed, continuous and outcome-blind. Report household leverage-adjusted intervals, a conservative Bonferroni-union omnibus and Holm-adjusted site/marker follow-ups. The original multivariate Wald/multiplier P value remains diagnostic only after mild over-rejection in simulation. Repeat 1 is split sensitivity on the same people. Raw D_Glu on the identical people is a planned sensitivity.
2. **Secondary:** twelve matched prediction comparisons: ecology/composition × linear/spline fiber × oral/gut/both. All retain the same oral and gut main-effect background. Product terms are centered and orthogonalized against that background using training rows only, preventing a duplicated main-effect direction from masquerading as interaction gain. Each interaction search includes an exact additive zero option. Negative increments and zero selections stay in the report.
3. **Display:** fixed fiber-density contrast 10→15 g/1,000 kcal, in mg/dL. Separate total model contrast, the interaction component and change relative to the matched null. Report outcome-free local exposure-range support and cross-repeat contrast stability.

The inferential target is a coefficient in a specified finite-additive-basis working model. It is not a general test of every biological interaction. Cross-fitting and household resampling do not eliminate confounding, nuisance-model misspecification or the effect of historical adaptive searches. See [METHODS.md](METHODS.md) and the detailed Chinese [summary.md](summary.md).

The primary screen now shows four-marker family 95% intervals and effective-household information diagnostics. Too few informative households yields UNAVAILABLE rather than a misleading P value. V11's total fusion R² is contextual prediction evidence, not a matched additive null. The same discovered participants were already used across the historical batches.

Recorded microbial calendar dates mostly follow the CGM envelope, and exact date meanings have not been independently verified. The full nutrient background uses its established valid-day person means; joint macro days are not strict diet/CGM common days. This batch changes neither the endpoint definition nor the 3,343-person population to create a daily/meal-response analysis.

## Upload

Copy only `diet_effect_modification_20261109_v12` as a sibling of V11 under the existing `experiments/` directory. If an earlier V12 copy was uploaded, update that V12 directory before starting a native run. Do not overwrite V11 or re-upload/replace the entire older handoff package. Copying files does not run scripts or change installed packages. The native helper still independently checks V11 completion and the prior-summary acknowledgement.

Upload this **one entire directory** beside the existing V6/V8/V10/V11 directories:

`collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/`

Keep the earlier discovery inputs, V4 preprocessing/reference, V6 helpers, V8 feature helper, V10 context reader and native V11 completed summary in place. No ZIP, participant download or new package installation is needed.

Immediately after upload, run the file-only check:

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python upload_check.py
```

It checks source bytes and existing adapter source files only; it opens no participant inputs, starts no model and modifies no input/result files. `UPLOAD_CHECK_OK` means the prepared files arrived intact.

Earlier results can inform the final design review. They are not V12 training predictors: V11's aggregate is used only for completion/population/scope checks and a provenance hash. The new batch reuses the established discovery inputs, folds and preprocessing helpers, rather than V11 fitted models or predictions. If the review changes the design, update this V12 directory and its manifest before the first native run; retain prior run identities. Reusing these discovery participants does not constitute independent confirmation.

## Read-only background inspection after V11

V11 completion has been evidenced locally. After uploading this updated directory, the separate helper can collect the user-authorized background information and verify the native population/inputs:

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12
bash server_background.sh
```

Send its single aggregate screenshot. It reports fiber quantiles, marker IQRs/correlations, and fiber ranges within low/high marker tails. It fits no outcome model, summarizes no Y values, prints no participant rows, changes no input/result files and does not take over the original chat's V11 experiment summary. The established discovery loader opens Y/reference files only to recover unchanged eligibility/folds. This helper requires V11 completion but does not require the later prior-summary acknowledgement, because it is a read-only background check rather than the V12 analysis.

## Run after the review sequence is complete

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12
bash server_run.sh --prior-summary-reviewed --threads 2
```

This explicit two-thread example is recommended when a separately budgeted V13 job is also running and the server resource snapshot permits it. A standalone V12 run can omit `--threads` to use the four-thread default. If V13 has also been uploaded, its optional read-only resource helper can be called from this V12 directory:

```bash
bash ../diet_model_complementary_20261109_v13/server_resources.sh
```

That helper reads CPU quota/affinity and available memory without loading participants or starting either analysis. V12 does not depend on it. Judge concurrency using effective CPU quota and memory, not just the advertised host core count; with insufficient resources, run sequentially or use one thread per job. Do not start the same V12 run ID in two terminals. Existing runs/locks are rejected by the launcher before synthetic preflight, with the worker retaining its independent registration/lock checks.

The flag records the user's acknowledgement that the V6–V11 summary has been reviewed. The program separately requires native V11 `summary.json` to be COMPLETE with both 3,343-person repeats and no active run lock. Without both prerequisites it stops before opening discovery inputs or fitting native models. Do not set the flag while V11 is still running or before reviewing the summary.

The helper uses `/home/ec2-user/.venvs/hpp_v8_20261108/bin/python`, checks upload and adapter hashes, checks the five already-installed dependency versions, runs synthetic preflight, then starts foreground execution with the selected thread count. Keep the terminal open. Do not redirect output to a mounted-drive log. For an already verified alternative interpreter, set `HPP_V12_PYTHON`; no environment is installed or altered.

## At most three result screenshots

```bash
bash server_report.sh
bash server_report.sh --screen 2
bash server_report.sh --screen 3
```

Screen 1: primary omnibus, follow-ups, IQR-scaled interaction differences, family intervals and effective-household diagnostics. Screen 2: all twelve matched increments, selected-option/nonzero-contrast counts and raw-product/full-correction SDs. Screen 3: local support, raw-outcome sensitivity, conditional ΔR² intervals and the two kinds of split stability. During execution:

```bash
bash server_report.sh --progress
```

`--progress` counts unique checksum-verified completed outer units, ignoring duplicate/unfinished receipts. `10/10` means the fold payloads are stored; `final_aggregate=missing` means final aggregation/audit has not yet been published. Use the ordinary result screens for the final checksum-verified report. Incomplete-result enumeration does not create missing checkpoint directories.

Do not download server files. The user can provide these aggregate screenshots; archive their displayed values locally inside this module. Participant-level arrays, fitted models and detailed taxon loadings stay under `runs/<run-id>/TRE_ONLY/`.

## Resume

```bash
bash server_run.sh --prior-summary-reviewed --threads 2 --resume
```

The resume example matches a two-thread initial run. If the original run used four or one threads, use that exact value instead. Resume requires identical code, configuration, data, dependencies and thread count. Complete checksummed units are not refitted. A failure retains `RUNNING.lock`; inspect its PID and the original terminal before manual recovery. Do not remove a live lock or modify sources during execution. If the scientific configuration changes after review, use a new run ID and update the manifest rather than mixing runs.

## Local/synthetic verification

```bash
python check.py
python calibrate.py --repetitions 200
```

These commands use generated data only. Calibration checks a finite-basis score core under stated simulations and includes a deliberately omitted-background stress case; it does not prove validity for every native nuisance approximation. Native feature dimensions and full candidate grids are additionally checked locally with generated microbiomes. Verification records belong in `research_records/verification/`, clearly separated from native result records.

The latest follow-up audit passed 41 checks and the full ten-fold synthetic replay. It corrected progress receipt counting and read-only incomplete-result enumeration, and added an early duplicate-run startup check, without changing the scientific configuration; see [the audit record](research_records/verification/deep_code_review_20261109.md).

`opt0/1` counts folds selecting a positive interaction option. `nz0/1` counts folds whose raw-product contrast varies numerically. `rawSD0/1` and `corrSD0/1` distinguish raw-product from full added contrast dispersion in mg/dL. None of these is a significance test or a count of people with proven responses.

V12 and a separately prepared V13 can use different terminals and their own run directories; V12 does not wait for V13 results. Keep the shared historical sources/inputs and installed environment unchanged during either run. Simultaneous jobs share server resources, so total elapsed time may increase.

## Interpretation rules

- Primary omnibus before individual marker/site follow-ups; the two follow-up families have separate Holm corrections. Repeat 1 and raw-outcome results cannot replace an unfavorable primary result.
- SHAP, feature importance, a nonzero fitted contrast SD, or one selected model cannot establish modification.
- Low support limits the fixed-contrast interpretation; no participants are removed to improve results.
- Household bootstrap intervals for ΔR² keep fitted OOF values fixed. They do not refit the full search and do not correct historical selection.
- PC loadings are fold-specific and compositional. Compare predicted contrasts across repeats, not unaligned PC coefficients. No taxon is labelled a mechanistic cause from this analysis.
- Any paper conclusion must preserve null, negative and unstable findings. Independent confirmation remains a separate, explicitly authorized future step.
