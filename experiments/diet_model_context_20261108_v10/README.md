# V10: nutrient context and matched microbial modification

This is a new discovery experiment package, not a completed scientific result.
It contains **68 new pipelines and 181 candidate settings**: 52 atomic
pipelines, six matched fiber corrections and ten current-inner-OOF ensembles.
Duplicate host-only and single-site parameter settings are removed.

The primary comparator is completed V8 `ridge_wide`. Completed V8 `ridge_joint`
and V9 `convex_wide` are evaluation references only. Old outer predictions never
enter V10 fitting, tuning, feature selection or ensemble training.

## Upload and run

Upload this entire folder, with its name unchanged, into:

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/
```

Then run in the server terminal:

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_context_20261108_v10
bash server_run.sh
```

`setup.sh` verifies upload checksums and the existing native V8 Python environment.
It installs nothing and does not create a virtual environment on FUSE. The default
Python is `/home/ec2-user/.venvs/hpp_v8_20261108/bin/python`. An alternative must
pass all seven dependency pins; set `HPP_V10_PYTHON` explicitly if necessary.

The launcher runs synthetic tests and a closed-file storage probe, then replaces
itself with the foreground worker. `CHECK_CANDIDATES n/181` is synthetic preflight
progress. Real training begins at `LOAD`, `CONTEXT`, `HASH`, then `START r0_f0`.
Keep this terminal and its session running. No persistent output log is opened.
Each completed unit is checksummed and closed before the next unit starts.

The native run may take several hours. Local synthetic elapsed time is not an
estimate of server training time. Real progress includes model names, candidate
counts and outer-fold elapsed minutes. First startup also reads and hashes inputs.

## Results for screenshots

From this folder, after `BATCH_COMPLETE`:

```bash
bash server_report.sh
```

The default report is 29 lines (plus the run-status line). Send this screenshot first. If requested:

```bash
bash server_report.sh --diagnostics
```

This prints matched input additions, microbial input comparisons and explicit
null-versus-correction scores, 35 lines (plus the run-status line). `--all` prints every
pipeline and is not needed for the first screenshot. During a run, use another
terminal for `bash server_report.sh --progress`; receipt counts are not a liveness
check. Partial runs do not publish pooled R2.

## Target and inputs

- Same `fiber_density -> D_Glu`, native mg/dL, discovery population and original
  household partitions: two outer repeats of five folds, three inner folds.
- Person-level nutrition uses current energy-eligible days, complete recorded-event
  nutrient sums, at least three valid days per feature, equal-day mean densities.
  Missing new features do not remove people from evaluation.
- New data reads are projected and filtered by participant, cohort and visit before
  conversion to pandas. Current energy/fiber anchors, person fiber exposure and
  fiber day counts must match; a conflict stops the batch before fitting.
- Numerical imputation, missing indicators, categorical levels, scaling, screening,
  PCA, spline ranges, kernel parameters and selection use the current training role.
  Alcohol frequency is categorical; unknown and missing are distinct from no use.
- No alcohol grams, sparse vigorous-activity field, BMI, other metabolic outcomes,
  sampling-date lags or confirmation outcomes are added.

See [METHODS.md](docs/METHODS.md) for the exact blocks and comparisons. The local
Chinese [summary.md](summary.md) gives the rationale and evidence limits.

## Runtime files and recovery

The worker saves `runs/context_v10_20261108/registration.json`, private units under
`TRE_ONLY/`, and final aggregate `summary.json` / `SUMMARY.txt`. Individual records,
fitted models and predictions remain on the server. Only aggregate terminal text
is needed for screenshots.

After a clean interruption, with no live worker or retained run lock:

```bash
bash server_run.sh --resume
```

Resume requires identical code, data, references, Python, dependency versions and
thread count. A crash may leave `RUNNING.lock`; inspect its recorded PID first.
Do not automatically delete locks, overwrite completed runs, change pins or
bypass preflight. Preserve an incompatible run and choose a new run ID instead.

## Local verification

[research_records/local_validation.json](research_records/local_validation.json)
records the final test status, source hashes, platform and scope. These are
synthetic local tests using the actual frozen preprocessor, not a Linux/data test
and not evidence of improved R2. The native launcher repeats preflight.

The required imported historical source hashes cover V6 `data.py`, `numerics.py`,
`storage.py` and V8 `features.py`. Unrelated historical display helpers are not
required. Existing V6-V9 sources and results are not modified by this package.
