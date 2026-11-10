# V11: food distributions and residual prediction

An exploratory discovery batch for `fiber_density -> D_Glu` (mg/dL). It keeps the original evaluation population (3,343 people), two repeats of five household outer folds, three household inner folds, and frozen outcome/microbial preprocessing. It adds no endpoint, confirmation outcomes, date-lag proxy or participant exclusion.

**79 new workflows = 50 base workflows + 8 predictive residual corrections + 4 explicit fiber corrections + 17 fusions.** There are 206 base/correction candidate settings; fusion penalties are additional internal training choices. Counts are checked by the preflight.

## Upload and run

Upload this whole folder into the existing handoff `experiments/` directory, beside V6, V8 and completed V10. Keep the earlier experiment sources, inputs and native checkpoints in place. No ZIP or data download is needed.

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_distribution_20261109_v11
bash server_run.sh
```

The helper verifies the upload, reuses `/home/ec2-user/.venvs/hpp_v8_20261108/bin/python`, checks the exact seven dependency versions, runs synthetic checks, probes closed-file storage, then starts foreground execution with four threads. It installs nothing and creates no new environment. Keep the server terminal and its process running. Do not redirect the worker to a mounted-drive log file.

Default run ID: `distribution_v11_20261109`. Progress lines identify outer repeat, fold, workflow and selected candidate. The initial `V10_CACHE_CHECK` stage validates existing checkpoints before new fits.

## Reports for screenshots

From this folder, in another terminal or after completion:

```bash
bash server_report.sh
bash server_report.sh --diagnostics
```

The first report is compact; the second shows matched food-profile comparisons, oral/gut comparisons, predictive residual gains and explicit matched nulls. Use separate screenshots. `--all` prints all workflows when requested. During execution:

```bash
bash server_report.sh --progress
```

Reports are read-only. They never refit a model or display participant-level rows. Private checkpoint models/predictions remain under `runs/<run-id>/TRE_ONLY/`. The final aggregate is `runs/<run-id>/summary.json`, with a compact `SUMMARY.txt`.

## Resume

```bash
bash server_run.sh --resume
```

Resume requires identical source/data/reference hashes, configuration, package versions and thread count. A live or retained run lock blocks restart; inspect the recorded PID before recovery. Do not remove locks or change code while a worker is running. Completed checksummed units are reused without fitting. A final summary requires all ten outer folds and a final integrity audit.

## Native prerequisites

- Existing V6 data/numerical/storage helpers, V8 role helpers, V10 context/feature helpers, frozen V4 preprocessing and the original handoff input tables.
- Completed V10 run `context_v10_20261108`, including its `registration.json`, `summary.json` and original `TRE_ONLY` inner candidate receipts/payloads. This batch intentionally does not retrain or silently substitute for missing old caches.
- Original discovery daily food table and shared food-day QC/lifestyle registries, already used by V10.
- Existing UPF lower/upper bound columns. They remain two bounds; no midpoint is invented.

If an old checkpoint/source fingerprint does not match, the batch stops before new experimental fitting. Send that short error instead of bypassing the check. Unrelated historical display scripts are not required.

## Interpretation

`R2_0/R2_1` are pooled outer OOF R2 for each repeated household partition. `dV10` compares total prediction with completed V10 `ridge_diverse`. Residual correction gains use the same selected background and are **prediction increments**, not diet-effect modification. Explicit fiber corrections also have matched zero options and a separate added 10-to-15 g/1,000 kcal contrast SD in mg/dL.

V10's old **INNER** OOF predictors come only from the current outer training households. They train the combined fusion; old **OUTER** predictions are opened only after new choices and meta fitting are fixed, for inference and evaluation. A fixed replay must reproduce V10 ridge fusion within `1e-8` mg/dL. See [METHODS](docs/METHODS.md) for details.

These repeats share participants, and this discovery set has already guided several batches. Conditional household intervals do not cover adaptive search or multiple comparisons. This batch does not promise R2 0.20 or 0.30. Only one identical workflow crossing a threshold in both repeats counts as passing it. Prediction gains do not establish microbial causal modification of dietary effects.

The detailed local Chinese design/result interpretation is in `summary.md`. Local checks use synthetic inputs only; they do not constitute native data validation or a completed scientific experiment.
