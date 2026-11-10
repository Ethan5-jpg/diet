# V14: structural prediction models

Status: NATIVE COMPLETE, observed in the user's 2026-11-10 completion screenshot.
All 10 household outer folds and 66 new workflows completed; fit flags: 0.
0/66 beat V11 in both repeats; 1/66 beat V13 in both repeats.
The leading new workflow by lower-repeat R2 is positive_combined: 0.131369 / 0.139524,
with delta_R2 versus V11 +0.000814 / -0.000331. All 34 paired diagnostics are archived.
See the [Chinese completion record](research_records/2026-11-10_123647_complete/summary.md).
See the [paired diagnostics](research_records/2026-11-10_124524_diagnostics/summary.md):
Fourier SIR lost all 16 matched comparisons in both repeats; CCA gained in both
matched heads, with substantially different magnitudes across repeats. Balanced
prediction pools gained modestly but have more members than the oral pools.
The earlier 42 local synthetic checks passed separately (84.239 seconds), including
all 80 default candidate fits and the actual two-repeat/ten-fold worker.
This is a prediction batch, separate from V12 effect-modification inference.

The pre-upload recheck fixed one boundary defect: metric regressors now apply the
same saved training-variance mask to training and query inputs. Unsupported constant
or near-constant columns cannot introduce query-only distance or slope shifts.
See the [Chinese recheck record](research_records/2026-11-10_recheck/summary.md).

- Fixed fiber_density -> D_Glu (mg/dL), the original 3343 evaluated discovery people.
- The same two household repeats x five outer folds; three inner household folds.
- 40 base workflows / 80 candidate settings and 26 meta workflows = 66 workflows.
- Linear SIR controls, Fourier-feature SIR, pilot-gradient directions and paired
  oral/gut CCA with an equal-width paired-PC control.
- Local ridge, split-sample forest-kernel local regression, normalized ReLU NTK,
  inductive graph residual smoothing, SIR-space SVR and sparse Bayesian RBF bases.
- Historical bank: 90 V10/V11 members + 38 selected V13 members = 128.
  Combined bank: 128 historical + 40 new = 168 members.
- Saved V11/V13 meta-models are replayed without re-solving their historical NNLS.
- Existing pinned V8 environment; no installation, pretrained download or data export.

## Upload and run

The current native run is complete. Commands below are retained for provenance;
use the Results section to inspect this run rather than starting it again.

Upload this whole folder next to V13 under the existing handoff `experiments/`.
Keep completed V10/V11/V13 sources and private run directories in place.
Do not overwrite old experiments. From the native terminal:

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_structural_20261110_v14
bash server_resources.sh
bash server_run.sh
```

The default interpreter is `/home/ec2-user/.venvs/hpp_v8_20261108/bin/python`.
The helper verifies dependencies and upload hashes, runs synthetic checks, then
executes the foreground worker with two threads. Native Linux preflight is required.
Set `HPP_V14_PYTHON` only to another verified compatible existing interpreter.

For a smaller CPU quota use `bash server_run.sh --threads 1`. Concurrent V12 and
V14 runs share CPU/RAM; additional terminals do not reserve more resources.
Keep the terminal/session available; browser persistence is controlled by the service.
Do not redirect continuous logs onto the project FUSE mount.

## Results

After BATCH_COMPLETE, normally send just the default report and diagnostics:

```bash
bash server_report.sh
bash server_report.sh --diagnostics
```

`--progress` reports verified completed folds without fitting; `--all` prints all
scores when a specific question requires them. Full detail, models, loadings and
individual predictions remain in native `runs/structural_v14_20261110/TRE_ONLY/`.
Aggregate JSON remains in the native run directory.

## Recovery and boundaries

Use `bash server_run.sh --resume` with identical source/data/environment/thread
settings after a clean stop. Completed units are reused without refitting. A live
or stale lock is never removed automatically. Changed inputs, provenance or closed
receipts stop the run. Initial/final hashing prints progress; the old V13 replay
repair is not applied to V14 and no prior registration is amended.

This batch changes predictive representations and model classes. It does not
identify individual causal diet responses or test microbial effect modification.
The local implementations are finite-budget variants; paper inference guarantees
do not automatically apply. Meta-CV is internal selection on INNER predictions,
not an independently nested external score. Only household OUTER results assess
prediction, and repeated exploration still needs separate confirmation.

See [methods and sources](docs/METHODS.md), [exact model grid](docs/MODEL_GRID.md),
[Chinese local summary](summary.md) and [verification records](research_records/README.md).
