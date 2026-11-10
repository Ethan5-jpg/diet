# V13: complementary prediction models

Status: local code and synthetic verification; HPP results pending native execution.
This is a separate prediction batch. V12 owns the effect-modification analysis.

- Fixed discovery task: fiber_density -> D_Glu (mg/dL), the original 3343 evaluated people.
- Original two household outer repeats x five folds, with three household inner folds.
- 38 base workflows / 84 candidate settings plus 17 fusions = 55 new workflows.
- PC/SIR/SAVE conditional representations; paired hinges; FITC sparse GP;
  quantile-bin probability regression; multi-quantile mean regression; graph ridge.
- Completed V10/V11 matched INNER caches supply the old 90-member bank. New combined
  pools have 128 members. Historical OUTER arrays are inference-only, after fitting.
- Default two threads. No GPU, new package installation, downloaded weights or data export.
- No prospective confirmation or microbial effect-modification claim.

## Upload and run

Upload this whole folder as a sibling of V11 under the native handoff experiments directory.
Keep completed V10/V11 runs and their sources/cache units in place. Do not overwrite them.

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_complementary_20261109_v13
bash server_resources.sh
bash server_run.sh
```

The run helper checks the upload hashes and exact existing dependencies, executes synthetic
preflight, then replaces itself with the foreground worker. No pip/venv command runs.
The default interpreter is `/home/ec2-user/.venvs/hpp_v8_20261108/bin/python`.
If needed, set `HPP_V13_PYTHON` to an already verified compatible interpreter.

For a separate V12 terminal, use two threads per process only if CPU/memory permit.
On a smaller quota use `bash server_run.sh --threads 1` or run sequentially.
Changing the thread setting when resuming changes the registered identity and is rejected.
The resource helper is a snapshot, not a reservation or guarantee. Browser/session persistence
is controlled by the remote service: keep the running terminal/session available.

## Screenshots and recovery

```bash
bash server_report.sh --progress
bash server_report.sh
bash server_report.sh --diagnostics
bash server_report.sh --all
```

Use `--progress` in a separate terminal while the worker runs. After BATCH_COMPLETE, send
screenshots of the short report and diagnostics. The full report is optional and longer.
Reports do not fit models or print individual data/weights. Private payloads stay in
`runs/complementary_v13_20261109/TRE_ONLY/`; aggregate JSON stays in the native run directory.

After a clean stop, `bash server_run.sh --resume` reuses identical completed units.
A stale lock, duplicate closed receipt, changed source/input or incomplete historical bank
stops execution. Inspect the process before any manual recovery; do not remove live locks.
Do not start two workers for one run-id or redirect continuous output onto the FUSE project mount.

See [METHODS](docs/METHODS.md), [frozen grid](docs/MODEL_GRID.md),
[Chinese local summary](summary.md), and [local verification](research_records/local_verification.json).
