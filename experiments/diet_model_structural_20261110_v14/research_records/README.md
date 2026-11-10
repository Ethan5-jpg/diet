# V14 research records

This folder holds local synthetic checks and user-supplied aggregate
screenshots. There are no native HPP individual records, model files or prediction
arrays here. Native completion and all 34 planned paired diagnostics are archived.

- `prior_runtime_hashes.json`: only actual shared historical runtime dependencies;
  optional old display helpers are deliberately excluded from native preflight.
- `initial_test_log.txt`: first local integration run; historical development log.
- `pre_rank_guard_test_log.txt`: 39-check integration pass before the added PCA
  numerical-rank guard; retained separately from final checked sources.
- `local_test_log.txt` and `local_verification.json`: final checked source and exact
  checks, created after verification. They must not be read as real HPP performance.
- `2026-11-10_recheck/`: second review, preserved original source/40-test record,
  reproducing regression failures, repaired regression pass and current review.
  The current full suite has 42 checks. Its current setup smoke log is in that folder.
- Future screenshots and transcriptions stay in dated subfolders of this folder.
- [Native first-fold progress, 02:48:37](2026-11-10_024837_native_start/summary.md):
  retrospectively archived from the conversation image; original temporary image
  was unavailable when archived, so no original-image copy is claimed.
- [Native completion, 12:36:47](2026-11-10_123647_complete/summary.md): original
  screenshot, all 13 visible result rows, workflow inventory and interpretation.
- [Paired diagnostics, 12:45:24](2026-11-10_124524_diagnostics/summary.md): original
  screenshot, all 34 model-reference deltas, group counts and interpretation limits.

Default terminal reports are short; full scores, conditional intervals, loadings
and per-person predictions stay on the server. Preserve rounded screenshot values
and distinguish observed results from plans, simulation and unshown rows.
