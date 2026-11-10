# V13 replay tolerance repair plan

**Goal:** Resume the stopped V13 run without changing any fitting recipe, participant, prediction, candidate selection or existing checkpoint.

**Evidence:** Native r0_f3 scalar diagnostics show identical members, candidate choices, penalty (0.01), target moments and input scaling. Coefficients differ by 1.56578591759e-9; replay predictions by 2.52560283798e-8 mg/dL. Saved V11-model inference exactly matches its saved predictions. V13/V11 thread counts are 2/4; the role of threads has not been causally established.

**Decision:** Change only the post-fit replay guard from absolute 1e-8 to 1e-6 mg/dL and include the actual error in failures. This is a comparison tolerance, not Ridge solver tolerance, a fit parameter or a score/goal threshold. Keep stored V11 predictions as comparator. The replay reference is not a training member in any new fusion.

**Alternatives:** Forcing four threads would change resume identity without guaranteeing numerical reproduction. Reusing original V11 training-model objects would require wider cache/provenance changes. Neither is necessary for this observed 2.53e-8 discrepancy.

1. Preserve original engine and upload manifest, screenshot and scalar evidence.
2. Add a regression test accepting the observed-order discrepancy and rejecting a substantive mismatch, while reusing cached base fits and fusion fits.
3. Make the narrow guard change and run all synthetic checks.
4. Deliver an explicit offline repair helper. Under the existing run lock it verifies the upload, registered source/data hashes, and records the user-reported scalar witness. It backs up engine, manifest and original registration, then changes only the engine checksum in registration and manifest. Keep a separate repair record with before/after hashes and actual witness. No checkpoint is written or removed. Roll back the three metadata/source files on a handled publication failure. Never remove a preexisting run lock.
5. Test repair idempotence, source drift and large-error refusal, rollback, original metadata preservation and real worker resume without refitting.
6. The user runs the helper on the native server, then uses the existing same-thread --resume entry. Native success remains pending their next screenshot; local testing is not a native execution claim.

All repair artifacts and backups remain under the experiment's research_records directory. No other experiment source or local memory file is modified.

## Post-interruption update (2026-11-09 22:53 screenshot)

The interrupted native helper was still in its initial bulk audit, waiting inside pathlib.open; publication had not started. The root filesystem cause was not identified. To avoid redundant full data/cache scans, repair now checks registered Python/config sources with per-file progress, then preserves all non-engine hashes verbatim. The unchanged normal resume entry recomputes the complete identity and rejects any mismatch before reading a cached outer result or fitting a model. A real-worker regression verifies this exact refusal after a data file changes. Repair never reads data/private checkpoint payloads. Engine-path lookup avoids resolving every registered cache pathname. Neither runtime.py nor run.py was changed.

The native installer updates only the unregistered helper under research_records. Installed and locally formatted helpers have equal ASTs; both source hashes are recorded. The original helper is backed up. Native repair/resume success still requires the user's next screenshot.
