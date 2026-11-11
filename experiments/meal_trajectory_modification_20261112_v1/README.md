# Time-resolved archived meal glucose response

Rebuild four consecutive 30-minute means of baseline-relative glucose from
the original archived meal windows. Time-resolved studies motivate
examining different portions of the response, including phases whose
changes could cancel when averaged over two hours.
[Functional meal response study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12457981/)

Reuse the frozen meal keys, selection criteria, W14 connection selection,
W09 per-file checksums, and the pinned response implementation. Use its
accepted piecewise-linear glucose traces, baseline and actual sensor
observations. Integrate each bin exactly by adding interpolated bin
boundaries. Repeat at the nominal recorded time and offsets of ±15 minutes.
Require all pre-meal glucose and peak values and the recombined two-hour
means to agree with the archived outcomes. The result has the same meal
keys as the archived analysis. This stage performs outcome reconstruction.

Use the verified V12 interpreter to run `python -B -m unittest -q test_curve`.
Run `prepare_curve.py --package HANDOFF --run ID --commit COMMIT --smoke`
for the first fixed group of people. After the smoke receipt passes, use
a new identifier for the complete run. An interrupted same-version run
can use `--resume` after the process and lock are verified and its saved
batch receipts are intact. All curves, receipts and logs stay in TRE.

## Matched curve models

Express the four 30-minute means in a fixed orthonormal time basis. The
constant mode is twice the original two-hour mean. The linear and
quadratic modes measure the prespecified smooth shape. The cubic mode
supplies an unrestricted sensitivity. Fit the archived meal main terms
separately per mode and retain the same terms in each comparison.

The main comparison adds joint oral/gut PC modification of the linear
and quadratic modes to the existing level-only interaction. Report
paired changes in four-phase curve loss, level loss, and shape loss.
Use the archived fixed interaction penalty and preprocessing within
training households. Preserve the original two household partitions
and compare the original recorded time and offsets of ±15 minutes.
The nominal level-only result must reproduce the archived two-hour
matched analysis. Recenter within the original balanced date halves
for early-to-late and late-to-early transfer. In pooled data, use
household-clustered Wald tests for the level and two smooth shape
modes, with Holm correction over the six tests across clock offsets.

After the complete preparation receipt verifies, run
`python -B -m unittest -q test_curve_models` and
`analyze_curve.py --package HANDOFF --commit COMMIT --run ID --smoke`.
Verify that smoke and run ten folds with a fresh identifier. Fitted
predictions, aggregate results and checkpoint contents remain in TRE.
