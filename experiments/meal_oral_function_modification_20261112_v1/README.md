# Oral pathway profiles and meal carbohydrate response

Test whether the association between a within-person 10 g carbohydrate
difference at fixed recorded energy and protein and the two-hour post-meal CGM
response differs across oral microbial pathway profiles. Use the archived
reconstructed meal windows, the existing discovery household partitions, and
the previously audited oral HUMAnN links and three fixed feature representations.
DNA-derived pathway profiles describe functional potential. Tily et al. studied
gut metatranscriptomic activity and meal response, including a suggestive
carbohydrate interaction with tyrosine metabolizers. The assays and sites address
different questions. [Tily et al.](https://pubmed.ncbi.nlm.nih.gov/34799839/)

Keep the previous meal model formula, interaction penalty and date-support rule.
The same participants and meal windows enter all variants, including the pooled
and bidirectional date comparisons. Use the prior training-only functional
transformations and the first three PCs. Compare each of relative pathway
abundance, coverage confidence and abundance retaining unreported status with
the same background. Add carbohydrate-by-technical terms based on raw and aligned
read counts and their ratio to both arms. The relative-abundance encoding uses
zero for unreported pathways as a working assumption. Exclude previously
audited whole-null profiles from this new comparison.

The primary comparison is pooled relative-abundance interaction for net
two-hour glucose response. Peak and pre-meal glucose provide supporting
comparisons. Compare relative-abundance PCs plus taxonomy against the same
taxonomy-only interaction block to assess incremental prediction. Repeat each
comparison with bidirectional time transfer and household separation. Report
all conditional paired household bootstrap intervals and point estimates from
both partition repeats. This is discovery sensitivity analysis of associations.

Use the verified V12 Python environment and commit-pinned sibling sources.
Run the synthetic test with `python -B -m unittest -q test_meal_function`.
Run `meal_function.py --package HANDOFF --commit COMMIT --run RUN_ID --smoke`
for one fold. After verifying the smoke receipt, use a fresh identifier and omit
`--smoke` for the full ten folds. Record and verify source hashes, fold
receipts, registration and summary. Keep all real artifacts and logs in TRE.
