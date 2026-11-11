# Carbohydrate from fruit and bread and microbial modification

Wu et al. observed differing personal glycemic responses to standardized
carbohydrate foods in a repeated-meal study. This discovery analysis asks
whether the within-person association of exchanging 10 g of carbohydrate
from the app's bread category for 10 g from its fruit category differs by
microbial profile at fixed recorded carbohydrate, energy, and protein.
The categories are app food-source labels. The recorded meals do not provide
assayed sugar, starch, glycemic index, or verified ingestion timestamps.
[Wu et al.](https://pubmed.ncbi.nlm.nih.gov/40467897/)

Use the archived meal windows and frozen discovery partitions. Link the original
audited food-event source by exact study role, person and archived log window.
Check all reconstructed meal dates and five nutrient totals against the archive.
Apply fixed support rules before reading model estimates. Require at least two
separate dates with at least 10 g of carbohydrate for each target source and
at least four meals per chronological half. As an overlap diagnostic, count
people with a different-date pair dominated by opposite categories within
10 g carbohydrate, 100 kcal energy, 10 g protein, and three clock hours.

Keep the archived carbohydrate main terms. Add source sum and difference terms
and common nonlinear diet and host terms to both models. Hold total and source
sum fixed. The null allows the same microbial profile to modify total grams
and the sum of grams from the two sources. Add the source exchange interaction
only in the comparison model. Use the same fixed penalty as the earlier
meal interaction. The primary readout is the joint oral and gut PC comparison
for the two-hour net response in pooled recording days. Report oral only, gut
only and the fixed literature panel, peak rise and pre-meal glucose, and
both chronological transfer directions. Fit all microbial transformations on
training households in prediction. Test the 12 pooled interaction blocks
with household-clustered Wald tests and Holm adjustment. Use paired household
bootstraps for conditional fold-out intervals in the two original partitions.

Run the synthetic checks with `python -B -m unittest -q test_sources`.
Use the verified V12 interpreter and `analyze_sources.py --package HANDOFF
--commit COMMIT --run RUN_ID --smoke` before launching all ten folds with
a fresh run identifier. Keep the archived results and all derived real-data
artifacts in TRE.
