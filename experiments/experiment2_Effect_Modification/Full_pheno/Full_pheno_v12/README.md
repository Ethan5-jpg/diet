# Whole diet and dietary categories

Run in AppStream with the pinned interpreter. The V11 input fingerprints and the locked V10 results are verified. Complete outcomes and results stay in TRE.

```bash
rt=$(cat ../../../experiments/diet_effect_modification_20261109_v12/.python_path)
"$rt" -B review_grid.py --run-id review-1 --commit <review-commit>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$rt" -B run_categories.py --run-id categories-1 --commit <group-commit>
```

The retrospective review assigns each of the 68 registered diet measures to exactly one class and each phenotype to a clinical domain. It replays all V11 common pairs and retains the counts of positive and negative diet main, microbiome interaction, and total diet increments separately. The domain-balanced summary averages rates over outcome domains with at least five eligible pairs.

The independent category fits use a common eligible cohort within each outcome and fold. Fit dietary feature checks, missing-value centering, and up to three within-category principal components using training households. Each category has equal training variance weight in the whole-diet representation. The seven diet blocks include nutrient density, fiber and sources, plant foods, animal foods, drinks and other food, processing, and daily patterns and meal context. Logging volume and uncertain NOVA label coverage form a diagnostic block. The primary whole-diet model combines the seven diet blocks. The sensitivity model also includes the diagnostic block.

Standalone recipes compare each category with the common no-diet model. After fitting the whole-diet main model, conditional recipes keep its prediction identical and add a single category interaction with gut PC1 and PC2. Animal-food sensitivity also fits the conditional interaction after excluding red meat from its category basis. Diet terms and microbiome interactions use fixed ridge shrinkage. The V10 models for the original 18 pairs remain locked.

Both analyses use the same historical and two refreshed household partitions as V11. Recurring interaction gains require a positive pooled gain and at least three positive outer folds per partition. Stronger gains meet +0.10 R² percentage points in all three partitions. Whole-diet and category analyses use household-paired tests with Holm correction across all recipes and eligible outcomes in each partition. Repeated partitions reuse the discovery cohort. Confirmation outcomes stay sealed.
