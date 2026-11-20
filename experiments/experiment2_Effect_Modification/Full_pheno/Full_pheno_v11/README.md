# Expanded quantitative phenotype and diet batch

Run in AppStream from this directory using the pinned V12 environment. Participant records and complete outputs stay in TRE.

```bash
rt=$(cat ../../../experiments/diet_effect_modification_20261109_v12/.python_path)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$rt" -B run_batch.py --run-id batch-1 --commit <commit>
```

The catalogue records raw table sizes from Parquet metadata. All model inputs use the registered discovery participants. The outcome inventory reports exact visit linkage, paired recording QC for CGM and ECG, assay tags for Nightingale, and sleep signal quality. Blood assays use the latest eligible blood draw within 365 days before the first approved food day. Repeated sleep records are summarized by participant with the median over nights with the specified quality score of at least 80. Exploratory measurements include quantitative clinical and omics variables, body composition, vascular markers, and objective physiologic traits. Diagnoses, identifiers, questionnaire fields, genetics and raw signals are outside this model grid. The confirmation outcomes stay sealed.

The diet inventory includes the prior V10 recipes and individual food-category energy shares for events whose calories reconcile to the registered energy day. The original two diets may pair with new outcomes. The original 18 pairs retain the V10 locked-main results and are excluded from this separate grid.

Outcomes with fewer than 200 observed discovery households and exact duplicates are removed from the grid inventory. Each pair must also satisfy the V10 minimum of 120 training and 40 held-out households, adequate diet variation, and nonconstant held outcomes. Fit host and microbial backgrounds per training fold and clip outcomes at training-only 1st and 99th percentiles. Matched recipes compare a no-target-diet baseline, a linear diet arm, and that same diet arm plus a fixed half-strength gut PC5 interaction. Batch fits share their design only across endpoints with the same observed participant mask. The analysis applies Holm adjustment to household-cluster-paired prediction tests across each complete partition grid.

The historical and two new partitions regroup the same discovery households. The stable list requires a positive pooled interaction gain and at least three positive folds in all three partitions. A stronger subset also requires +0.10 R² percentage points in every partition, with a separate positive diet-total requirement. The complete per-pair results and audit receipts are in the run directory.
