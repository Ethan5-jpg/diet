# Training-household calibration of category interactions

Run in AppStream using the pinned interpreter. V11 input fingerprints and the V12 category archive anchor the analysis. Raw and complete results stay in TRE.

```bash
rt=$(cat ../../../experiments/diet_effect_modification_20261109_v12/.python_path)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$rt" -B run_adaptive.py --run-id adaptive-1 --commit <commit>
```

The V12 archived partition verifies the no-diet, whole-diet main, and fixed whole-diet interaction scores. The nested evaluation uses two fresh partitions of the discovery households. The original 18 fitted models remain fixed.

Within each outer training set, two held household subfolds separately refit microbes, UPF calibration, and diet patterns. These give side-by-side validation gains for the whole-diet correction and the seven category corrections over the same main prediction. Domain policies require gains on both inner folds and weight each source equally within its domain. The category mix can select a single correction or a capped blend of two. The endpoint guard checks single corrections and the prior domain mix, requiring a positive endpoint gain in both inner folds and ranking with 20% domain evidence. A policy with insufficient support or gain returns a zero correction. Fixed, half-strength, and zero-correction arms provide controls.

All outer model comparisons use exactly matched no-diet and whole-diet main prediction scores. Recurring gains require at least +0.10 R² percentage points, a positive total diet contribution, and at least three positive folds in each new partition. Recurrent losses are at most −0.10 points in each partition. Holm-corrected paired tests compare the interaction arm against the shared diet main. Repeated folds use the same discovery cohort. Confirmation outcomes stay sealed.
