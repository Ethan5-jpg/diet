# Broad method exploration: fiber density and D_Glu

User-directed breadth-first discovery screen, extending the 2026-11-03 v1 pilot.
The objective is to explore genuinely different prediction mechanisms before
choosing a small number for deeper interpretation and confirmation.

## Literature and method rationale

- Zeevi et al., Cell (2015), doi:10.1016/j.cell.2015.11.001, and the US validation
  by Mendes-Soares et al. (2019), https://pmc.ncbi.nlm.nih.gov/articles/PMC6599737/:
  gradient-boosted trees combine meal, host, and microbial features for postprandial
  glucose prediction. We borrow the model family, not their meal-level endpoint,
  performance targets, or exact feature set.
- Asnicar et al., Nature Medicine (2021), doi:10.1038/s41591-020-01183-8:
  random forests predict metabolic traits from microbial profiles; their twin-aware
  evaluation reinforces retaining the existing household folds here.
- Huang et al., Supervised learning and model analysis with compositional data,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10343141/: KernelBiome motivates flexible
  kernel models for microbial data. Our RBF random-feature models on existing
  standardized CLR/PCs are simple kernel-inspired candidates, NOT a reproduction
  of KernelBiome or its compositional interpretation procedures.
- Extra Trees, RBF-SVR, PLS, MLP and splines broaden the functional assumptions.
  These are comparison candidates, not claims that those papers used every method.

## Screen: 54 families, 126 settings, 630 five-fold candidate fits

Prediction track: random forest, Extra Trees, histogram gradient boosting,
classical gradient boosting, RBF-SVR, RBF random features plus ridge, PLS,
one-hidden-layer MLP, and additive splines. All nine use the same host design with
PC10 inputs under host-only / oral / gut / both-site ablations (36 families).
RF, Extra Trees, and histogram boosting also use full CLR for the three microbial
input configurations (9 more). Two configurations per family. Predictors use
fold-training-only target standardization; no row-random validation early stopping.
MLP optimization warnings are retained rather than hidden.

Varying-coefficient track: diet times (i) spline functions of PC10, (ii) RBF
features of PC10, or (iii) RBF features of full CLR, for oral / gut / both (9
families, four penalties each). These explicitly let microbial composition modify
the dietary slope. BOTH each interaction model and its matched null contain the
same nonlinear microbial main effects as well as v1's dual-site main background.
The combined-site RBF map can also represent oral–gut joint patterns.

## Evaluation

Use the existing 4427-row discovery interface, 3343 eligible out-of-fold people /
3059 households, fixed five folds, and original preprocessing and endpoint rules.
Training transforms, spline knots, scaling and model fitting exclude validation
rows. Numerically reproduce v1 gut-PC10 predictions in each fold before proceeding.
Each family's settings are selected using mean five-fold MSE. Report all candidates
and selected families, not only winners. This is a broad development screen using
the same folds for selection and reporting, not an independent accuracy estimate.
The confirmation outcomes remain unopened for later direction selection.

For general predictors, compare total R2/RMSE, original-null delta, and site input
ablations. Their gain may come from main effects as well as interactions.
For explicit varying-coefficient models, additionally compare the correctly matched
no-interaction model. Report model-implied 10-to-15 g/1000 kcal contrasts in original
units. The general-predictor contrast includes host interactions; the VC contrast
isolates the centered microbial modifier. Do not compare their SD as the same
quantity. Marginal dietary support is recorded; these are associative model
contrasts, not measured individual intervention effects.

## Execution and layout

The new source directory is `experiments/diet_response_breadth_20261103_v2/`
relative to both the GitHub repository and the HPP collaborator package. All paths
are explicit; this script does not infer the package by directory depth.
Original data, v1 source and results are read-only. Fitted objects and individual
predictions stay in `results/TRE_ONLY`-named files/directories on the server.
Fresh sequential writes are used for the mounted filesystem.

```bash
python verify_breadth.py --pilot-code /absolute/path/to/v1/code
python -u run_breadth.py \
  --package /absolute/path/to/collaborator_handoff_20261102_v1 \
  --pilot-code /absolute/path/to/v1_attempt02/code \
  --output /absolute/path/to/collaborator_handoff_20261102_v1/experiments/diet_response_breadth_20261103_v2/results \
  --code-commit COMMIT_OR_SOURCE_SHA256
```

Environment observed before implementation: HPP Python 3.11.5, sklearn 1.4.1.post1,
four CPUs. XGBoost, LightGBM, CatBoost, torch, interpret and econml absent.
This wave uses installed NumPy/SciPy/sklearn/pandas/pyarrow and at most two forest
workers. Those absent packages and GRF/causal forests are follow-on candidates,
not silently claimed as tested algorithms.
