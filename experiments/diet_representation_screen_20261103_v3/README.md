# V3: matched-input prediction and representation screen

Follow-up to the completed v2 breadth screen, authorized on 2026-11-03.
The central question is whether gains arise from prediction algorithms, retained
microbial information, or both. This directory is independent of mentor source,
v1 and v2. Dependencies are read-only; output creation refuses existing results.

## Design

- Four algorithms: histogram gradient boosting, random forest, Ridge and Elastic Net.
- Host-only reference plus oral, gut and both-site input under CLR/PCA10/PCA30/PCA50.
- 52 families, 182 parameter configurations, 910 candidate fits across five household folds.
- All models use the same host design, target preprocessing and eligible sample.
- PCA uses a single train-only full-SVD PCA50 map per site; 10 and 30 components are exact prefixes.
  Both-site PCAk concatenates k oral plus k gut components. The PCA is applied to the
  existing fold-specific CLR feature matrix, without whitening. New PCA10 is not
  assumed identical to the archived v2 PC10 design.
- Ridge and Elastic Net standardize input columns using training rows only. All
  outcomes are standardized on training rows and predictions restored to original units.
  Their regularization is distinct from the v1 partially penalized background.
- Existing full-CLR and host-only tree configurations act as replay anchors: 60
  fold/configuration prediction comparisons must agree with v2 to absolute tolerance 2e-8.

## Initial grids

| Algorithm | Settings per input | Settings |
| --- | ---: | --- |
| Histogram boosting | 3 | 7/15/31 leaves; L2 10/30/30; 160/160/240 iterations; learning rate .04/.04/.03 |
| Random forest | 2 | 200 trees; leaf 10/max_features .5 or leaf 20/max_features 1.0 |
| Ridge | 3 | sklearn alpha 1, 100, 10000 |
| Elastic Net | 6 | alpha .001/.01/.1 crossed with l1_ratio .2/.8; 20000 iterations; tol 1e-5 |

This is a bounded, v2-informed development experiment. Settings are selected by
mean five-fold MSE; the same folds provide reported performance. No claim of an
independent performance estimate or significance is made. Confirmation outcomes
are not accessed. Relative changes in R2 are not presented as relative error
reductions; residual MSE reduction is reported separately.

## Validation and outputs

Synthetic checks test held-out feature invariance of training PCA, nested PCA
prefixes, unchanged CLR columns, all parameter settings and an independent ridge
normal-equation solution. After fitting, all 910 fold losses are recalculated from
saved prediction arrays; source hashes and household grouping are checked.

`results/` contains all candidate scores, diagnostics (including warnings), chosen
models, PCA explained-variance summaries, aggregate analysis and an audit receipt.
Individual predictions, outcomes and estimators use `TRE_ONLY` names. Public
output includes only an explicit aggregate allowlist. It is staged to the formal
Egress Store after a passing audit; staging does not imply platform approval.

Run on HPP using the existing `pheno` environment:

```bash
python launch.py --package /absolute/path/to/collaborator_handoff_20261102_v1 \
  --pilot-code /absolute/path/to/v1_attempt02/code \
  --breadth-code /absolute/path/to/diet_response_breadth_20261103_v2 \
  --code-commit GITHUB_COMMIT_SHA \
  --egress-store /absolute/path/to/egress-store-WORKSPACE_ID
```

The launcher uses a local live log and archives it on completion. Each completed
fold writes a prediction checkpoint in TRE. No previous experiment files are changed.
