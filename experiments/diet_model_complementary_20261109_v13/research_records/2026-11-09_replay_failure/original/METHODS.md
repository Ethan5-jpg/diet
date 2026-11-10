# V13 methods and interpretation

## Fixed question and evaluation

Predict the existing processed D_Glu in mg/dL from the original fiber-density/host inputs,
oral/gut profiles and the already validated joint-day nutrition context. This is one
exposure-outcome task, not an average across the original 18 combinations.

The frozen V4 evaluation rows and processed outcomes remain authoritative. Both outer
repeats evaluate the same 3343 people. Household groups never cross an outer or inner
training/validation boundary. Candidate selection minimizes pooled three-fold inner MSE.
R2, RMSE and MAE use all outer predictions in a repeat, not mean fold R2. Signed negative
scores and negative increments are retained. Original outcome processing is inherited.

The nutrition encoder exactly replays V11's joint context: complete carbohydrate/fat/protein
days, sodium, lifestyle and alcohol; training medians, missing indicators and training-only
category coding. No daily outcome, date lag, new cohort or complete-case exclusion is added.

## Conditional microbial representations

Within each training role and site, recover the exact original taxon indices behind frozen
CLR columns. Select up to 128/256 variable columns using training raw prevalence, with stable
ties. No univariate outcome screening precedes this step. Host/context nuisance coefficients
are fitted to CLR in training; their residuals form R. The host/context inputs remain in the
prediction design. All transforms are replayed with stored training state.

- PC control: leading eigenvectors of R'R/n; near-null directions are zeroed.
- SIR-inspired: whiten with W=(Cov(R)+rho I)^(-1/2), where rho is 5% of mean variance
  (at least 1e-8). Residualize training Y on the same nuisance design, use its training
  quantile cuts to form 8/12 slices, and take leading directions of weighted slice-mean
  outer products. Reduce slice ties deterministically.
- SAVE-inspired: use the same whitening/slices and eigenvectors of the weighted squared
  covariance change [0.8*(Gamma-Cov(Z|slice))]^2, with Gamma=Z'Z/n. This is a regularized
  predictive adaptation; it is not an assumption-free sufficient-dimension estimator.
- Graph representation: retain the selected nuisance-residualized CLR coordinates.

Directions and score scaling fit on training data only. Constant residual outcomes give zero
SIR/SAVE directions. The synthetic tests include a linear planted SIR signal and a symmetric
quadratic SAVE signal. Success there does not establish performance on HPP.

SIR is grounded in [Li (1991)](https://doi.org/10.1080/01621459.1991.10475035).
SAVE has additional slice-size/asymptotic conditions; see the original research by
[Li and Zhu (2007)](https://doi.org/10.1214/009053606000001091).
No consistency claim from that literature is applied to these fixed small grids.

## Complementary learners

HGB/CatBoost RMSE controls share their grid and context across PC/SIR/SAVE. HistGradientBoosting
has no validation-driven early stopping. CatBoost uses fixed iterations and explicit CPU threads.

**Paired hinges.** Standardize inputs, consider knots at training 0.2/0.5/0.8 quantiles, and add
paired hinge terms by orthogonalized squared-error reduction. Limit the forward basis to 13/25
terms and degree 1/2. Backward deletion scores ridge refits with a GCV-style complexity of
1+2.5 times the retained-column count and a denominator floor. Compare with an intercept-only
fallback. This is a bounded custom MARS-style model inspired by
[Friedman (1991)](https://doi.org/10.1214/aos/1176347963), not a full MARS implementation.

**Sparse GP.** Standardize each input and weight blocks by inverse square-root block width.
Use a Matern-3/2 covariance and 96/192 training-only MiniBatchKMeans inducing centers.
With Kuu jitter 1e-8, Qff=Kfu Kuu^-1 Kuf and D=noise+diag(Kff-Qff), solve
H=Kuu+Kuf D^-1 Kfu and alpha=H^-1 Kuf D^-1 (Y-mean(Y)). Predict mean(Y)+K*u alpha.
These are FITC posterior-mean equations, independently checked against a dense corrected-prior
solve and the all-training-inducing exact-GP limit. Noise is on the standardized-target scale.
No predictive or causal uncertainty interval is supplied. See
[Quinonero-Candela and Rasmussen (2005)](https://www.jmlr.org/papers/v6/quinonero-candela05a.html).

**Quantile-bin probability regression.** Training-only outcome quantiles define 4/8 bins.
Fit CatBoost/HGB classification to observed classes; return the class-probability-weighted
training outcome means. Persist the cuts/means. Tied/empty bins and one-class outcomes are
handled. This is not a proportional-odds ordinal regression and supplies no modifier test.

**Multi-quantile mean.** CatBoost MultiQuantile learns 5/9 equally spaced midpoint quantiles;
average their predictions as a coarse quadrature approximation to the conditional mean.
Tail discretization and quantile estimation can introduce bias; full mean accuracy is evaluated
with outer MSE. This is distinct from using the median as if it were the mean. The pinned
1.2.8 output shape is exercised in tests; see the
[official objective documentation](https://catboost.ai/docs/en/concepts/loss-functions-regression).

**Graph ridge.** Compute positive within-site standardized CLR residual correlations in training,
retain eight nearest positive neighbors, symmetrize, and form normalized Laplacian L. The
coefficient prior in original standardized coordinates is block_width*(I+gamma L), with
identity host/context blocks, gamma in {0,1,10}, and ridge alpha=n*0.01. Isolated nodes have
zero Laplacian penalty. Gamma=0 is a matched ordinary block-ridge option; all options are
selected internally, not reported as separate unselected outer wins. The statistical graph
is learned from co-occurrence, not a validated pathway or causal network. Network penalties
have a literature precedent in [Li and Li (2008)](https://doi.org/10.1093/bioinformatics/btn081);
this implementation uses a different, data-derived graph.

All base learners standardize training targets and invert to mg/dL for predictions. Exactly
constant training outcomes use a constant native-unit prediction instead of a failing learner.

## Historical bank and fusion

Validate completed V11/V10 registration scope, all required source/data hashes and closed
cache receipts. Missing/corrupt units stop execution; no fallback refits an old experiment.
For the current outer training people, reconstruct V10's 28 selected INNER columns and V11's
62 selected INNER columns from their original role/candidate caches. Recompute selection
from the identical current inner targets. The 90-column historical bank is only INNER OOF
training information from the current outer training families.

Rebuild V11 positive_combined with its original standardization and household meta-CV penalty
recipe. The new 38 columns join this old bank for the 128-member combined pool. Pools of oral,
balanced, non-tree, all-new and combined get convex, ridge and nonnegative fusion; new/combined
also get sparse elastic-net fusion. Meta-CV on inner OOF features is an internal training
heuristic, not an independent validation score or a third outer layer.

Only after candidate choices, new outer model fits and all meta fits are fixed, deserialize
historical OUTER arrays for inference/comparison. Check row/processed-Y alignment and selected
settings. Require reconstructed V11 positive predictions to match the stored reference within
1e-8 mg/dL. Persist the maximum error. Do not combine independently favorable repeat scores.
Contributing fitting warnings propagate through a fusion when their standardized coefficient
or convex weight exceeds 1e-8; full native diagnostic records remain available.

## Reporting and interpretation

Report 55 new workflows and four references. Default rows rank by the smaller repeat R2;
this is exploratory ranking after several development batches, not prospective selection.
Matched tables compare SIR/SAVE vs PC, distribution vs RMSE objectives, oral vs balanced
fusion, and combined vs new pools. Algorithm/grid selection still differs within a family;
these are recipe comparisons, not isolated mechanistic mediation estimates.

For paired gains, delta_i=(Y_i-p_ref_i)^2-(Y_i-p_new_i)^2. Divide mean(delta) by Var(Y) for
DeltaR2. A household sandwich SE of mean(delta), with G/(G-1) correction, gives the conditional
normal interval in JSON. It holds trained OOF predictions fixed and excludes model fitting,
prior/adaptive model searches, selection and population transport uncertainty. No P-value is
used to claim that the screened best pipeline is confirmed. Both repeats share participants.

A stronger predictor, nonzero graph edge, quantile spread or latent-direction loading does
not establish microbial effect modification. The independent V12 program handles that
scientific question. R2 0.20/0.30 counts are retained only as historical descriptive targets.
