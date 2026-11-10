# V14 methods, implementation scope and primary sources

The frozen task is prediction of D_Glu in mg/dL from fiber density, host covariates,
validated joint-day nutrition/lifestyle context and oral/gut microbiomes. All
learned transforms use the current training role; validation features are only
transformed. No validation response, confirmation outcome or date proxy enters fitting.

## 1. Conditional microbial inputs

Frozen training CLR output columns are mapped back through both reference_indices
and output_indices before raw training prevalence is computed. At most 128/256
variable columns per site are retained by prevalence. This mapping is checked.
Host/context linear nuisance fits are removed before microbial projections.
The final prediction model still receives host and context main inputs.
New PCA stages preserve their declared output widths but set eigen-directions
below the fixed numerical-rank threshold to zero. This applies to input PCs,
private-view PCs and nonlinear meta PCs, preventing later standardization from
amplifying unsupported floating-point directions. It does not change frozen V13 SIR.

For non-tree distance/kernel regressors, each input block is standardized using
training means/variances and divided by the square root of its number of active
columns. This avoids letting a larger block dominate Euclidean distance solely
because of its width. Constant columns contribute zero.
The active mask is fitted from training variance (>1e-12), saved and applied to
both training and query transforms. Query-only values in inactive columns are
zeroed rather than entering distance, kernel or local-slope calculations.

## 2. Three conditional single-site projections

**SIR control.** Reuse the exact regularized SIR implementation from V13. Its
positive V13 contrasts are development evidence, not independent confirmation.

**Fourier-feature SIR.** Residual CLR is reduced to 24/32 conditional PCs, scaled,
then mapped to 128/192 random cosine features with Gaussian frequencies. SIR is
fitted to those mapped features with the same training nuisance and response.
This is a finite randomized approximation inspired by kernel SIR; it is not an
exact RKHS estimator or the complete algorithm of a cited paper. See
[Wu, Liang and Mukherjee: regularized kernel SIR](https://www2.stat.duke.edu/~mukee002/reappointment/papers_under_rev/consistency_of_kernel_sliced_inverse_regression.pdf)
and [Rahimi and Recht: random features](https://pages.cs.wisc.edu/~brecht/papers/07.rah.rec.nips.pdf).

**Pilot-gradient directions.** Fit a ridge pilot to training residual response
using the same random cosine map. With pilot gradient g_i, eigendecompose
G = sum_i g_i g_i^T / n and retain leading directions in conditional-PC space.
For z(x)=sqrt(2/m)cos(xW+b), the analytic gradient is
-sqrt(2/m)[sin(xW+b) * beta] W^T. Project, then scale using training scores.
Unlike average derivatives, the outer product can retain a symmetric direction
whose signed mean gradient cancels. This specific regularized Fourier pilot is
a prediction variant; it is not full MAVE and has no imported consistency or
inference guarantee. The OPG motivation is discussed in
[Xia et al.: adaptive dimension reduction](https://doi.org/10.1111/1467-9868.03411).

The SIR/rff_sir comparisons hold learner, input site scheme and candidate grid
fixed. Pilot-gradient models use the same HGB/CatBoost heads as their SIR controls.

## 3. Paired oral/gut views

Training-only conditional PCs A,B are regularized with C_A+lambda I and
C_B+lambda I. The SVD of C_A^(-1/2) C_AB C_B^(-1/2) supplies canonical axes.
For k=6/10 (capped by dimensions), retain k paired sums, k paired differences
and k residual private-PC coordinates per site: 4k output columns.
The matched PC control retains 2k PCs per site, also 4k columns.

These coordinates describe statistical shared/discordant variation, not oral-to-gut
transfer, biological pathways or microbial effects. Classical motivation:
[Hotelling: relations between two sets of variates](https://doi.org/10.1093/biomet/28.3-4.321).

## 4. Complementary regression heads

**Local ridge.** Use 80/160 training nearest neighbors with Gaussian distance weights.
Normalize weights, fit weighted linear slopes with ridge 0.1/0.5 and an unpenalized
local intercept, and evaluate the local line at the query. Training/validation
scores always replay the saved role encoder rather than fitting a new projection.

**Split-sample forest-kernel local ridge.** Split training households into two
parts. Fit a random forest partition on one half, construct leaf weights from the
other, swap halves, and average kernels. Empty estimation leaves fall back to the
corresponding full estimation half. Fit the same local ridge under these weights.
The partition and leaf-label households are disjoint at this learner stage.
However, upstream supervised SIR was fitted on all training-role labels: this is
not a fully honest local-linear forest and does not inherit formal intervals.
It is a finite-budget prediction implementation inspired by
[Friedberg et al.: local linear forests](https://arxiv.org/abs/1807.11408), not the grf package.

**Normalized ReLU neural tangent kernel.** Append a constant coordinate, normalize
input covariance to a cosine, recursively apply normalized ReLU covariance and
derivative updates, and divide the tangent kernel by layers+1. Its diagonal is
one. Use 96/192 training-only centers and a PSD eigendecomposition to build
Nystrom features; fit ridge with alpha=n*penalty and an intercept. With all centers,
the result matches centered exact KRR to numerical tolerance. This is a fixed
infinite-width kernel prior, not a pretrained or trained finite deep network.
See [Jacot et al.: neural tangent kernel](https://arxiv.org/abs/1806.07572).

**Inductive graph residual smoothing.** Fit a global ridge trend, construct a
symmetric training-only nearest-neighbor affinity graph and combinatorial
Laplacian L scaled by mean degree. Solve s=(I+lambda L)^(-1) residual. Predict
by the saved ridge trend plus normalized training-neighbor interpolation of s.
Validation records never enter graph construction. Unlike transductive learning,
this procedure does not exploit the joint validation feature distribution.
Graph-smoothness motivation: [Zhou et al.](https://papers.nips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf);
the exact residual/inductive algorithm here is a separate implementation.

**Sparse Bayesian RBF bases.** Choose 64/128 training centers by seeded sampling,
concatenate Gaussian RBF responses with scaled linear inputs, standardize the
basis and fit sklearn ARDRegression. Relevance precisions shrink basis coefficients;
iteration-limit warnings are retained. These are statistical basis centers, not
independently validated biomarkers. It is not the exact sequential RVM algorithm.
See [Tipping: sparse Bayesian learning](https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf).

**SIR-space epsilon-SVR.** Use the existing sklearn SVR implementation on the new
role-fitted SIR or Fourier-SIR inputs. HGB, CatBoost and SVR themselves were tried
before; their controlled input/representation combinations, rather than their
names, distinguish this batch. Linear ARD and random-feature diet summaries also
appeared before; the microbial nonlinear projections/RBF dictionary here differ.

## 5. Historical bank, replay and fusion

Read the completed V13 prior-INNER snapshot (90 V10/V11 members), plus every V13
candidate's INNER predictions. Reconstruct the 38 V13 choices using current role
responses and retain those predictions, giving 128 historical columns. Check member
order, source/input fingerprints, closed receipts, processed rows/targets and choices.

Saved V11 and V13 positive-fusion objects are loaded, not re-optimized. The V13
member names are prefixed without altering learned coefficients/scalers. Only
after all new candidates and meta-functions are fixed are historical OUTER results
deserialized. Both saved replays must match their native baselines within 1e-6 mg/dL.

Fit convex/ridge/nonnegative stacks on oral, balanced, non-tree, new, historical
and combined pools; add sparse stacks for new/combined. For historical and combined
pools, also fit local-ridge, NTK and RBF-ARD stacks after training-meta-fold PCA.
All scores return to mg/dL using training means/scales. A nonlinear stack has no
per-member linear coefficients: all its inputs conservatively propagate fit flags.

Meta-CV is a training heuristic on selected INNER-OOF columns. Their base training
sets overlap meta partitions and base selection used the outer-training outcomes;
it is not fully nested independent meta-CV. Its loss is never advertised as external
performance. Current OUTER predictions remain untouched by these training choices.

## 6. Reporting and scientific limits

Preserve pooled outer-OOF R2/RMSE/MAE, matched positive and negative delta_R2,
household conditional intervals and all fit flags. The two repeats share people.
The default display orders by the smaller repeat R2, still an exploratory rule.
Intervals condition on learned predictions and exclude all adaptive model-search
and refitting uncertainty. Do not interpret them as confirming a selected winner.

This is supplementary prediction. Nonlinear main representations, better pooled
R2 or local differences do not establish diet-by-microbiome effect modification.
V12 owns its prespecified interaction estimands and tests. Data timing, measurement
error, missingness and repeated development still limit interpretation. A large
gain is a hypothesis to evaluate, not a promised outcome or a known data ceiling.
