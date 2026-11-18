# V6 method specification

## Target and estimand

The analysis replays the V11 discovery anchor: 4,427 archived discovery index rows,
3,343 evaluated people in 3,059 households, person-level D_Glu and fiber density.
No additional eligibility or outcome-dependent filtering is introduced. Archived
training-role preprocessing and training-only outcome limits remain in use.
The original two five-fold household assignments and three inner household folds
are replayed. Both outer repeats contain the same evaluated people.

The reported contrast is predictive, not a causal partition:

\[R^2=1-\mathrm{SSE}/\mathrm{SST},\quad
\Delta R^2=({\mathrm{SSE}_0-\mathrm{SSE}_1})/\mathrm{SST},\quad
P=100\Delta R^2/R^2_1.\]

All quantities within a repeat use concatenated out-of-fold predictions on the
same people, outcome and household assignment. R2s and ratios are not averaged
across folds. Historical V11 scores are reference values only. A ratio of
prediction scores is neither a mediated proportion nor the biological fraction
of a diet effect. Imperfect additive main-effect approximation can also leave
signal that product corrections exploit.

## Shared main-effect baseline

The left design block is the unchanged archived host/diet/context background.
The right block contains training-fitted, scaled PC1–40 from each microbiome,
oral/gut Shannon and dominance and their squares, the fixed fermentation score
and its square, six exact random-feature representations (three sites by
PC10/20, 48 features each), six PC hinge representations (PC5/10 by three sites),
and the CLR, balance and presence/abundance views described below.
Rank-deficient synthetic inputs use the available
nonconstant PC rank and record it. Native rank is recorded, too.

HistGradientBoostingRegressor interaction constraints allow paths within the
left block or within the right block; age/sex may be shared. Diet and microbial
features cannot appear in the same baseline tree path. There are two archived
baseline configurations; pooled inner SSE selects one, shared across all 78
routes in that outer fold. Changing the baseline by route is disallowed.

## Interaction families

Every correction uses a centered training dose basis multiplied by centered
microbial predictors, without a fitted intercept or independent microbial main
term. The correction is exactly zero when the entire centered dose vector is
zero. For two-dose-column candidates, both archived centered basis columns are
used: the linear column and the archived nonlinear spline column. The small
generated fixture uses a quadratic stand-in, while the full archive fixture
replays the real spline code. Zero of the linear component alone does not force
the nonlinear component to zero.

- Ridge: PC5,10,20,40 with the linear dose, plus PC10,20 with the two archived dose bases.
- PLS: three uncentered PLS1 latent directions of the interaction product
  design, then ridge. No implicit intercept from centered sklearn PLS is used.
- Elastic Net: alpha is a fixed fraction (1, .1, .01) of training alpha_max;
  l1_ratio=.5, no intercept. Nonconvergence stops the run and is not a zero fit.
- RBF modifier: a fixed-seed Gaussian Fourier mapping of PC10/20 (48 columns,
  gamma=1/input_width), followed by dose multiplication and ridge. The exact
  mapping also enters the shared baseline as microbial main effects. It is
  not a black-box unrestricted tree of dose plus microbes.
- Fixed Shannon and three-species panel comparators: linear-dose ridge.
- Compositional views: training prevalence >= .1, descending training CLR
  variance selects up to 16 taxa per site. With fixed log floor 1e-6 in fraction
  units, build selected CLR coordinates, all pairwise balances, and log abundance
  plus presence indicators. Each receives ridge and Elastic Net linear-dose
  corrections in oral, gut and combined sites. These are finite selbal-inspired
  views, not the original greedy selbal algorithm or a hurdle likelihood.
- Tensor-inspired corrections: the original frozen two dose columns times
  centered PC5/10 linear and training-quartile hinge columns, with ridge. Their
  exact microbial marginal columns enter the baseline. Dose endpoints continue
  to use the archived dose basis; no new dose knots are fitted, so the old delta
  is used correctly. This is not Wood's full anisotropic tensor smoothing penalty.
- Tree varying coefficients: only PC10/20 microbial inputs partition depth-two
  trees. Forty centered, ridge-shrunk updates learn g(Z), and correction is
  centered-dose * g(Z). The training mean of g is zero. No separate g main term
  is added. PCs enter the baseline nonlinear microbial block; outcome-learned
  leaf indicators themselves are not separately added to that baseline. Thus
  main-effect approximation error may still contribute to predictive gain.

The three species are Ruminococcus bromii, Bifidobacterium adolescentis and
Eubacterium rectale (Agathobacter rectalis accepted as its alias). Terminal
species labels must resolve exactly once. The score is
`log1p(sum(relative abundances)/0.0001)`, the unchanged V3 definition. Relative
fractions must sum to approximately one; percentage units are rejected.

There are 68 families × 3 penalties = 204 nonzero settings. Each family includes
zero as an additional inner choice. For combined-site candidates k PCs from
each site are concatenated, rather than dropping the second site.

## Nested learning and combinations

Each outer training role has three inner train/validation roles. Feature
transforms fit only inner training predictors. Household-cross-fitted nuisance
learners supply training residuals; their transforms are reused from the
enclosing training role rather than refitted in each nuisance subfold. No
double-robustness, nuisance cross-fitting theorem or formal interaction test is
claimed. Correction learners never receive the current validation targets.

Each family chooses minimum summed inner-held-out SSE, including zero (zero
wins numerical ties). The primary nested_selector chooses across all 204
settings and zero by the same rule. With fixed positive baseline R2, maximizing
positive delta/full is monotone in full R2, so inner SSE is the appropriate
selection objective. This avoids choosing noisy fold-specific ratios.

Five deterministic stack constructions first choose each component family's
penalty, then average within algorithm class and average classes equally.
Balanced uses ridge/PLS/Elastic Net/RBF; gut restricts to gut; linear omits RBF.
Inner SSE chooses scaling .25/.5/1 or zero. The same inner outcomes are reused
for component and scale choices, so inner stack error is optimistic. The whole
procedure is independently evaluated in the outer fold. The original three
stacks retain their PC-only algorithm classes. A compositional stack averages
taxa/balance/two-part classes equally; the expanded stack averages six classes
(original PC, taxa, balance, two-part, tensor and varying-coefficient trees).
Five selectors and five stacks bring the total to 78 reported routes. The
original-family selector searches only the original 38 families on exactly
the same expanded baseline. Its paired comparison against the expanded
selector and added stacks is separately reported; its comparator already
contains interactions and is not the main no-interaction contrast. Final correction and
10-to-15 fiber contrasts use exactly the same fitted coefficients.

## Uncertainty and ranking

The 1,999 paired bootstrap draws resample households, jointly preserving all
outcomes and both models' errors for their members. Each draw recomputes SST,
baseline/full R2, delta and their ratio. Intervals are percentile intervals
conditional on fitted OOF predictions; they omit model refitting and search
uncertainty. A separate actual-target-shift test checks that outer targets have
no effect on fitting or selection.

Negative gains remain signed. Negative/near-zero R2 denominators are excluded
from positive-share ranking. Ratio intervals are withheld when the denominator
interval reaches zero or numerical-zero draws occur; the nonpositive draw
fraction remains visible. Point-positive exploratory maxima and denominator-
stable maxima are separate fields. Neither constitutes a multiplicity-adjusted
best-model claim. No subgroup, exposure, outcome or denominator is searched to
inflate the ratio. The previously scored V5 reserved cohort is not accessed.

## Historical audit and provenance

182 shipped historical aggregate pairs are merged with existing allowlisted
V1 summaries, V3/V4 completed logs, V5 reserved summary and legacy V5 round
summaries. Source labels and stages remain attached; duplicate logs are not
independent replications. Missing summaries remain missing. Legacy same-CV
tuning/reporting maxima stay exploratory and are not new V6 validation.
Historical delta intervals are retained but never divided by a fixed denominator
to fabricate ratio intervals. `adapters/` contains exact copies of V2 input and
model utilities and the V3 score; their original paths and hashes are supplied
in `adapter_provenance.json`. Native input sources must match the archived V11
registration. V6 registers the release, sources and full fixed grid before fit.
The expanded release also binds Python/package versions, requested thread count
and actual numerical thread pools to its registration. Completed runs require
the full expected set of ten fold checkpoints and matching result schemas.

## Implementation references

- [scikit-learn interaction constraints](https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html): feature-set constraints underlying the additive baseline.
- [Random Fourier feature mapping](https://scikit-learn.org/1.4/modules/generated/sklearn.kernel_approximation.RBFSampler.html): basis for the explicitly implemented fixed random mapping.
- [R2 definition and negative scores](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.r2_score.html).

These versioned documentation pages currently render 1.4.2; execution remains
pinned to the project's existing 1.4.1.post1 environment, tested locally.

Primary-paper rationale for the added structures and their precise scope is
provided in `LITERATURE_AND_EXPANSION.md`.
