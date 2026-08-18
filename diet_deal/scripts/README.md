# Diet processing scripts

Scripts are separated into shared preparation and score-specific pipelines.

- `shared/`: common daily summaries and the reviewed food dictionary.
- `amed/`: AMED mapping, component-intake aggregation, and scoring.
- `ahei/`: AHEI mapping, component-intake aggregation, and scoring.
- `hpdi/`: hPDI mapping, component-intake aggregation, and scoring.
- `rdii/`: rDII nutrient preparation and scoring.
- `redih/`: rEDIH mapping, component-intake aggregation, and scoring.

Each script resolves source data and output paths from the `diet_deal` project
directory, so it can be run from any working directory.

`amed/05_calculate_amed_scores.py` retains the eight-component raw AMED score,
clips that raw score at the cohort 0.5th and 99.5th percentiles, applies the
total-energy residual method, and writes the adjusted score in the AMED scale.
It then writes `amed_energy_adjusted_score_z` using the eligible sample mean
and sample standard deviation (`ddof=1`) and assigns analysis quintiles from
the energy-adjusted score. `amed_score_winsorized` records the pre-regression
score; the obsolete post-adjustment truncation field is no longer generated.

For hPDI, run `04a_prepare_hpdi_weight_review.py` before the intake builder.
For each mapped event with missing weight, the script finds positive weights
for the exact same `food_id`, collapses repeated records to one median per
donor participant, excludes the target participant, and imputes the median of
the remaining participant medians. This gives each other participant equal
weight. Generated positive medians are marked `use_resolved_weight` and are
applied by `04_build_hpdi_intakes.py`; rows without another-participant donor
remain `unresolved` and are excluded from the primary score rather than scored
as zero. Previously approved manual values remain authoritative. The intake
and score outputs retain imputed-event counts and participant imputation flags
for complete-case sensitivity analyses.

Run `04b_audit_hpdi_missing_weight_direction.py` against the complete server
`hpdi_participant_component_intakes.csv` to report exact missing-weight event
counts for positive and reverse components, affected-participant overlap, and
component-level concentration. The audit is read-only and does not change any
score input or output.

`hpdi/05_calculate_hpdi_scores.py` retains the 18-component raw score, clips
that raw score at the cohort 0.5th and 99.5th percentiles, applies the total
energy residual method, and writes the adjusted score in the original hPDI
scale. It also writes `hpdi_score_energy_adjusted_z`, standardized with the
eligible sample mean and sample standard deviation (`ddof=1`), plus analysis
quintiles based on the energy-adjusted score. The parameter and QC outputs
record the clipping cutpoints, residual-model slope, Z-score mean/SD, and final
quintile counts.

## AHEI / mAHEI-7

The current HPP protocol is a seven-component modified AHEI, not the complete
11-component AHEI. Vegetables, fruit, whole grains, sugar-sweetened beverages
plus fruit juice, nuts plus legumes, red plus processed meat, and alcohol each
contribute 0--10 points. Sodium, trans fat, and PUFA are excluded from both the
numerator and denominator; they are never filled with zero or scored as zero.
Long-chain EPA+DHA was already unavailable in the paper. The unscaled raw
field is `mAHEI7_raw_0_70`; do not label or publish a rescaled 0--100 version
as the formal AHEI.

Run the AHEI scripts only after the reviewed AMED and hPDI mappings and the
AMED participant intake output exist:

```bash
python scripts/ahei/00_audit_ahei_inputs.py --enforce-expected-counts
python scripts/ahei/00b_scan_ahei_lookup_sources.py --file-limit 0
python scripts/ahei/03_prepare_ahei_mapping.py
python scripts/ahei/04_prepare_ahei_score_intake.py
python scripts/ahei/05_calculate_ahei_scores.py
```

`03_prepare_ahei_mapping.py` reuses the reviewed hPDI mapping to separate
whole fruit from fruit juice and to identify vegetables, whole grains, nuts,
legumes, and sugar-sweetened beverages. It reuses the reviewed AMED mapping
for red and processed meat. The two beverage groups and nuts/legumes are then
combined as required by AHEI. Any source mapping still marked
`review_required` stops the pipeline.

`04_prepare_ahei_score_intake.py` preserves every original participant-day,
including days containing only an all-label-missing event. Such events are
excluded from component totals without changing the day denominator. The
approved food proxies are 30 kcal/serving for vegetables, 70 for fruit, 70 for
sugar-sweetened beverages plus fruit juice, 220 for red plus processed meat,
and 28.35 g/serving for nuts plus legumes. Explicit whole-grain food weight is
retained as `mean_daily_whole_grain_proxy_g`; it is not described as strict
dry whole-grain grams. Approved hPDI event-level weight resolutions are reused
when present. Alcohol and total energy come from the existing AMED participant
output, with 14 g ethanol per standard drink.

`05_calculate_ahei_scores.py` applies the published continuous component
endpoints, including 75 g/day whole grains for women, 90 g/day for men, and
the sex-specific nonlinear alcohol rule (nondrinkers receive 2.5 points). It
then winsorizes the raw 0--70 score at P0.5/P99.5, applies residual-method
energy adjustment, calculates a sample-SD Z score (`ddof=1`), and assigns
tie-safe quintiles. At least two complete participants with positive and
variable energy are required for the residual adjustment. The small local CSV
snapshots are schema examples only and may legitimately stop at this check;
run the complete pipeline on the server-scale baseline data.
