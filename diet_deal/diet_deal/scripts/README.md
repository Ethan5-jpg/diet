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
