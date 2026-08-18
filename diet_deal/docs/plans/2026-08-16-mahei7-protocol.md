# HPP mAHEI-7 Protocol Update

**Decision date:** 2026-08-16

**Goal:** Apply the user's decision to exclude sodium together with trans fat
and PUFA, and keep the baseline missing-label count separate from the broader
food-mapping dictionary count.

**Score contract:** The included components are vegetables, fruit, whole
grains, sugar-sweetened beverages plus fruit juice, nuts plus legumes, red plus
processed meat, and alcohol. Each contributes 0--10 points. The raw score is
`mAHEI7_raw_0_70`, with range 0--70. Sodium, trans fat, and PUFA are excluded
from both numerator and denominator and are never scored as zero.

**Audit contract:** The audit remains restricted to `cohort=10k` and
`research_stage=00_00_visit`. In this baseline scope, 333 missing-label
`food_id` values account for 8,052 excluded events affecting 2,663
participants. The existing 368 count is retained only as a reference for the
broader food-mapping dictionary. Participants and 125,374 original logging
days remain in the index.

**Sodium handling:** Continue reporting `sodium_mg` coverage as informational
QC so the omission is transparent. Do not recover missing sodium, calculate
sodium deciles, remove participants because of sodium, or score sodium as
zero.

**Verification:** Tests must assert seven included component rows, three
`excluded_by_user_protocol` rows, a 0--70 raw range, the 333 baseline expected
count, the separate 368 dictionary reference, and Python 3.7-compatible syntax.
