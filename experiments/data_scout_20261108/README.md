# Server data scouting, 2026-11-08

This is a read-only preparation step before deciding on another model batch. The user runs terminal commands and returns screenshots. Do not connect to the server or transfer participant data.

Step 1 reads only the metadata of six explicitly listed Parquet files. It prints row counts, column counts and selected field names, capped at 32 lines. Upstream source tables may include broader populations; no values from those tables are read. Field filters and omitted schema lines are explicit. A missing table does not imply that the underlying variable is absent from the project.

The command uses the working V8 environment. No package installation, model fitting, server file write or existing-result modification is needed. Schema availability does not establish valid measurement timing, individual coverage or eligibility as a model input.

After the first screenshot, select the next aggregate queries using the actual schema. Planned questions are discovery-only missingness and usable sample counts; dietary recording breadth and repeatability; exact food/CGM day overlap; phenotype variability; and microbiome collection timing. Do not perform all possible outcome or feature screens merely because the schema exists.

`server_step1.sh` is the exact copyable command. `research_records/` will retain screenshots, visible transcriptions, definitions and limits. Step 1 has returned a six-table schema screenshot; see the local research index. Step 2 returned the same 3343-person cohort, covariate coverage and descriptive calendar timing. See the screenshot index and Chinese summary.

## Step 2: fixed-cohort aggregates

`profile_step2.py` retains the existing V4 repeat-0 eligible discovery rows and prints repeat-1/shared sizes. It projects only specified columns and filters upstream values by participant ID, cohort and visit before DataFrame materialization. It reports native-glucose and diet distributions, cleaned activity/sleep coverage, alcohol-label availability, prepared-common-day coverage, and selected O/G sample calendar dates relative to the CGM envelope. No event-level nutrient table is scanned, no model is fitted and no server result is modified.

The normal report has 20 lines. The glucose distribution is raw, not a recalculated CV score. Absent prepared common-day membership does not prove that no shared day exists; the prepared table already imposes its own eligibility. Calendar-envelope overlap does not establish exact day overlap or event timing. Sample dates come from the selected-sample visit table, not the candidate registry. Extra covariate availability does not prove that its measurement timing is suitable for adjustment.

`server_step2.sh` is the exact terminal command; `research_records/step2_validation.json` records six actual synthetic checks. The user can paste this command without uploading files. Future additions to the host condition set must be shared by matched null and interaction comparisons.

## Step 3: daily nutrient availability and same-window consistency

`nutrients_step3.py` joins existing W06 daily complete sums to the current QC-approved energy days on identity + normalized calendar date. Energy and fiber anchors must match before any nutrient interpretation; missing day joins or anchor mismatches are reported and stop the summary. No event-level table is scanned. Source values are filtered to the frozen repeat-0 discovery identities before materialization.

The normal output is 18 lines: valid logged days, people with >=3 and >=7 valid days, equal-day person-mean densities, and chronological odd/even consistency. Complete sums refer to logged event fields, not proven complete daily intake. Odd/even rho is a same-window descriptive Spearman correlation, shown only with >=3 days in each half, >=10 paired people and variable half-means. Missing is distinct from zero, including alcohol. It is not an outcome association or predictive score.

`server_step3.sh` contains the exact copyable, standalone terminal command using the V8 environment. No upload or installation is needed. Six actual synthetic checks are recorded in `research_records/step3_validation.json`. The user returned Step 3 on 2026-11-08 at 20:07:35: 43158 daily matches, no energy/fiber mismatches, and joint three-macronutrient >=3-day availability for 3330/3343 people. See the screenshot index. No new model has been fitted in this scouting phase.
