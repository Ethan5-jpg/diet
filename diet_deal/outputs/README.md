# Diet processing outputs

This directory separates shared diet preparation from score-specific artifacts.

## Shared outputs

- `01_daily_summary/`: participant-day nutrient totals, participant summaries,
  and preparation QC shared by all five scores.
- `02_food_dictionary/`: reviewed `food_id` dictionary, category summaries,
  conflicts, overrides, and QC shared by food-group-based scores.

## Score-specific outputs

The following directories each contain the same five score folders: `amed/`,
`ahei/`, `hpdi/`, `rdii/`, and `redih/`.

- `03_score_mapping/`: component definitions, food or nutrient mappings, and
  mapping QC.
- `04_score_intakes/`: participant-day and participant-level component
  intakes and QC.
- `05_diet_scores/`: component scores, total scores, thresholds, and scoring
  QC.

The five target scores are AMED, AHEI, hPDI, rDII, and rEDIH. Outputs from
one score must not be placed in another score's folder.

Do not place source CSV files here. Source data remain under `Data/Transfer/`.
