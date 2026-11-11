# Meal carbohydrate source by audited NOVA classification

This discovery-only extension preserves the archived cohort, isolated meal windows,
macronutrient contrasts and the primary diet–glycemia analysis. It evaluates an
additional exposure: carbohydrate grams from W07-classified NOVA 4 foods relative
to grams from W07-classified NOVA 1–3 foods. The W07 crosswalk inherits the legacy
exact-name catalogue where available. Its labels have incomplete coverage and lack
independent adjudication of current products.

`audit.py` projects meal keys, dates, local hours and logged nutrients from the
archived eligible windows. It verifies the W07 receipt and the original food-event
source hash, reconstructs meal event membership and verifies the archived total
energy, carbohydrate, fat, protein and fiber. Unmapped FoodIDs remain unknown.
The audit reads no glycemic outcome columns. It reports support on meals with
zero unclassified carbohydrate and at least 90% classified energy, meals with
at least 80% coverage of both quantities, and meals with every event classified.

Repeated support requires at least two distinct meal dates with at least 10 g
from each class. The comparable-meal diagnostic requires two meals on different
dates, each dominated by a different source, within 10 g carbohydrate, 100 kcal,
10 g protein and three hours of local clock time. Date-transfer support uses the
archived calendar rule (at least four dates and four meals per half). The two
provider bread categories are assessed separately as a descriptive candidate
contrast. The broader `bread` category can contain multiple bread types.

Any later regression should keep the total-carbohydrate slope, the original
covariates and the generic carbohydrate-by-microbiome term in the matched
baseline. The target increment is the interaction with a 10 g replacement of
classified NOVA 1–3 carbohydrate by classified NOVA 4 carbohydrate, with
total and unclassified carbohydrate held fixed. Classifications and support
are evaluated before opening outcome columns. Meal logs support observational
contrasts and do not randomize dietary processing.

Run inside the study after installing the experiment folder and its pinned meal
dependencies.

```sh
python -B -m unittest discover -s /study/package/experiments/meal_processing_modification_20261112_v1
python -B /study/package/experiments/meal_processing_modification_20261112_v1/audit.py \
  --package /study/package --commit COMMIT --run audit-smoke-1 --smoke
python -B /study/package/experiments/meal_processing_modification_20261112_v1/audit.py \
  --package /study/package --commit COMMIT --run audit-discovery-1
```

Use a fresh run name for each audit. Keep run receipts and results in TRE.
