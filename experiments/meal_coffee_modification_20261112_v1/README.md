# Coffee co-intake and microbiome modification of the carbohydrate slope

The HPP coffee study identified the archived `Clostridium_phoceensis`
label as associated with usual coffee intake and discusses its
correspondence to `Lawsonibacter asaccharolyticus`. A multi-cohort study
also found coffee associations with that species and tested coffee in
culture. These studies motivate the choice of marker for a separate
meal-level interaction test. [HPP coffee study](https://pubmed.ncbi.nlm.nih.gov/41047001/)
[Coffee and Lawsonibacter study](https://doi.org/10.1038/s41564-024-01858-9)

The food source is the same hashed event table used in the archived meal
preparation. Coffee co-intake uses exact trimmed, case-folded short names
`Coffee` and `Cappuccino` in the archived meal window. Controls must have
observed names without a documented or ambiguous coffee label. The
low-calorie sensitivity uses `Coffee` only, caps combined coffee event
energy at 20 kcal, and requires complete names with no ambiguous coffee
labels. This is an analysis of recorded beverages. Caffeine dose and
decaffeination status are unresolved. Exposure reconstruction checks all
five archived nutrient totals before glycemic outcomes are opened.

For both co-intake groups, each participant must have at least four meals
on two different days and at least a 10 g carbohydrate range within the
fixed 15–60 g interval. The nearest-pair diagnostic checks overlap within
10 g carbohydrate, 100 kcal, 10 g protein and three hours on different
dates. Logged usual coffee frequency is estimated from earlier observed
dates, with one success and one failure added before the first date.
The current and future meal dates cannot enter that estimate.

The baseline retains the original carbohydrate and other dietary
covariates. It adds coffee source energy and carbohydrate, clock time and
preceding interval interactions, and the lower-order terms for coffee,
carbohydrate and the marker, including interactions with logged coffee
frequency. The target is the change in the microbial modification of a
10 g carbohydrate contrast in meals with coffee relative to meals
without coffee. The primary outcome is net 0–120 minute mean glucose.
Peak, offset-start and pre-meal results are secondary.

The exact archived species leaf supplies detection and standardized
log-positive abundance. Undetected samples have zero conditional
abundance. Each matched test conditions on the other component of the
three-way term. The primary effect contrasts the 75th and 25th percentiles
of detected abundance in the documented-coffee subset. Both scales are
estimated using training households for prediction. The 20 associations
use household clustering and one Holm family.

Two archived household partitions are evaluated with 1,999 paired
household bootstrap draws. Date transfer requires at least two meals of
each context in each calendar half and centers the within-person target
separately in each half. The date-transfer task conditions on the observed
within-person mean in the evaluation half.

```sh
python -B -m unittest discover -s /study/package/experiments/meal_coffee_modification_20261112_v1
python -B analyze_coffee.py --package /study/package --commit COMMIT --run smoke-1 --smoke
python -B analyze_coffee.py --package /study/package --commit COMMIT --run discovery-1
python -B verify.py --run discovery-1
```

Run from the installed experiment directory with fresh run IDs.
Receipts, logs, checkpoints and results remain in TRE.
