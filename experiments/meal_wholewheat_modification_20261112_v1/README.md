# Bread carbohydrate source and Lachnoclostridium

Pierson et al. reported an exploratory baseline Lachnoclostridium-by-diet
interaction after two weeks of whole-wheat or refined-white bread in a
controlled crossover trial of adults with prediabetes. Higher baseline genus
abundance was associated with a more negative whole-wheat versus white
contrast for the post-intervention oral glucose tolerance test.
[Trial report](https://doi.org/10.3389/fnut.2026.1907908)

This is a directional observational extension using the archived HPP discovery
meal windows and 0–120 minute continuous glucose responses. HPP's broader
`bread` label encompasses multiple bread types, and the meal exposures differ
from the trial's sustained interventions. The input requires separately
documented `bread_wholewheat` and `bread` categories. Each participant must
have at least two dates with at least 10 g of each source. The script
reconciles all five archived nutrient sums and checks the earlier
outcome-blind support receipt. It leaves the primary HPP analyses intact.

The primary contrast replaces 10 g of carbohydrate from `bread` with 10 g from
`bread_wholewheat` at fixed total carbohydrate, combined bread carbohydrate,
and the original logged covariates, including energy, protein and fiber.
The matched baseline includes the bread-source contrast and allows the genus
to modify linear and quadratic total carbohydrate and total bread
carbohydrate. The primary modifier is genus detection, defined by a positive
sum of the relative abundance of terminal species with the exact genus path.
A separate secondary term tests positive log-abundance beyond detection.
Detection scaling, and positive log-abundance centering and scaling, use the
training households. Undetected samples have zero for the conditional
abundance term. The main endpoint is the two-hour mean net glucose response
at the logged start, with peak, time-offset and pre-meal checks.

Evaluation reuses both archived household partitions. Date transfer requires
both sources in both halves, at least four distinct dates and at least four
meals per half. Outcomes and regressors are centered separately in each half.
For comparability, a secondary analysis takes one pair per eligible person.
The pair must be recorded on different dates and have different dominant
bread sources. It minimizes squared scaled distance among pairs within 10 g
carbohydrate, 100 kcal, 10 g protein and three hours local time. This subset
evaluates detection. Household-clustered associations are two-sided, with one
Holm family of 15. Held-out changes are summarized with 1,999 paired household
bootstrap draws per partition.

```sh
python -B -m unittest discover -s /study/package/experiments/meal_wholewheat_modification_20261112_v1
python -B analyze_bread.py --package /study/package --commit COMMIT --run smoke-1 --smoke
python -B analyze_bread.py --package /study/package --commit COMMIT --run discovery-1
python -B verify.py --run discovery-1
```

Run the last three commands from the installed experiment directory. Use
unique run IDs. Inputs, results, checkpoints and logs stay inside TRE.
