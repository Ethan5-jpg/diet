# Carbohydrate modification within repeated FoodID templates

This exploratory branch tests the established 10 g carbohydrate contrast
against the two-hour net glucose increment within the same person's
recorded food combinations. A controlled feeding study found appreciable
response variability even for duplicate presented meals.
[Hengist et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11747189/)

Verify archived window and event totals before identifying a template by
the sorted multiset of nonmissing FoodIDs. Repetition requires different
calendar dates. Quantify the within-template carbohydrate variation that
remains after adjusting for recorded log-energy and protein terms, and
count groups with a different-date 10 g contrast within 50 kcal, 5 g
protein and two local clock hours. These paired counts are diagnostics.
FoodID repetition does not establish identical portions or ingredients.

The matched comparison demeans the original meal design and outcome by
person and template. Both models include oral/gut PC interactions with
log energy and protein. The added block is the centered 10 g carbohydrate
contrast by the five training-derived CLR PCs at each site, fitted with
the original ridge penalty of 0.03. Reuse household partitions, weight
people equally and report 1,999 paired household draws. For calendar
transfer require two different dates of the same template in each
person-level half and center each template-half separately. Report both
date directions with household holdout. The scored target is the
deviation from the person's observed template or template-half mean.

The primary endpoint is recorded-start two-hour net glucose. Peaks,
timing shifts and pre-meal glucose retain their archived definitions.
The pooled direct comparison adjusts across five endpoints and uses the
number of absorbed template effects in its clustered covariance.
Neither cohort reuse nor repeated household splitting establishes
independent validation. Results remain observational.

Run from the isolated TRE experiment directory with the pinned study
interpreter after verifying the deployed source hashes.

```sh
python -B -m unittest discover -s .
python -B analyze_template.py --package STUDY_HANDOFF --commit COMMIT --run support-1 --audit
python -B analyze_template.py --package STUDY_HANDOFF --commit COMMIT --run support-1 --verify
python -B analyze_template.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B analyze_template.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --verify
python -B analyze_template.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
python -B analyze_template.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1 --verify
```

Keep all household losses, result tables and run records inside TRE.
