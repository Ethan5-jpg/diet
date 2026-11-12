# Documented milk product support in archived meals

This outcome-blind audit asks whether the archived HPP meals can support a
future comparison of recorded milk products and the carbohydrate response
across gut Bifidobacterium profiles. Milk intake, lactase variation and
Bifidobacterium have been associated in cohort data with long-term outcomes.
[Primary cohort study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11097298/)

Only the provider short names `Milk` and `Low fat Milk` define the milk
category. Literal free and reduced lactose declarations use the recorded
product name. Unspecified milk has unknown lactose status. Mixed or
ambiguous meals are reported separately. A meal without a listed milk
item can contain milk ingredients. No lactose quantity is inferred.

Reconstruct meal membership and verify all archived nutrients and dates.
Audit within-person overlap over 15–60 g of meal carbohydrate, requiring
two observed dates in each category and a common carb interval of at least
10 g. Retain only meals inside the common interval and verify that both
categories still have two dates. Count detection using
exact terminal species paths in the archived Bifidobacterium genus.
This audit uses frozen discovery rows and reads no glucose outcome columns.

Run from the isolated TRE experiment directory using the pinned study
interpreter. Test locally and verify the deployed source hashes first.

```sh
python -B -m unittest discover -s .
python -B analyze_support.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B analyze_support.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
```

The registration, summary and checksum marker remain in TRE.
