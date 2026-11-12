# Three-species Bacteroidales candidate in archived HPP meals

Takeuchi and colleagues reported reduced postprandial glucose in
high-fat-fed mice treated with Alistipes indistinctus, Alistipes
finegoldii or Bacteroides thetaiotaomicron. Those three species
define the panel before this analysis reads the meal outcomes. HPP
species abundance provides an observational candidate for the
carbohydrate slope. The original study administered specific strains
and the present analysis uses a newly defined species score.
[Takeuchi et al., Nature 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499599/)

Require all three exact terminal taxonomic names. Sum their relative
abundance fractions and use log(1 + sum / 0.0001), with mean and scale
estimated on training households. The hypothesized direction is a
smaller glucose slope at higher scores. In the full archived meal
sample, use the within-person recorded-energy and protein-adjusted
10 g carbohydrate contrast. Both matched models share the original
covariates and the panel and PC interactions with energy and protein.
Compare fixed-penalty panel, oral/gut PC, and joint carbohydrate
interactions. The pooled primary endpoint is recorded-start two-hour
mean net glucose, with peak, both shifted means and the pre-meal
value specified as secondary. Apply Holm adjustment over all ten
direct panel tests, including conditioning on the PC interaction.

Use the existing household partitions and both recorded-date
directions. Freeze all discovery fits before opening the previously
prepared additional-household outcomes. Report absolute R² and
paired household intervals on identical evaluation meals. Reuse of
these additional HPP households is a frozen transfer within the same
cohort. Report a separate calibration of the frozen panel direction.

```sh
python -B -m unittest discover -s .
python -B run_panel.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B run_panel.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --verify
python -B run_panel.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
python -B run_panel.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1 --verify
python -B run_panel.py --package STUDY_HANDOFF --commit COMMIT --run validation-1 --stage transfer
python -B run_panel.py --package STUDY_HANDOFF --commit COMMIT --run validation-1 --stage transfer --verify
```

Use the pinned study interpreter and keep all outcomes, scores, fold
losses and detailed results inside TRE.
