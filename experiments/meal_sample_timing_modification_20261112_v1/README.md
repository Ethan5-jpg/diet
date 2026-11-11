# Stool handoff proximity and carbohydrate modification

Shen et al. reported a timing-dependent contribution of microbiome features
to postprandial prediction in meal-level validation in HPP. Their
participant-held-out results showed limited improvement. This extension
asks whether the within-person carbohydrate association varies more with
the gut microbiome when the meal falls closer to the documented stool
handoff. [HPP prediction study](https://doi.org/10.1016/j.cdnut.2026.109429)

The source dictionary defines `collection_date` as the date the sample
was given. The analysis uses recorded handoff proximity as a measurement
context. It does not establish the biological collection date or
prospective availability of the profile. The run verifies the sample
registry and abundance matrix against W13 manifests, links by frozen
participant/visit keys and sample ID, and checks every linked species
value against the archived gut matrix in the archived precision. The
monitoring-day index starts on the local calendar date of the first
verified CGM observation for the archived connection.

Near meals are within seven calendar days of the handoff date in either
direction. Each included person must have at least four meals on at least
two distinct dates and at least 10 g of within-context carbohydrate range
in each context. The primary population uses all archived doses. A fixed
15–60 g restriction checks sensitivity to a narrower dose range.
Context selection uses projected dates and food values before loading
the glycemic outcomes. Both share the original meal preparation and
carbohydrate main-effect covariates.

The matched baseline includes the near indicator and its diet terms,
smooth monitoring day, carbohydrate-by-monitoring day and
carbohydrate-by-handoff schedule terms, and the common linear and
quadratic carbohydrate-by-microbiome interactions. It also allows the
microbial dose slope to vary with quadratic monitoring day and linear
handoff schedule. The target is the increment for a 10 g carbohydrate
contrast in the near context relative to the far context beyond those
terms. The primary modifier uses the first five gut CLR principal
components and the primary outcome is mean net 0–120 minute glucose at
the logged start. Gut Shannon diversity, peak, offset starts and pre-meal
glucose are secondary. All 20 clustered association tests use one
Holm family.

Training-only transforms and the two archived household partitions
are used for prediction. The models train on both contexts and compare
matched held-out loss pooled and by context using 1,999 paired household
bootstrap draws. Pooled and context-specific predictions use the same
within-person centering. They evaluate within-person deviations
conditional on the participant's observed mean across eligible meals.

```sh
python -B -m unittest discover -s /study/package/experiments/meal_sample_timing_modification_20261112_v1
python -B analyze_timing.py --package /study/package --commit COMMIT --run smoke-1 --smoke
python -B analyze_timing.py --package /study/package --commit COMMIT --run discovery-1
python -B verify.py --run discovery-1
```

Run from the installed experiment directory. Use unique run IDs.
Receipts, logs, checkpoints and model outputs remain inside TRE.
