# Daily diet–microbiome modification of CGM glucose

This discovery study asks whether the same person's daily diet–glucose association differs across measured microbiome profiles. It uses the frozen Experiment 2 discovery cohort and both existing household partitions, with exact person, visit, and calendar-day matching. Daily mean CGM glucose is in mg/dL.

## Exposures and contrasts

- Carbohydrate replaces fat at fixed protein share and recorded energy, from 45% to 55% of accounted macronutrient energy. Fractions use the complete daily carbohydrate, fat, and protein sums with 4/9/4 kcal per gram. They describe accounted macro energy and do not assume that recorded total energy equals the sum.
- Fiber density changes from 10 to 15 g/1000 kcal at fixed macronutrient shares and recorded energy. Both exposures use the same eligible records. Each participant needs at least three common days. The report includes empirical within-person variation and counts spanning both contrast endpoints.
- The carbohydrate comparison was motivated by the high/low carbohydrate crossover and microbial score in [Fu and Gou et al., eBioMedicine](https://doi.org/10.1016/j.ebiom.2024.105483). HPP is observational and has a different time scale and microbiome sampling scheme. This batch uses HPP-derived microbial features.

## Analysis

Participant fixed effects absorb time-invariant factors. Each person has total weight one. The common background has linear and quadratic diet terms, carbohydrate–fiber and carbohydrate–protein terms, energy, weekday, recording day, and diet interactions with age and sex. All comparisons use that same background. Each candidate adds the specified dose-by-microbiome block. In the joint candidates, carbohydrate and fiber modification are fitted together.

The designated direct test is the household CR1 omnibus F-test for the five gut CLR PC × linear carbohydrate terms. The direct-test report includes the raw and Holm-adjusted P values for every route. For the single-Shannon routes it also includes the Q75-minus-Q25 modification estimate and its pointwise interval. The direct tests are conditional on the fitted unsupervised microbial features and are discovery results. Modeled microbial contrast spread is reported in mg/dL.

Two fixed ridge penalties per route are compared with the matched no-interaction prediction in two five-fold household partitions. Microbial CLR filtering, zero handling, scaling and PCA are fitted in each training fold. Each validation person's days are centered within that person, including the validation glucose when scoring the centered target. This evaluates prediction of within-person deviations. Penalties are reported individually and there is no best-route selector.

The window is concurrent, and the microbiome profile may come from another date in the same visit. The analysis estimates conditional associations. Independent confirmation and time-ordered or intervention evidence require separate data and designs.

## Run inside HPP

Place the release in `<handoff>/experiments/diet_daily_modification_20261112_v1` and compare its source SHA-256 values against the pushed commit. Use the previously verified V12 interpreter. The input adapter checks the Experiment 2 registration and frozen input fingerprints. No confirmation outcomes are used.

```bash
h=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1
d="$h/experiments/diet_daily_modification_20261112_v1"
p="$(cat "$h/experiments/diet_effect_modification_20261109_v12/.python_path")"
cd "$d"
"$p" -B -m unittest -v test_daily
"$p" -B run.py --package "$h" --commit FULL_COMMIT --run smoke-1 --smoke
"$p" -B run.py --package "$h" --commit FULL_COMMIT --run discovery-1
```

The smoke run exercises the real discovery input adapter and first outer fold. The full run needs all ten fold receipts, the aggregate summary, and a completion marker. Add `--resume` to continue an interrupted run after confirming that its recorded process has exited. The source and input registration must match and the run lock must be clear. Store run logs, receipts, and results inside the study.
