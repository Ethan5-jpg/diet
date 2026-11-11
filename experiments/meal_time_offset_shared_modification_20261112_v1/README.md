# Shared microbiome interaction across meal timing offsets

The HPP meal logs were self-reported. The original meal archive has a
nominal start and two fixed alternative starts at minus and plus 15 minutes.
The analysis uses those existing windows and their matched outcomes.
[HPP meal prediction study](https://doi.org/10.1016/j.cdnut.2026.109429)

Each start retains its original within-person diet background, glucose
baseline and pre-meal trend. The joint oral and gut interaction block is
the existing 10 g carbohydrate contrast multiplied by the first five CLR
PCs at each site. For the shared fit, the three residualized squared-error
objectives are averaged with equal weight, and the original penalty of
0.03 is applied once. The training weight of a household is unchanged
from the separate-offset fit. This represents sensitivity to recorded
meal timing over a fixed interval. The true start is unobserved.

The primary evaluation is two-hour mean net glucose at the nominal start.
The matched comparison measures the difference in held-out R² between
the shared fit and the original nominal-only interaction fit. Both use
the same nominal diet background and the same rows. The two shifted
endpoints are fixed secondary checks. All separate-offset losses must
reproduce the original archived household folds before comparison.
CLR references and principal axes are estimated in training households.
Intervals use 1,999 paired household draws separately per partition.

The full run applies the shared coefficient to the existing additional
HPP households, reusing the original frozen microbiome coordinates and
reconstructing the exact previous baseline and offset-specific losses.
The shared coefficient is fixed before those outcomes are read by the
new analysis. This is a follow-up on households previously evaluated with
the original interaction. The prediction target remains the deviation from
each participant's observed mean across eligible meals.

Run from the installed experiment directory with a verified release.

```sh
python -B -m unittest discover -s .
python -B evaluate.py --package /study/package --commit COMMIT --run smoke-1 --smoke
python -B verify.py --run smoke-1
python -B evaluate.py --package /study/package --commit COMMIT --run discovery-1
python -B verify.py --run discovery-1
```

Use fresh run IDs. Model, losses, logs and receipts stay in TRE.
