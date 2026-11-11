# Previous meal carbohydrate and microbial modification of the current meal

Controlled feeding studies motivate accounting for the preceding meal when
comparing glycemic responses. This experiment measures recorded total
carbohydrate, with other nutrients and timing included as covariates.
[Second-meal trial](https://pubmed.ncbi.nlm.nih.gov/16600933/)
[HPP prior-intake prediction study](https://doi.org/10.1016/j.cdnut.2026.109429)

The existing meal-log anchor and isolation algorithm defines the current
meal and the previous recorded episode. The code reconstructs episodes
from the hashed event table and verifies the archived preceding interval
for every current anchor. The outcome-blind support audit is replayed.
Analysis requires complete previous energy and all four logged nutrients,
at least 50 kcal in the preceding episode, an elapsed interval of 135–360
minutes, and 15–60 g current carbohydrate. Each person needs at least two
different dates with previous carbohydrate at or below 20 g and two with it
at or above 40 g. The model uses continuous previous carbohydrate across
all eligible meals of those people.

Both compared models retain the archived within-person current carbohydrate
background, the lower-order diet terms, and the common linear and quadratic
microbial modification of current and previous carbohydrate. The diet
terms include previous energy, protein, fiber, elapsed time, and current
meal clock time. The added term is the interaction of the oral and gut
five-PC blocks with the product of the current and preceding carbohydrate
contrasts, each in units of 10 g. The primary outcome is the mean net
two-hour response at the recorded meal start. Peak, two timing shifts and
pre-meal glucose are secondary. Pre-meal glucose can reflect carryover
from the preceding meal. The pooled household-clustered tests for the joint
block form a five-outcome Holm family.

A secondary one-dimensional interaction uses the direction of ordinary
current-meal microbial modification. That direction, CLR references and
PC axes are estimated with the relevant training data only. Both versions
share the same matched lower-order model and the original ridge penalty
of 0.03. The two archived household partitions and two chronological
directions use paired household bootstrap intervals. The date-transfer
sample has both preceding-carbohydrate categories in both halves and
centers within person and half.

The additional households use the verified new-household meal preparation.
The exposure screen and selection project food and timing columns only.
After discovery evaluation, the coefficients and microbial scores are
frozen and checksummed before opening the additional households' outcomes
for this analysis. The additional-household comparison requires at least
ten households for resampling. It is a transfer test within HPP. Logged
meal episodes are observational measurements and omit unrecorded intake.

```sh
python -B -m unittest discover -s .
python -B analyze_previous.py --package /study/package --commit COMMIT --run smoke-1 --smoke
python -B verify.py --run smoke-1
python -B analyze_previous.py --package /study/package --commit COMMIT --run discovery-1
python -B verify.py --run discovery-1
```

Run from the installed experiment directory with fresh run IDs. Data,
model snapshots, logs and checkpoints remain inside TRE.
