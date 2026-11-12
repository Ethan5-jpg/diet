# Detectability of injected meal carbohydrate modification

This conditional simulation follows the original pooled meal design,
repeated FoodID templates, and frozen additional-household transfer.
The real exposures, microbiome profiles, meal keys and household
structure are fixed. The simulation preserves observed residual scales
and within-household patterns by multiplying each household's residual
vector by one independent random sign. Fit a nuisance mean from the
discovery data and use household-held-out discovery residuals from the
first archived partition. The additional households use residuals from
the template baseline frozen on discovery.
[Simulation design and reporting](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/)

Inject 0, 0.25, 0.5, 1, 2, or 4 mg/dL per 10 g carbohydrate and one
standard deviation of the known microbial direction. Use the frozen
gut PC1 and a fixed alternating-sign combination of all ten archived
oral and gut axes as two generating directions. Direction scaling uses
the participants in the corresponding discovery design. Fit the
ten-axis interaction at penalties 0.03 and 0.3, and a one-dimensional
oracle at 0.03. The oracle receives the known generating direction and
illustrates a correctly specified marker. Retain the original two-hour
mean net glucose and secondary peak definitions.

For each route and endpoint report median held-out delta R², the
fraction of 128 replicates with a positive paired 95% interval, and
Monte Carlo uncertainty. Use 399 common household draws across
amplitudes and methods. Report the first tested amplitude with at
least 80% detection at that and every larger tested dose, when present.
All models refit on the relevant simulated training outcomes. The
additional households use only discovery observations for model
fitting in each replicate. These results describe sensitivity to the
specified simulation and conditional evaluation target.

Run from the isolated TRE experiment directory using the pinned study
interpreter. All synthetic outcomes and detailed results stay in TRE.

```sh
python -B -m unittest discover -s .
python -B run_simulation.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --smoke
python -B run_simulation.py --package STUDY_HANDOFF --commit COMMIT --run smoke-1 --verify
python -B run_simulation.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1
python -B run_simulation.py --package STUDY_HANDOFF --commit COMMIT --run discovery-1 --verify
```
