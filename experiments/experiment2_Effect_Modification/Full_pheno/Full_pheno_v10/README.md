# Fixed diet main effects and a separate expanded panel

Run both analyses from this directory in AppStream using the project's pinned V12 interpreter. Keep individual records and outputs in TRE. Both analyses use discovery households and never open confirmation outcomes.

```bash
rt=$(cat ../../../experiments/diet_effect_modification_20261109_v12/.python_path)
"$rt" -B run_locked.py --run-id locked-1 --commit <commit>
"$rt" -B run_expanded_grid.py --run-id grid-1 --commit <commit>
```

The original 18 pairs use the registered V1 diet-off-interaction predictions and V4 no-diet predictions in every candidate. The run verifies the original partition against the archived 18-pair half-strength PC5 reference. Candidate variants adjust only the correction to the fixed predictions, with the original fiber–CGM glucose interaction pinned. The nested options require gains in at least three inner held household folds. Both new partitions refit all original training-fold transforms using the original recipes. The output records a digest of the baseline predictions and paired household simultaneous intervals. The original 17-pair coverage gate remains separate from the retained anchor.

The expanded grid uses eligible candidate diet features from the audited food inventory plus food-group energy diversity. Clinical outcomes use exact discovery-visit keys or the latest eligible blood draw recorded up to 365 days before the first qualified food day. CGM variability requires three qualified days on the selected connection. Each new pair has its own complete-case no-diet and diet models, with shared background terms and a fixed half-strength PC5 interaction. At least 120 training and 40 held households and nondegenerate training exposure are needed in every fold. The two original exposures are also scored against new endpoints under this exploratory recipe. No original pair enters this grid.

The three partitions reassign the same discovery households. The grid reports recurring positive performance when the pooled interaction is positive and at least three of the five folds have gains in all three partitions. The stronger screen also requires +0.10 R² percentage points in every partition, alongside a separate diet-total check. Household-cluster tests compare paired held-out squared errors with Holm adjustment across each full grid. Repeated partitions are a stability check within discovery.
