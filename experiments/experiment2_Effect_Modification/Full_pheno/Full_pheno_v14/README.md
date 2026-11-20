# Expanded dietary main effects and persistent losses

Run from this directory in AppStream with the pinned V12 interpreter. Complete data and real outputs remain in TRE. Confirmation outcomes remain sealed.

~~~bash
rt=$(cat ../../../experiments/diet_effect_modification_20261109_v12/.python_path)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$rt" -B run_main.py --run-id main-1 --commit <commit>
~~~

The V12 archive verifies the reference no-diet, diet-main, and fixed-interaction predictions. The original 18 fitted models remain locked. Archived V12 category and V13 full-diet scores quantify prior losses by outcome domain and family coverage. Registered food days give a separate first-versus-second-half agreement check for nutrient, fiber, and processing indicators.

Seven predefined expanded main effects share the same no-diet background. The comparison includes the original full component set, ridge penalties of 0.35 and 1.00, one leading component per diet class at penalties of 0.10 and 0.35, half the fitted diet contribution, and a contribution scaled by observed energy-log days divided by ten and capped at one. The leading-component model keeps equal diet-class variance scaling. No candidate selection uses the new held outcomes.

Each B model has three microbiome settings. The fixed and half-strength versions apply the same interaction design over the full diet representation and fit the correction to the indicated B residuals. The off model returns B. The seven diet classes, eligible outcome roster, household splits, and no-diet predictions are shared across the comparison.

Two fresh household partitions reassign the discovery cohort. The diary diagnostic scores the reference diet main in the three-to-nine-day and ten-plus-day strata with at least 20 held households per stratum and fold. The groups are observational. Family-coverage and diary comparisons describe where losses concentrate. Joint gains require at least +0.10 R² percentage points from the interaction, at least three positive folds per partition, and a positive total diet increment in both partitions. Results include the overlap with V13 stable candidates and paired comparisons at the same interaction strength.
