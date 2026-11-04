# Execution and interpretation

Use the historical collaborator_handoff_20261102_v1 package and the pheno Python environment. Deploy these source files into a new sibling of diet_nested_stability_20261103_v4. Verify SHA256 against the reviewed GitHub commit, including the v1 attempt02 and v3/v4 dependency hashes used by the original v4 registration. Keep the per-file deployment receipt and complete commands in the study.

Run in this order from this directory:

    python test_features.py
    python audit_v4.py
    python run.py pc
    python interpret.py pc
    python run.py module
    python interpret.py module

run.py skips completed fold checkpoints within the same version. Restart with identical verified inputs and code, and use a fresh directory for a changed specification. Each repeat stores the actual model and reference MSE, foldwise direction and conditional slope interval. The PC replay checks A and fixed-penalty B against v1 and the evaluation target and inclusion masks against v4. Interpret the two affected splits on the same participants separately.

Only after deciding to designate the PC10 candidate, run:

    python freeze_pc.py
    python closeout.py VERIFIED_SOURCE_COMMIT

The freeze step refits the penalty by household CV on discovery, then rebuilds the selected score on discovery. It does not estimate a new validation metric. Frozen taxon identifiers, reference set, training pseudocounts, sum-zero log weights, intercept and score units stay in frozen_pc_TRE_ONLY/rule.json. In TRE, apply_score.score_new takes abundances in the original units, column IDs and that rule. It aligns reference taxa by ID, rejects missing columns or invalid values and returns NaN for unsupported all-zero samples. Every weight includes the CLR reference subtraction. The score is the centered gut contribution to the fixed fiber contrast relative to the training mean, not the whole model prediction.

Real-data stage notes, inputs, result tables and rules have no repository sync step.
