# V8 second review

Status: 29 synthetic checks passed in 84.744 seconds on local macOS ARM64. No server data were read and no research job was started.

## Reproduced and corrected

| Area | Evidence before correction | Correction | Regression coverage |
|---|---|---|---|
| Fixed diet contrasts | A synthetic constant-sex training fold raised `'sex' is not in list` | Update only retained host diet columns; reconstruct numeric covariates using saved training means/scales when needed | Constant sex, dropped age main effect with retained diet-age term, all four explicit contrasts |
| Cached representation diagnostics | An injected PLS solver warning was recorded on first projection but absent from a later cached fit | Save factor warnings with representation state and merge them into downstream diagnostics | Two learners share the same warned projection; warning survives serialization |
| Ridge fusion diagnostics | The regression case has a raw coefficient below 1e-8 but a standardized contribution above 1e-8 | Use standardized contributions to identify affected members; save coefficient type explicitly | Scaled prediction member remains flagged in ridge fusion |

The full checks also cover all 138 configured cases, 55 pipelines, serialization, gradients/proximal optimization, household separation, unchanged choices after held-out label/feature changes, resume without candidate refits, corrupt checkpoints and final-summary recovery. The complete two-repeat workflow uses reduced synthetic fitting budgets; full default candidate smoke tests use the actual default settings on small synthetic profiles.

Python syntax, Ruff, shell syntax and dependency compatibility (`uv pip check`: 31 installed packages compatible) passed. Local UV environment does not include a pip module; no package installation was performed during the review. This is not an Amazon Linux validation. The server installer uses a standard native-filesystem venv with pip, as in the successful V7 setup.

Prior V6/V7 Python source hashes remain unchanged. `source_before/` retains the modified files' prior text, old validation record and old upload manifest. The current `../validation.json` and top-level `upload_manifest.json` describe the corrected package. Actual predictive performance is pending manual server execution.
