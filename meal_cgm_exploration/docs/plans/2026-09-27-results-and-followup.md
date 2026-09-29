# Results figures and server follow-up implementation plan

**Goal:** Deliver local figures/tables from archived screenshot-precision results and an independently uploadable, read-only server follow-up package.
**Authorization:** User approved execution on 2026-09-27; implement the saved research plan without another approval checkpoint.
**Architecture:** Frozen analysis packages and runs remain immutable. Local plotting reads two cumulative JSON files. New `response_followup/` reuses frozen loaders, verifies provenance and exact B-model membership, transforms existing interaction coefficients, and audits disjoint-date feasibility without fitting new effects.
**Tech stack:** Python >=3.8, numpy/pandas/scipy/matplotlib, SQLite read-only; Markdown/JSON and PNG/PDF/SVG artifacts.

1. Verify every research archive manifest entry. Inspect frozen loaders, model contracts and cache schema. Record existing limitations.
2. Implement reproducible local plotting: three main figures (adjustment and full-CGM matrices have companion panels), seven-diet evidence table, AMED supplement. Preserve original q families, all seven diets, source page/row metadata and screenshot precision.
3. Implement new server package: frozen run/hash checks; fixed 42 interaction specifications; exact participant/meal membership verification; equal-person descriptive score/carb/energy support; prespecified support-based relative contrasts with pointwise CI; no absolute response intercept or new p/q.
4. Implement provisional disjoint-date feasibility: same original connection, local dates/timezone, fixed midpoint split, exclude meal response windows crossing split, count qualified meal dates and second-window CGM coverage/consecutive-day pairs. Report feasibility and unknown definitions, not new bridge effects.
5. Add targeted synthetic tests for scientific invariants, membership/hash refusal, interval scaling, no overlap, read-only cache and paging. Validate Python3.8 syntax and shell runner. Produce upload manifest and concise Chinese README with screenshot commands.
6. Render and inspect local figures. Verify frozen packages unchanged. Preserve living archive files before updating index/manifest. Deliver links, run command, and exact requested result pages.

No data-dependent choice of significant models, no new date-split association models, and no four-bin AMED fits before reviewing feasibility. Synthetic tests are explicitly synthetic, never evidence.
