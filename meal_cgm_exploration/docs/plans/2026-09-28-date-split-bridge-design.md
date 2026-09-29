# Date-separated meal-to-CGM bridge

User authorized implementation and an upload folder. Preserve all frozen previous packages and output directories. New folder: date_split_bridge, Python>=3.8, existing numpy/pandas/scipy/matplotlib/pytz only. No agents or new runtime downloads.

## Fixed design before real outcomes

Read completed response_followup 2026.09.27.2 plus its frozen nutrition/quality/cache lineage; verify hashes, exact bridge and first-window memberships, cuts and feasibility counts. Keep the calendar-midpoint and actual meal interpolation-support cutoff unchanged.

Use an explicitly NEW window metric recipe, not a claimed reproduction of the unknown upstream iglu export. Local preprocessing only selects/cleans exported iglu values. Consulted official iglu MODD, SDw and grid documentation; freeze the following operational rules in a versioned contract.

Within each complete, unique-connection local date already having >=90% raw 15-minute coverage, form an actual-time 15-minute grid with bounded linear interpolation <=45 minutes. No borrowing readings across a date/cut, extrapolation, or interpolation across explicit invalid observations. Require >=90% finite grid coverage; use >=3 qualifying dates for Mean/CV/SDw. Mean and sample-SD CV use pooled grid values, SDw is the unweighted mean of daily sample SDs. Actual DST-day length is retained for these three. MODD15 is pooled absolute same-clock differences on adjacent non-DST qualifying dates, >=48 finite matches per pair, >=2 pairs. Excluding DST for MODD avoids ambiguous duplicate/nonexistent clocks. This is a named operational implementation; do not equate it to legacy MODD. Report extra exclusions beyond the previous proxy.

Use early-meal personal means of net/peak/positive responses. Primary: late-window CGM metrics. Comparator: full eligible-date CGM recomputed with the SAME recipe, same persons and same early X; full comparator includes later dates and may share early readings. Direct comparison: regress late-minus-full Y on the identical design, so covariance is respected. This is an internal association, not prospective prediction/external validation.

24 primary specs = 3 X * 4 Y * 2 settings. 24 full-record comparators and 24 paired-difference contrasts are explicitly secondary. BH over all24 primary specs including failed slots; separate fixed24 full and difference families. Adjust original BASE, one row/person, HC3 + t intervals via frozen fit. Reference X mean/SD and full-period Y mean/SD fixed from source_primary paired cohort for each X/Y and shared by joint + all3 targets. Also save raw-unit slopes/scales. No outcome clipping/logging or reuse of old full-period Z as a late-window outcome. Old full-period Z used only for concordance diagnostics.

Save all rows incl non-estimated reasons, membership hashes, daily/person audit privately, source-to-new metric concordance and selection summaries. Only three returned summaries: run/quality; original12 rows; joint12 rows, with primary beta/CI/q, full beta and paired-difference CI/q. Never select rows by q.

## Verification

Analytic metric examples; interpolation gap/invalid/read-boundary tests; DST and true adjacent-date MODD; early perturbation cannot affect late metrics; frozen membership/count mismatches and tampering rejected; scalar scales shared across settings; paired-difference sandwich SE; BH includes missing hypotheses; full synthetic end-to-end and short readback; Python3.8 syntax/shell; frozen dependency/archive checks. Real data unavailable locally; do not claim real model completion.
