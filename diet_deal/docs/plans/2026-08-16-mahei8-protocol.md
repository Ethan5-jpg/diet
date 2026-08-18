# HPP mAHEI-8 Protocol Implementation Plan

> **Superseded 2026-08-16:** 用户随后决定同时排除钠。当前有效协议见
> `2026-08-16-mahei7-protocol.md`；本文件仅保留为 mAHEI-8 决策历史。

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Exclude trans fat and PUFA from the HPP score because the current event data do not provide reliable values, and document the resulting modified eight-component score.

**Architecture:** Keep all ten candidate rows in the input audit so the omission remains visible, but set trans fat and PUFA to `excluded_by_user_protocol` and `included_in_final_score=False`. Define the resulting score as `modified_AHEI_2010_8_component`, retain the unscaled raw range 0–80, and prohibit presenting a rescaled percentage as a formal AHEI score.

**Tech Stack:** Python 3.7-compatible syntax, pandas, unittest.

---

### Task 1: Freeze the mAHEI-8 contract in tests

**Files:**
- Modify: `tests/ahei/test_audit_ahei_inputs.py`

1. Assert that trans fat and PUFA are excluded rather than scored as zero.
2. Assert that exactly eight components remain in the final-score denominator.
3. Assert that the raw score range is 0–80 and the protocol name is mAHEI-8.
4. Assert that lookup scanning no longer treats trans fat or PUFA as blocking inputs.
5. Run the tests and confirm they fail before implementation.

### Task 2: Update the input audit

**Files:**
- Modify: `scripts/ahei/00_audit_ahei_inputs.py`

1. Add mAHEI-8 protocol constants and QC rows.
2. Add `included_in_final_score` to component availability.
3. Set trans fat and PUFA to user-protocol exclusions regardless of candidate lookup headers.
4. Preserve the rows and reasons so the omissions are auditable.
5. Do not calculate a final score in the audit.

### Task 3: Update lookup interpretation

**Files:**
- Modify: `scripts/ahei/00b_scan_ahei_lookup_sources.py`

1. Mark trans fat and PUFA lookups as not required under mAHEI-8.
2. Keep any discovered columns informational only.
3. Continue treating whole-grain strict values as optional and serving conversion as pending where applicable.

### Task 4: Verify and hand off

1. Run all AHEI audit tests.
2. Compile both scripts and parse them with Python 3.7 grammar.
3. Confirm no Python 3.9 built-in generic annotations are present.
4. Provide the next server audit command; do not run or request the LabData nutrient-loader inspection.
