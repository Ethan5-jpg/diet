# HPP AHEI Input Audit v3 Implementation Plan

> **Superseded 2026-08-16:** 当前有效评分协议为 mAHEI-7，钠与 trans fat、
> PUFA 一并整项排除。`sodium_mg` 覆盖仅保留为信息性 QC。

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the AHEI input audit to reflect the approved HPP-specific whole-grain weight proxy, the verified 14 g standard drink, current complete AMED alcohol/energy inputs, and the existing missing-label event exclusion contract.

**Architecture:** Keep the audit read-only and do not calculate a final AHEI score. Record protocol constants and component-level readiness in the audit output. Reuse the existing AMED explicit whole-grain food mapping later in the intake builder; only mapped whole-grain events contribute their valid or approved imputed `weight_g`, while ambiguous/refined grains contribute nothing to this component.

**Tech Stack:** Python 3.7-compatible syntax, pandas, unittest.

---

### Task 1: Freeze protocol decisions in tests

**Files:**
- Modify: `tests/ahei/test_audit_ahei_inputs.py`

1. Add assertions for `STANDARD_DRINK_G == 14.0` and the HPP kcal-per-serving constants.
2. Assert that complete AMED alcohol plus complete sex is directly usable after the 14 g conversion.
3. Assert that whole grains are labelled as a user-approved weight proxy and not as strict dry whole-grain grams.
4. Assert that trans fat and PUFA remain unavailable when no real nutrient source exists.
5. Run the AHEI audit tests and confirm they fail before implementation.

### Task 2: Update the input audit

**Files:**
- Modify: `scripts/ahei/00_audit_ahei_inputs.py`

1. Increment the script version and document the approved proxy.
2. Add `STANDARD_DRINK_G = 14.0`, source notes, nondrinker scoring note, and drinks/day formula metadata.
3. Add HPP kcal-per-serving constants for vegetables, fruit, SSB plus fruit juice, and red plus processed meat.
4. Mark whole grains as `proxy_protocol_approved_mapping_pending`, with `whole_grain_proxy_g = mapped weight_g` and explicit exclusions for ambiguous/refined grains.
5. Preserve the current three-label deletion and participant/day retention checks.
6. Keep nuts plus legumes, sodium gaps, trans fat, and PUFA visibly unresolved where their required data are absent.

### Task 3: Update the lookup-only scan

**Files:**
- Modify: `scripts/ahei/00b_scan_ahei_lookup_sources.py`

1. Increment the script version.
2. Record whole grains as no longer blocked on an actual `whole_grain_g` lookup under the approved proxy protocol.
3. Continue scanning for a better whole-grain source as optional sensitivity information, while retaining serving, trans-fat, and PUFA source checks.

### Task 4: Verify locally

**Files:**
- Test: `tests/ahei/test_audit_ahei_inputs.py`

1. Run the complete AHEI audit unittest file.
2. Compile both scripts.
3. Parse both scripts with Python 3.7 grammar and reject Python 3.9+ built-in generic annotations.
4. Review output wording to ensure the proxy is never presented as strict AHEI dry whole-grain grams.

### Task 5: Server handoff

1. Provide copy-paste commands from `/home/ec2-user/Desktop/HPP/Data`.
2. Run the v3 input audit before any scoring script.
3. Request only terminal output/screenshots; do not download server data locally.
