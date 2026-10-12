# Daily alcohol revision

```text
DAILY REVISED 1/3 | DATA / SOURCE AUDIT | SYNTHETIC
VERSION=2026-10-11.alcohol.1 | STATUS=COMPLETE_WITH_LIMITATIONS
Reuses verified saved daily panel; no CGM rebuild. Person FE; cluster CR1.
PRIMARY: observed alcohol or classified no-drink days; estimates excluded.
QC days=960; unique AMED match=960; ambiguous=0; unmatched=0
QC source candidates: observed=814 recovered=960
Excluded source flags (overlap): fallback-zero=0 invalid=0 missing-label=0
SYNTHETIC validation only; no real screenshot count reproduction claimed.
PRIMARY Diet Taxon People Days MedianDays
NOVA4 T068 80 814 10
NOVA4 T096 80 814 10
Fiber T068 80 814 10
Fiber T096 80 814 10
P/P0 and R/R0: 24/24 person-day sample identity checks passed.
Classified no-drink is based on food logs, not proof of no alcohol intake.
```

```text
DAILY REVISED 2/3 | ALL 12 PRIMARY TESTS | SYNTHETIC
Beta: NOVA4 +10pp or Fiber +5g/1000kcal, times original microbe CLR-Z +1.
SD/Mean mg/dL; CV percentage points; pointwise 95% CI.
Diet Taxon Y People Days Beta [L95,U95] qSD4 qAll12
NOVA4 T068 SD 80 814 -0.3066 [-0.3293,-0.2839] 6.171e-41 1.851e-40
NOVA4 T068 Mean 80 814 -0.003179 [-0.03026,0.0239] NA 0.9786
NOVA4 T068 CV 80 814 -0.2856 [-0.3081,-0.2632] NA 6.666e-39
NOVA4 T096 SD 80 814 -0.02159 [-0.1188,0.07557] 0.6595 0.8793 *
NOVA4 T096 Mean 80 814 -0.0004725 [-0.03544,0.0345] NA 0.9786 *
NOVA4 T096 CV 80 814 -0.02252 [-0.1128,0.06774] NA 0.8793 *
Fiber T068 SD 80 814 -0.08295 [-0.2083,0.04244] 0.3834 0.4601
Fiber T068 Mean 80 814 -0.004935 [-0.09042,0.08055] NA 0.9786
Fiber T068 CV 80 814 -0.07972 [-0.192,0.03258] NA 0.4601
Fiber T096 SD 80 814 0.04477 [-0.1061,0.1956] 0.6595 0.8793 *
Fiber T096 Mean 80 814 -0.09018 [-0.2004,0.02007] NA 0.4299 *
Fiber T096 CV 80 814 0.06017 [-0.07341,0.1938] NA 0.7453 *
qSD4=four SD tests; qAll12=all 12; failed tests remain in denominators.
* limited microbial tails/IQR or >10% focal information from one person.
```

```text
DAILY REVISED 3/3 | SENSITIVITY | SYNTHETIC
P=primary S=strict Q=coverage95 D=days7 N=nutrients B=baseline W=equal-weight
R=recovered alcohol; P0/R0=same sample without alcohol controls; A0=all eligible days, no alcohol controls
Spec Fit/12 qSD4<.05 qAll12<.05 SameSign/12 Flags
P 12/12 1 2 12/12 6
S 12/12 1 2 11/12 12
Q 12/12 1 2 12/12 6
D 12/12 1 2 12/12 6
N 12/12 1 2 10/12 6
B 12/12 1 2 11/12 6
W 12/12 1 2 12/12 3
R 12/12 1 2 11/12 6
P0 12/12 1 2 11/12 6
R0 12/12 1 2 11/12 6
A0 12/12 1 2 11/12 6
Each specification uses its full 4/12 family. SameSign is not significance.
Primary SD diet slopes at microbial Q25 / Q75 (point estimates):
NOVA4 T068: 0.772 / 0.2659
NOVA4 T096: 0.5653 / 0.538
Fiber T068: 0.182 / 0.045
Fiber T096: 0.09035 / 0.1471
Stool record AFTER day, known-date fraction: 1..1
Same-cohort observation; overlapping samples are not replication or mediation proof.
Negative interaction means a slope difference; not evidence of clinical protection.
Full estimates/CI/slopes/flows in tables/; participant-day rows stay in private/.
```
