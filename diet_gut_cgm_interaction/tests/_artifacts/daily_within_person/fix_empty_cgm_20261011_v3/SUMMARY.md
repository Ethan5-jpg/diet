# Daily diet–gut–CGM

```text
DAILY 1/3 | DATA / FEASIBILITY | SYNTHETIC
STATUS=COMPLETE_WITH_LIMITATIONS
Person FE; participant-cluster CR1; same-visit observational associations.
CGM: observed 15-min bins; local calendar days; no gap interpolation.
Selected connections=80; covariate missing=0; exclusion unknown=7; both taxa=80
Diet days=960; CGM days=960; matched=960
Timing-invalid people: diet=0; CGM=0
Exposure Taxon People Days MedianDays Ready | QCdays MissDiet MissAlc MissGut
NOVA4    T068      80     960         12 True | 960 0 0 0
NOVA4    T096      80     960         12 True | 960 0 0 0
Fiber    T068      80     960         12 True | 960 0 0 0
Fiber    T096      80     960         12 True | 960 0 0 0
Missing counts overlap within QCdays, before complete-case / minimum-day filters.
Diet-day QC is a recording proxy; unlogged meals cannot be detected fully.
NOVA4: classified-energy share; Fiber: g/1000 kcal. CLR-Z is relative composition.
```

```text
DAILY 2/3 | ALL 12 PRIMARY-SPEC INTERACTIONS | SYNTHETIC
Beta per NOVA4 +10pp or Fiber +5g/1000kcal, times original microbe CLR-Z +1.
SD/Mean: mg/dL; CV: percentage points. Pointwise 95% CI.
Diet Taxon Y People Days Beta [L95,U95] qSD4 qAll12
NOVA4 T068 SD 80 960 -0.3045 [-0.3219,-0.2871] 3.85e-49 1.155e-48
NOVA4 T068 Mean 80 960 -0.003608 [-0.02774,0.02052] NA 0.8365
NOVA4 T068 CV 80 960 -0.2842 [-0.3019,-0.2665] NA 3.66e-46
NOVA4 T096 SD 80 960 -0.01703 [-0.1176,0.08356] 0.7371 0.8365 *
NOVA4 T096 Mean 80 960 -0.009132 [-0.04088,0.02262] NA 0.8365 *
NOVA4 T096 CV 80 960 -0.01529 [-0.1091,0.07848] NA 0.8365 *
Fiber T068 SD 80 960 -0.06518 [-0.1831,0.0527] 0.4891 0.5488
Fiber T068 Mean 80 960 0.002468 [-0.07394,0.07888] NA 0.9489
Fiber T068 CV 80 960 -0.06298 [-0.1674,0.04147] NA 0.5488
Fiber T096 SD 80 960 0.07677 [-0.09159,0.2451] 0.4891 0.6289 *
Fiber T096 Mean 80 960 -0.06609 [-0.1816,0.0494] NA 0.5488 *
Fiber T096 CV 80 960 0.08374 [-0.05866,0.2261] NA 0.5488 *
qSD4 = frozen four SD tests; qAll12 = all 12. Failed tests stay in denominators.
* = sparse microbial tails, zero microbial IQR, or >10% focal information from one person.
```

```text
DAILY 3/3 | SENSITIVITY / INTERPRETATION | SYNTHETIC
Specification Fit/12 SD q4<.05 All q12<.05 SupportFlags
primary              12/12        1          2          6
strict_known         12/12        1          2          6
coverage95           12/12        1          2          6
days7                12/12        1          2          6
nutrient_context     12/12        2          4          6
baseline_slopes      12/12        1          2          6
equal_person_weight  12/12        1          2          6
Primary SD diet slopes at microbial Q25 / Q75 (point estimates):
NOVA4 T068: 0.7632 / 0.2605
NOVA4 T096: 0.566 / 0.5444
Fiber T068: 0.1298 / 0.02215
Fiber T096: 0.04452 / 0.1418
Stool record AFTER day, known-date fraction across SD models: 1..1
Overlapping samples are not replication; negative interaction is a slope difference.
Single stool measurement is not a baseline predictor or proof of causal mediation.
Full estimates/slopes/flows: tables/. Participant/day records stay in private/.
```
