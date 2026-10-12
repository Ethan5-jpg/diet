# 三屏完整可见文本

来自用户截图；仅规范列间空格，保留所有可见数值和状态。

```text
DAILY 1/3 | DATA / FEASIBILITY | SERVER_INPUTS
STATUS=NO_ESTIMABLE_PRIMARY_MODELS
Person FE; participant-cluster CR1; same-visit observational associations.
CGM: observed 15-min bins; local calendar days; no gap interpolation.
Selected connections=7493; covariate missing=0; exclusion unknown=989; both taxa=6530
Diet days=88956; CGM days=94058; matched=79946
Timing-invalid people: diet=0; CGM=0
Exposure Taxon People Days MedianDays Ready | QCdays MissDiet MissAlc MissGut
NOVA4 T068 1 5 5 False | 38536 0 38210 4255
NOVA4 T096 1 5 5 False | 38536 0 38210 4255
Fiber T068 3 17 5 False | 48791 22210 48380 5403
Fiber T096 3 17 5 False | 48791 22210 48380 5403
Missing counts overlap within QCdays, before complete-case / minimum-day filters.
Diet-day QC is a recording proxy; unlogged meals cannot be detected fully.
NOVA4: classified-energy share; Fiber: g/1000 kcal. CLR-Z is relative composition.
```

```text
DAILY 2/3 | ALL 12 PRIMARY-SPEC INTERACTIONS | SERVER_INPUTS
Beta per NOVA4 +10pp or Fiber +5g/1000kcal, times original microbe CLR-Z +1.
SD/Mean: mg/dL; CV: percentage points. Pointwise 95% CI.
Diet Taxon Y People Days Beta [L95,U95] qSD4 qAll12
NOVA4 T068 SD N=1 D=5 NOT_ESTIMATED
NOVA4 T068 Mean N=1 D=5 NOT_ESTIMATED
NOVA4 T068 CV N=1 D=5 NOT_ESTIMATED
NOVA4 T096 SD N=1 D=5 NOT_ESTIMATED
NOVA4 T096 Mean N=1 D=5 NOT_ESTIMATED
NOVA4 T096 CV N=1 D=5 NOT_ESTIMATED
Fiber T068 SD N=3 D=17 NOT_ESTIMATED
Fiber T068 Mean N=3 D=17 NOT_ESTIMATED
Fiber T068 CV N=3 D=17 NOT_ESTIMATED
Fiber T096 SD N=3 D=17 NOT_ESTIMATED
Fiber T096 Mean N=3 D=17 NOT_ESTIMATED
Fiber T096 CV N=3 D=17 NOT_ESTIMATED
qSD4 = frozen four SD tests; qAll12 = all 12. Failed tests stay in denominators.
* = sparse microbial tails, zero microbial IQR, or >10% focal information from one person.
```

```text
DAILY 3/3 | SENSITIVITY / INTERPRETATION | SERVER_INPUTS
Specification Fit/12 SD q4<.05 All q12<.05 SupportFlags
primary 0/12 0 0 0
strict_known 0/12 0 0 0
coverage95 0/12 0 0 0
days7 0/12 0 0 0
nutrient_context 0/12 0 0 0
baseline_slopes 0/12 0 0 0
equal_person_weight 0/12 0 0 0
Not estimated (84): Too few people after daily quality/completeness/variation filters
Overlapping samples are not replication; negative interaction is a slope difference.
Single stool measurement is not a baseline predictor or proof of causal mediation.
Full estimates/slopes/flows: tables/. Participant/day records stay in private/.

RESULT_DIR=/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person/outputs/daily_20261011T055621_479640Z_666d40
(pheno) [ec2-user@ip-10-0-84-116 daily_within_person]$
```
