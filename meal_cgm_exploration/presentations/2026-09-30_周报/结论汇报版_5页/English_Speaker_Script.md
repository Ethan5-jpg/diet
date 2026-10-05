# HPP Meal and Glucose Study English Presentation Script

30 September 2026. Five slides, approximately four minutes. The headings identify slide changes and do not need to be read aloud.

## Slide 1 This week’s work

Hi everyone. This week, we examined glucose responses to individual meals.

Our earlier work used overall CGM measures. This analysis adds more detail: the glucose level before eating, how much it rises afterward, and how the response differs between the first and second hours.

I will present three findings and explain what they contribute to our study.

## Slide 2 Premeal glucose improves prediction

The first finding is that glucose before a meal helps us predict the average glucose level during the next two hours.

We started with a model using food records, historical information, and meal context. We then added premeal glucose and its recent trend.

We trained on earlier dates and tested on later dates. Adding this information reduced the average absolute prediction error by about 0.74 milligrams per deciliter.

This number describes better prediction accuracy. It does not mean that anyone’s glucose fell by 0.74.

The useful conclusion is that premeal glucose adds information beyond the existing model. However, we tested only known participants and previously seen meal templates. Whether this works for new people or new foods remains untested.

## Slide 3 Dietary associations are small

The second finding concerns how much glucose rises after eating.

Here, we subtract premeal glucose from the average glucose during the following two hours. We call this the mean net increment.

When we compared different meals within the same person, a ten percentage point increase in the energy share from ultra-processed foods, measured by NOVA4, was associated with an increment about 0.17 milligrams per deciliter higher.

Separately, a one-unit increase in the original standardized rEDIH score was associated with an increment about 0.49 milligrams per deciliter lower after full adjustment.

Both results have statistical support, but the differences are small. The rEDIH result also depends on adjusting for meal nutrients, and other dietary scores showed no consistent pattern.

We can report these specific associations. We cannot yet conclude that changing a person’s diet will produce these glucose changes.

## Slide 4 AMED relates to response timing

The third finding concerns when the glucose response occurs.

AMED is a Mediterranean diet score. Its association with the overall two-hour average response did not have statistical support.

However, when we directly examined the second hour minus the first hour, we found a supported association.

For each one-unit increase in the original standardized AMED score, this difference was about 0.27 milligrams per deciliter lower.

In other words, higher AMED was associated with a lower response in the second hour relative to the first.

This suggests that examining separate periods can reveal information that the overall average misses. It does not establish faster glucose clearance or better insulin function.

## Slide 5 Conclusions

This week’s contribution is a more detailed description of glucose around individual meals.

Premeal information improved prediction in the tested setting. Selected dietary measures showed small associations with response size. AMED also showed an association with the difference between the first and second hours.

These are results we can report, while keeping their limitations clear.

The next step is to organize the existing results and complete the write-up. A dietary causal mechanism remains unproven.

Thank you.

## Source notes

The estimates are rounded from archived results. Confidence intervals are pointwise. Prediction improvement refers to participant-equal MAE with fixed fitted models. The NOVA4 mean increment equals net iAUC divided by 120 minutes. rEDIH was reverse-coded upstream.

- [早期真实结果_核对转录.json](/Users/blizzardxu/Desktop/HPP/Data/meal_cgm_exploration/research_records/2026-09-29_完整研究盘点与分析报告/早期真实结果_核对转录.json)
- [nutrition_bridge_累计已展示结果_截图精度.json](/Users/blizzardxu/Desktop/HPP/Data/meal_cgm_exploration/research_records/nutrition_bridge_累计已展示结果_截图精度.json)
- [quality_timecourse_累计133项已展示结果_截图精度.json](/Users/blizzardxu/Desktop/HPP/Data/meal_cgm_exploration/research_records/quality_timecourse_累计133项已展示结果_截图精度.json)
