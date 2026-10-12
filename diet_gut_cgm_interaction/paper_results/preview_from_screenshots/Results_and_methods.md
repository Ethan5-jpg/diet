# 饮食与菌群交互的论文结果草稿

来源状态：SCREENSHOT_PREVIEW。本地截图精度预览；用于讨论和排版，正式引用请使用服务器导出替换。

## 中文 Results

在参与者层面的整体CGM分析中，预先固定的24项重点交互中有2项通过重点家族的BH校正。完整84项主交互共有21项通过整体校正，其中19项来自60项扩展检验。上述检验涉及在同一队列既往分析中筛选的四个菌群特征，结果解释为探索性效应修饰关联。

NOVA4×E. rectale 对CV的交互系数为-0.1978（逐项95%CI -0.3068至-0.08871；N=3691；qFocus24=0.004581；qAll84=0.00401），单位为每1原饮食Z×1菌CLR-Z对应的percentage points。
NOVA4×E. rectale 对SDw的交互系数为-0.2087（逐项95%CI -0.3202至-0.09712；N=3693；qFocus24=0.004581；qAll84=0.00298），单位为每1原饮食Z×1菌CLR-Z对应的mg/dL。

在查看主结果之后，对全部60项扩展组合补充六组敏感性分析；每个设定均连同24项重点组合在完整84项内校正。原显著19项中，19项在全部六组中可估计且方向一致，7项进一步在六组中均满足qAll84<0.05。通过设定的数量仅作为描述性稳定性指标，不能视为相互独立的复制。

六组均保留的组合及其原主分析效应见Table 4；原全部显著扩展见Table 2，包含未能在全部设定保留者。条件斜率展示的是同一连续模型在菌群参考分位点处的估计，并非将人群分组重新拟合。统计显著性不代表效应具有临床重要性，也不证明菌群造成饮食效应改变。

## English Results

In participant-level analyses of overall CGM phenotypes, 2 of 24 designated focus interactions met the focus-family Benjamini–Hochberg threshold (qFocus24 < 0.05). Across the full set of 84 primary interactions, 21 met qAll84 < 0.05, including 19 of the 60 extension tests. The four microbial features had been selected in earlier analyses of the same cohort; these analyses therefore assess exploratory effect modification rather than independent validation.

The NOVA4-by-E. rectale interaction for CV was -0.1978 (pointwise 95% CI -0.3068, -0.08871; N=3691; qFocus24=0.004581; qAll84=0.00401), expressed in percentage points per original dietary-score Z unit by microbial CLR-Z unit.
The NOVA4-by-E. rectale interaction for SDw was -0.2087 (pointwise 95% CI -0.3202, -0.09712; N=3693; qFocus24=0.004581; qAll84=0.00298), expressed in mg/dL per original dietary-score Z unit by microbial CLR-Z unit.

Following inspection of the primary results, six sensitivity specifications were applied to all extension combinations, with BH adjustment across 84 tests within each specification. Of the 19 originally significant extension interactions, 19 were estimable and retained the same sign in all six specifications, and 7 additionally met qAll84 < 0.05 in every specification. This consistency across overlapping samples provides within-cohort sensitivity evidence and does not establish replication or causal effects.

The extension combinations meeting all six sensitivity criteria were AMED × R. lactatiformans (CV, SDw); rEDIH × E. rectale (CV, Mean, SDw); rEDIH × R. lactatiformans (Mean, SDw).

For CV, the conditional NOVA4 slope was 0.3104 (95% CI 0.1941, 0.4267) at E. rectale Q25 and 0.1228 (95% CI -0.006447, 0.252) at Q75, in percentage points per original NOVA4 Z unit.

For SDw, the conditional NOVA4 slope was 0.4246 (95% CI 0.3082, 0.541) at E. rectale Q25 and 0.2267 (95% CI 0.09707, 0.3563) at Q75, in mg/dL per original NOVA4 Z unit.

## 可用于 Methods 的英文段落

We fitted participant-level ordinary least-squares models for mean CGM glucose, glucose coefficient of variation (CV), and the mean of daily within-day glucose standard deviations (SDw), including the dietary score, microbial feature, their product term, and the prespecified covariates. Dietary scores and microbial centred-log-ratio features retained their original standardised units. Diet and microbial features were centred using their corresponding primary-model sample means; these reference values were retained in sensitivity analyses. HC3 heteroskedasticity-robust covariance estimates and residual-degrees-of-freedom t inference were used. Coefficients and pointwise 95% confidence intervals were expressed on the original clean CGM scale (mg/dL for mean glucose and SDw; percentage points for CV). Models used complete cases for each combination. Conditional dietary slopes were evaluated at microbial reference quantiles using the full coefficient covariance; these were not subgroup regressions.

Covariates comprised age, sex, education, smoking, sleep duration, physical activity, vitamin use, hormone use, BMI and CGM device. hPDI models additionally adjusted for alcohol consumption, whereas EAT13, NOVA4 and carbohydrate energy-percentage models additionally adjusted for mean energy intake. Primary analyses excluded participants with an affirmative combined diabetes/A10 medication flag; the strict specification required an explicitly negative flag. Further sensitivity specifications used a common medication-information sample with and without antibiotic/PPI adjustment, added dietary and microbial quadratic terms, or added interactions of diet and microbiota with age, sex and BMI. The original focus common-sample analysis required completeness for two dietary scores; the extension follow-up required all seven scores, all four taxa, all three outcomes and their required covariates.

Benjamini–Hochberg adjustment was performed separately for the designated 24-test focus family and the full 84-test primary family. Original focus sensitivities used their own 24-test families. The post-primary extension follow-up used a fixed 84-test family within each of seven specifications (primary plus six sensitivities). Unavailable tests remained in each denominator with p=1, while their own q values were left missing. Extension retention required an estimable interaction, the same sign as its original primary estimate and qAll84<0.05 in each sensitivity. No multiplicity correction over only the selected extension hits was performed.

## 作者需保留的解释边界

- SDw按项目已核对定义表述为各日内葡萄糖SD的平均值，不称日间SD。历史计算版本和参数的溯源边界仍沿用原项目。定义参考[IGLU官方SD说明](https://irinagain.github.io/iglu/reference/sd_measures.html)。
- rEDIH 沿用项目原变量，不在本图表包内翻转或重算。其原始计量单位及评分定义溯源仍须结合上游文档确认，正文暂保留变量名，不据名称自动添加健康方向解释。
- 交互系数描述饮食斜率随菌群CLR-Z变化；CLR-Z不是绝对菌量。不能由交互正负直接给菌贴有益或有害标签。
- 四菌同队列预选、单次菌群测量、测量时间关系和未测量混杂限制继续保留。相同队列的敏感性结果不是独立验证。
- 六组全部保留的筛选发生于看到结果之后；正文展示选择必须披露，同时保留全部阴性结果。不同结局及饮食之间相关，不能将条目数作为独立发现数。
- 如需报告临床量级，应同时给出交互原单位、条件斜率和双IQR对比；不将较小q值等同于较大或更重要的效应。
