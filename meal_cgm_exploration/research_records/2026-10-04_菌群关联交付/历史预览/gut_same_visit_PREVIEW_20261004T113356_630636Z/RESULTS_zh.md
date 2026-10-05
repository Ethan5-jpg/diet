# 同访视四候选菌关联结果

DATA=REAL_SCREENSHOT_ROUNDED

截图精度预览；未重新估计模型。未显示值不补造，正式写作待服务器全精度导出。

本次按已确认口径：保留每餐CGM质控、每人至少5餐/3日及必要变量完整性；不新增全程缺失<5%门槛，不按菌群采样先后或日期未知排除。

主样本：3,631人、91,144餐；strict：3,398人、85,160餐。个人等权，餐不是独立的回归样本。

## 餐次关联

| 结局 | 菌 | 主样本 β [95%CI]; q | strict β [95%CI]; q |
|---|---|---|---|
| Mean postprandial increment | T023 | 0.0535 [-0.125, 0.232]; q=0.556 | 0.0464 [-0.14, 0.233]; q=0.625 |
| Mean postprandial increment | T046 | 0.202 [0.00389, 0.4]; q=0.0913 | 0.243 [0.038, 0.448]; q=0.0404 |
| Mean postprandial increment | T068 | 0.0846 [-0.103, 0.272]; q=0.501 | 0.0931 [-0.101, 0.287]; q=0.463 |
| Mean postprandial increment | T096 | 0.23 [0.033, 0.427]; q=0.0885 | 0.251 [0.0471, 0.454]; q=0.0404 |
| Pre-meal glucose | T023 | 0.455 [0.164, 0.746]; q=0.0087 | 0.403 [0.102, 0.704]; q=0.0233 |
| Pre-meal glucose | T046 | -0.389 [-0.688, -0.0903]; q=0.0285 | -0.444 [-0.753, -0.135]; q=0.0195 |
| Pre-meal glucose | T068 | 0.213 [-0.0897, 0.517]; q=0.223 | 0.215 [-0.1, 0.531]; q=0.241 |
| Pre-meal glucose | T096 | -0.696 [-0.996, -0.396]; q=4.42e-05 | -0.705 [-1.02, -0.394]; q=7.21e-05 |
| Peak glucose rise | T023 | 0.127 [-0.138, 0.392]; q=0.396 | 0.112 [-0.164, 0.389]; q=0.426 |
| Peak glucose rise | T046 | 0.253 [-0.0372, 0.543]; q=0.14 | 0.318 [0.0189, 0.616]; q=0.0744 |
| Peak glucose rise | T068 | 0.104 [-0.18, 0.388]; q=0.472 | 0.152 [-0.142, 0.445]; q=0.356 |
| Peak glucose rise | T096 | 0.264 [-0.0208, 0.548]; q=0.139 | 0.288 [-0.00423, 0.581]; q=0.0855 |

单位：mg/dL/原菌群CLR-Z。餐前水平不是空腹血糖；平均净增量为净iAUC/120，可为负；峰值升幅取实际观测的峰值减餐前基线。

## 同连接整体CGM参照

| 结局 | 菌 | 主样本 β [95%CI]; q | strict β [95%CI]; q |
|---|---|---|---|
| Whole-record CGM mean | T023 | 0.0482 [NA, NA]; q=0.0005 | 未在本地截图显示 |
| Whole-record CGM mean | T046 | -0.0412 [NA, NA]; q=0.00265 | 未在本地截图显示 |
| Whole-record CGM mean | T068 | 0.0463 [NA, NA]; q=0.00807 | 未在本地截图显示 |
| Whole-record CGM mean | T096 | -0.0646 [NA, NA]; q=4.85e-06 | 未在本地截图显示 |
| Whole-record CGM CV | T023 | 0.00527 [NA, NA]; q=0.734 | 未在本地截图显示 |
| Whole-record CGM CV | T046 | 0.0107 [NA, NA]; q=0.607 | 未在本地截图显示 |
| Whole-record CGM CV | T068 | 0.0667 [NA, NA]; q=0.000268 | 未在本地截图显示 |
| Whole-record CGM CV | T096 | 0.00946 [NA, NA]; q=0.627 | 未在本地截图显示 |
| Whole-record CGM SDw | T023 | 0.0401 [NA, NA]; q=0.0173 | 未在本地截图显示 |
| Whole-record CGM SDw | T046 | -0.0178 [NA, NA]; q=0.346 | 未在本地截图显示 |
| Whole-record CGM SDw | T068 | 0.0931 [NA, NA]; q=1.23e-07 | 未在本地截图显示 |
| Whole-record CGM SDw | T096 | -0.0268 [NA, NA]; q=0.148 | 未在本地截图显示 |

参照单位为原CGM Z/原CLR-Z，不能与餐次系数直接比较大小；监测时长/覆盖影响仍适用于整体CGM。整体和餐次可能共享读数，不是独立验证。

## 结果解读边界

主结局仍为平均餐后净增量，不因餐前更显著而改变主次。候选由同队列选择；FDR没有消除选择后推断问题。strict与主样本重叠。缺失只在不涉及某餐所需读数时不直接改变该餐计算，仍可能影响餐次代表性。饮食日志也可能漏记进食。菌样本身份/访视仍须正确匹配；同访视允许关联分析，不单独建立时间因果顺序。

## 英文Results草稿

In the primary sample, 3631 participants contributed 91144 eligible meals; the overlapping strict subset included 3398 participants and 85160 meals. The four candidates were selected in earlier analyses of the same cohort. Using the original multiple-testing families, associations with pre-meal glucose met q < 0.05 for T023, T046, T096. For the primary outcome, mean postprandial increment, candidates meeting the threshold were none; for peak rise, they were none. In the strict subset, candidates meeting the threshold for mean increment were T046, T096. These comparisons describe associations conditional on the specified covariates. Different significance patterns across outcomes or overlapping samples do not establish differences between effects, independent replication, or causal pathways.

所有模型、阴性结果和缺失字段均保留。p/q显示为0只表示数值下溢，不代表真实概率为0。
