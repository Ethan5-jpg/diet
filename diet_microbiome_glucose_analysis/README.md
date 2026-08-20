# Diet–CGM baseline analysis

本目录完成当前分析的前两步：

1. 对齐baseline AMED、hPDI与CGM参与者；
2. 运行两个饮食评分与三个Primary CGM表型的6个无协变量OLS模型。

肠道微生物当前不限制饮食–CGM模型样本，可选输入只用于统计三者交集。

## 输入字段

```text
AMED exposure:  amed_energy_adjusted_score_z
hPDI exposure:  hpdi_score_energy_adjusted_z

CGM outcomes:   cgm_mean_z
                cgm_cv_z
                cgm_above_140_z
```

模型固定为：

```text
CGM Z-score = intercept + beta × diet-score Z-score
```

模型不加入年龄、性别、BMI、能量摄入或其他协变量。6个P值统一计算
Benjamini–Hochberg FDR。

## 运行前更新hPDI

如果服务器上的hPDI CSV还没有 `hpdi_score_energy_adjusted_z`，先在HPP项目目录运行：

```bash
python3 Data/diet_deal/scripts/hpdi/05_calculate_hpdi_scores.py
```

最新文件应生成在：

```text
Data/diet_deal/outputs/05_diet_scores/hpdi/hpdi_participant_scores.csv
```

## 运行分析

```bash
cd Data/diet_microbiome_glucose_analysis
bash run_diet_cgm_analysis.sh
```

如需同时报告肠道微生物样本交集，但仍不限制模型样本：

```bash
GUT_CSV=/home/ec2-user/Desktop/HPP/Data/Transfer/gut_microbiome/gut_microbiome.csv \
bash run_diet_cgm_analysis.sh
```

## 输出

```text
outputs/data/00_diet_cgm_aligned.csv
outputs/models/01_unadjusted_diet_cgm_models.csv
outputs/reports/00_id_membership.csv
outputs/reports/00_overlap_summary.csv
outputs/reports/01_unadjusted_model_summary.csv
outputs/logs/diet_cgm_analysis_*.log
```

`00_diet_cgm_aligned.csv` 每人一行。AMED和hPDI先取并集，再与CGM取交集，因此某人
缺少一种饮食评分不会影响另一种评分的模型。每个CGM结局也独立采用完整案例，不要求
三个CGM表型同时非缺失。

`01_unadjusted_diet_cgm_models.csv` 包含每个模型的N、beta、标准误、95%CI、t值、
双侧P值、R²和BH-FDR。这是未调整的探索性关联，不能直接解释为独立效应或因果效应。
