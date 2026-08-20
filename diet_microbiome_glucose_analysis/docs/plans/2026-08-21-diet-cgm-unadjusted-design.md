# Diet–CGM 对齐与无调整关联设计

## 目标与边界

本阶段只完成两件事：建立参与者级饮食–CGM分析表，以及运行不含协变量的饮食–CGM
线性回归。肠道微生物暂不进入模型，也不限制饮食–CGM样本；仅为之后的三轴分析保留
接口。本阶段不修改 `diet_deal` 或 `cgm_deal` 的任何结果文件。

## 输入

- AMED：`amed_energy_adjusted_score_z`
- hPDI：`hpdi_score_energy_adjusted_z`
- CGM Primary：`cgm_mean_z`、`cgm_cv_z`、`cgm_above_140_z`

三张表均限定 `cohort=10k`、`research_stage=00_00_visit`，使用
`participant_id + cohort + research_stage` 一对一连接。AMED和hPDI先外连接，随后与CGM
内连接，使每个饮食评分按自身可用样本建模，而不是强制两种评分同时完整。若hPDI旧CSV
尚未含最新Z字段，脚本停止并提示重跑 `05_calculate_hpdi_scores.py`。

## 输出与模型

`00_diet_cgm_aligned.csv` 每名参与者一行，包含两个饮食Z分数和CGM raw/clean/z字段。
交集报告分别记录AMED、hPDI、CGM及各组合人数；可选gut CSV只增加样本存在性统计，
不改变分析表。

第二步运行6个模型：两个饮食评分分别对应三个Primary CGM Z-score。模型固定为带截距的
一元普通最小二乘回归，不加入年龄、性别、BMI、能量等协变量。输出样本量、截距、beta、
标准误、95%CI、t统计量、双侧P值和R²，并对6个P值统一执行Benjamini–Hochberg FDR。
每个结局独立完整案例分析，不因某一CGM表型缺失而删除参与者的其他模型记录。

## 安全与验证

缺字段、重复参与者键、非有限分数、暴露无变异或样本少于3人时立即停止。单元测试覆盖
外连接/内连接逻辑、旧hPDI字段报错、已知OLS系数、缺失结局的outcome-specific N和FDR。
脚本兼容Python 3.7，不使用Python 3.9+内置泛型，也不依赖statsmodels。
