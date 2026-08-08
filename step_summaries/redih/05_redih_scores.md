# 第5步：17组件 rEDIH 评分

## 状态

脚本已在本地完成并通过合成测试；服务器第3和第4步完成前不能生成全量评分。

脚本：`Data/diet_deal/scripts/redih/05_calculate_redih_scores.py`

版本：`2026-08-08-redih-scores-v2`

## 输入与计分资格

- rEDIH 摄入表：`outputs/04_score_intakes/redih/redih_participant_component_intakes.csv`
- 酒精完整性来源：`outputs/04_score_intakes/amed/amed_participant_component_intakes.csv`

第二个文件仅提供本项目已经审核过的参与者级 `alcohol_complete` 标记及酒精审计字段；rEDIH 不复用 AMED 的食物映射或酒精分值。两个文件按 `participant_id + cohort + research_stage` 一对一合并，任何未匹配或重复索引都会报错。

全部基线参与者行保留。仅 `alcohol_complete=true` 者计算正式评分；`false` 者的 EDIH、rEDIH、温莎化、能量校正和五分位字段均留空。当前 AMED 结果预期约7,192人酒精完整、2,545人不完整，最终仍以服务器本次合并 QC 为准。

## 评分定义

17个 HPP 可用组件全部进入当前评分：

| 组件 | 权重 |
|---|---:|
| red meat | 0.250 |
| low-energy beverages | 0.053 |
| cream soups | 0.787 |
| processed meat | 0.199 |
| poultry | 0.183 |
| butter | 0.094 |
| French fries | 0.581 |
| other fish | 0.172 |
| high-energy drinks | 0.104 |
| tomatoes | 0.095 |
| low-fat dairy | 0.025 |
| eggs | 0.124 |
| wine | -0.165 |
| coffee | -0.035 |
| whole fruits | -0.029 |
| high-fat dairy | -0.046 |
| green leafy vegetables | -0.055 |

计算公式：

```text
EDIH_17 = Σ(mean_daily_component_g × published_weight)
rEDIH_17 = -EDIH_17
```

margarine 是标准18组中唯一因 HPP 未测量而排除的组件。wine 已纳入评分；合成回归测试会检查增加 wine g/day 时，rEDIH 按 `wine_g/day × 0.165` 增加。

## 后处理

以下参数都只在当前 `alcohol_complete=true` 样本中估计：

1. 计算原始 rEDIH 的0.5和99.5百分位并温莎化。
2. 以平均每日总能量为自变量、温莎化 rEDIH 为因变量拟合一元 OLS。
3. 能量校正分等于残差加回温莎化分均值。
4. 按分数和参与者稳定键生成尽量均衡的1至5五分位。
5. 最终分析样本确定后再计算Z-score；本步骤不预先标准化。

以后处理并补齐酒精缺失后，不能只填补原评分列：必须重新运行本步骤，在更新后的合格样本内重算温莎界值、能量回归参数和五分位。

## 输出

- `redih_participant_scores_17.csv`
- `redih_component_coefficients.csv`
- `redih_scoring_parameters_17.csv`
- `redih_score_distribution_17.csv`
- `redih_score_qc_17.csv`

参与者评分表保留9,737行的预期基线框架；实际有分人数由 `alcohol_complete` 决定。主要字段为 `edih_score_raw_17`、`redih_score_raw_17`、`redih_score_winsorized_17`、`redih_score_energy_adjusted_17`、`redih_energy_adjusted_quintile_17` 和 `redih_score_complete`。

## 服务器运行

```bash
grep 'SCRIPT_VERSION' scripts/redih/05_calculate_redih_scores.py
python3 scripts/redih/05_calculate_redih_scores.py 2>&1 | tee /tmp/redih_05_v2.log
```

需要检查总参与者数、酒精完整和不完整人数、17项贡献完整性、wine 非零人数、EDIH/rEDIH 严格反向、温莎界值、能量回归参数、四个分数分布及五分位人数。分布表只统计有完整评分者，不能用不完整者的0或部分和替代缺失。
