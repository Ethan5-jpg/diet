# HPP CGM 数据预处理说明

## 1. 处理目标

本流程用于整理 HPP（Human Phenotype Project）的连续血糖监测（CGM）数据，生成后续
饮食—肠道微生物—血糖关联分析可以直接使用的参与者级表型文件。

主流程不重新计算每15分钟一条的原始血糖时间序列，而是使用 HPP 已通过 `iglu`
计算好的 CGM 表型。处理内容主要包括：

1. 选择每名参与者最合适的一条 baseline CGM connection；
2. 执行 connection-level 数据质量控制；
3. 提取7个预先固定的 CGM 表型；
4. 按参考论文的方法处理表型异常值；
5. 生成 raw、clean 和 Z-score 三套数值。

表型异常值方法参考 Nature Communications 论文
`s41467-026-72748-3` 中的 metabolic phenotype cleaning 方法。

---

## 2. 输入数据

服务器输入文件位于：

```text
Data/Transfer/cgm/cgm.csv
Data/Transfer/cgm/iglu.csv
Data/Transfer/cgm/iglu_daily.csv
```

三个 CSV 均已恢复导出时丢失的身份字段：

```text
participant_id
cohort
research_stage
array_index
connection_id
```

### 三个数据表的作用

- `cgm.csv`：每个 CGM connection 一行，包含设备、采集时间和 QC loss 等信息；
- `iglu.csv`：每个 CGM connection 一行，包含 HPP/iglu 计算的46个血糖表型；
- `iglu_daily.csv`：每个 connection 每天一行，用于推导有效 CGM 天数。

需要特别注意：`iglu_daily.array_index` 是逐日数组中的行号，会随
`connection_day` 改变，因此不能作为 connection-level 分组键。daily 表按照以下字段
定义一条 connection：

```text
participant_id + cohort + research_stage + connection_id
```

而 `cgm.csv` 与 `iglu.csv` 的一对一核对仍使用五字段完整键。

---

## 3. Baseline connection 选择

只使用：

```text
cohort == "10k"
research_stage == "00_00_visit"
```

首先从 `iglu_daily.csv` 计算每条 connection 的 `cgm_days`，并验证：

- `connection_day` 从1开始且连续；
- 每天只有一行；
- collection date 数量与 connection day 数量一致；
- daily connection 与 `cgm.csv` connection 完全覆盖。

如果同一参与者存在多条 baseline connection，按照以下顺序选择一条：

1. `cgm_days` 最大；
2. `percentage_of_cgm_datapoints_lost_in_qc` 最小；
3. 如果存在 datapoint count，则 count 最大；
4. collection timestamp 最早。

当前 CSV 未提供 datapoint count。已观察到的5名重复参与者均可通过天数和 QC loss
确定最佳 connection，不需要任意选择。

实际数据流：

```text
全部 CGM connections                 9,497
baseline connections                 9,410
baseline participants                9,405
有多个 baseline connections 的参与者      5
选择后 participants                   9,405
```

所有重复候选及选择结果记录在：

```text
outputs/reports/01_duplicate_baseline_connections.csv
```

---

## 4. Connection-level QC

每名参与者选好一条 baseline connection 后，必须同时满足：

```text
cgm_days >= 10
QC loss fraction <= 0.30
cgm_mean、cgm_cv、cgm_above_140 均非缺失
```

血糖水平本身不是排除条件。例如，不能因为平均血糖高、CV高或目标范围内时间低而删除
参与者，否则会人为删除真正需要研究的代谢差异。

服务器实际结果：

```text
选择后 participants                         9,405
cgm_days >= 10                              7,499
QC loss <= 0.30                             9,049
三个 primary 表型完整                         9,405
同时通过全部 connection QC                   7,493
排除                                          1,912
```

最终通过 QC 的7,493人是后续 CGM 表型分析人群。

---

## 5. 固定的7个 CGM 表型

为了避免46个高度相关指标带来的重复检验和结果冗余，预先固定7个表型。

| 表型 | 含义 | 单位/方向 | 角色 |
|---|---|---|---|
| `cgm_mean` | 监测期间平均血糖 | mg/dL；越高通常越差 | Primary |
| `cgm_cv` | 血糖变异系数 | %；越高表示波动越大 | Primary |
| `cgm_above_140` | 血糖高于140 mg/dL的时间比例 | %；越高表示高血糖负担越大 | Primary |
| `cgm_gmi` | 根据平均血糖估算的葡萄糖管理指标 | %；越高通常越差 | Secondary |
| `cgm_in_range_63_140` | 血糖处于63–140 mg/dL的时间比例 | %；越高通常越好 | Secondary |
| `cgm_mage` | 平均血糖波动幅度，反映明显的日内波动 | mg/dL；越高波动越大 | Secondary |
| `cgm_modd` | 相邻两天同一时间点的平均差异 | mg/dL；越高表示日间稳定性越差 | Secondary |

基础合法性检查包括：

- `cgm_above_140` 和 `cgm_in_range_63_140` 必须位于0–100；
- CV、GMI、MAGE、MODD 不得为负；
- 非数值或无限值设为缺失并写入报告；
- 非法表型值只影响对应表型，不删除整名参与者。

本次7,493人的7个表型最初均完整，未发现基础合法性错误。

---

## 6. 表型异常值处理

对每个表型独立执行以下步骤：

1. 对全部非缺失值排序；
2. 找到能够覆盖 `ceil(0.95 × N)` 个观测的最窄区间；
3. 使用该区间边界内的观测计算均值 `μ` 和样本标准差 `σ`；
4. 距离 `μ` 超过 `8σ` 的值设为缺失；
5. 其余超出 `μ ± 5σ` 的值截尾到相应边界；
6. 原始值始终保留，不被覆盖。

实际清洗结果：

| 表型 | QC后人数 | clean非缺失 | 8SD设为缺失 | 缺失比例 |
|---|---:|---:|---:|---:|
| `cgm_mean` | 7,493 | 7,481 | 12 | 0.16% |
| `cgm_cv` | 7,493 | 7,488 | 5 | 0.07% |
| `cgm_above_140` | 7,493 | 7,355 | 138 | 1.84% |
| `cgm_gmi` | 7,493 | 7,481 | 12 | 0.16% |
| `cgm_in_range_63_140` | 7,493 | 7,382 | 111 | 1.48% |
| `cgm_mage` | 7,493 | 7,478 | 15 | 0.20% |
| `cgm_modd` | 7,493 | 7,470 | 23 | 0.31% |

合计316个“表型值”被设为缺失，另有418个表型值进行了5SD截尾。这里的316不是
316名参与者：一名参与者的某个表型被处理后，其他表型仍然可以继续分析，最终数据表
仍保留7,493名参与者。

`cgm_above_140` 和 `cgm_in_range_63_140` 的处理数量相对较多，主要因为百分比指标常有
边界集中和偏斜分布。后续分析保留 raw 版本作为敏感性分析，以检查异常值处理是否影响
结论。

每个被处理的 participant、表型、原值和处理动作记录在：

```text
outputs/reports/04_outlier_actions.csv
```

---

## 7. Raw、clean 和 Z-score

最终每个表型保留三套字段，例如：

```text
cgm_mean_raw
cgm_mean_clean
cgm_mean_z
```

- `raw`：通过基础合法性检查后的 iglu 原始值，用于追溯和敏感性分析；
- `clean`：经过8SD删除和5SD截尾后的值，仍保持原始单位；
- `z`：使用 clean 非缺失样本的均值和样本标准差计算的标准分数。

Z-score 的定义为：

```text
Z = (clean value - clean sample mean) / clean sample SD
```

Z-score 适合后续回归、微生物关联和中介分析，因为不同表型的单位不同，标准化后效应
可以统一解释为“结局改变多少个标准差”。需要临床单位解释时使用 clean 值。

---

## 8. 缺失值处理原则

被8SD规则设为缺失的值不进行均值、中位数或0填补，也不会删除整名参与者。

后续分析采用 outcome-specific complete case：

- 分析 `cgm_mean_z` 时，仅排除 `cgm_mean_z` 缺失者；
- 分析 `cgm_cv_z` 时，仅排除 `cgm_cv_z` 缺失者；
- 不要求7个 CGM 表型同时完整；
- 每个模型单独报告有效样本量。

主分析使用 `*_clean` 或 `*_z`，并使用 `*_raw` 重复分析作为敏感性验证。

---

## 9. A10 降糖药信息

当前 CGM CSV 中没有 ATC A10 降糖药输入，因此本阶段不执行药物排除。最终文件中：

```text
A10_excluded_flag = missing
A10_exclusion_status = "not_evaluated"
```

这表示“尚未评估”，不能解释为“参与者未使用降糖药”。获得用药数据后可另行建立
no-A10 sensitivity cohort。

---

## 10. 输出文件

全部运行结果统一位于：

```text
Data/cgm_deal/outputs/
├── data/
├── reports/
└── logs/
```

最终主文件为：

```text
outputs/data/05_cgm_core_phenotypes.csv
```

该文件每名参与者一行，共7,493行，包含身份、connection、设备、QC 信息，以及7个表型
的 raw、clean 和 Z-score。

重要中间文件包括：

```text
outputs/data/01_baseline_cgm_connections.csv
outputs/data/02_cgm_qc_pass_all.csv
outputs/data/03_iglu_core_raw.csv
outputs/data/04_cgm_core_clean.csv
outputs/reports/01_duplicate_baseline_connections.csv
outputs/reports/02_cgm_qc_exclusions.csv
outputs/reports/04_cleaning_parameters.csv
outputs/reports/04_outlier_actions.csv
outputs/reports/05_standardization_parameters.csv
```

---

## 11. 运行方式

在服务器 HPP 项目中运行：

```bash
cd /home/ec2-user/Desktop/HPP/Data/cgm_deal
conda activate pheno
bash run_cgm_preprocessing.sh
```

脚本按01–05顺序运行，任何关键字段缺失、键重复、daily coverage 不一致或合并失败时会
立即停止。完整终端输出同时保存在 `outputs/logs/`。

---

## 12. 后续分析建议

- Primary outcomes：`cgm_mean_z`、`cgm_cv_z`、`cgm_above_140_z`；
- Secondary outcomes：`cgm_gmi_z`、`cgm_in_range_63_140_z`、`cgm_mage_z`、
  `cgm_modd_z`；
- 使用 `participant_id` 与饮食和肠道微生物数据连接；
- 主模型使用 clean/Z-score，raw 结果作为异常值处理敏感性分析；
- 不根据血糖高低本身排除参与者；
- 每个结局单独报告分析样本量和缺失数量。
