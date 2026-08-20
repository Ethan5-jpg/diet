# HPP CGM 正式预处理设计

## 目标和输入

在服务器 `Data/cgm_deal` 中建立可重复运行的 CGM 主流程，输入为已经恢复五个身份
字段的完整 CSV：

- `Data/Transfer/cgm/cgm.csv`
- `Data/Transfer/cgm/iglu.csv`
- `Data/Transfer/cgm/iglu_daily.csv`

三表结构已由服务器终端确认：`cgm` 与 `iglu` 均有9,497个 connection 且完整键
一一对应；`iglu_daily` 覆盖全部 connection；baseline 有9,410个 connection、
9,405名参与者，其中5人有两个 baseline connection。

## 方案选择

采用五个顺序脚本，每一步写一个可独立核对的 checkpoint。相比单体脚本，这种结构
可以在 participant 选择、connection QC、表型合法性和异常值方法之间保留清晰边界；
任何一步失败都不会覆盖前一步结果。临时源数据 audit 脚本删除，不进入正式流程。

## 数据流

### 01 Baseline connection selection

从 `iglu_daily` 按 connection 派生 `cgm_days`。只保留 `cohort=10k`、
`research_stage=00_00_visit`，每名参与者按照以下顺序选一条 connection：

1. `cgm_days` 最大；
2. `percentage_of_cgm_datapoints_lost_in_qc` 最小；
3. 若源数据有 datapoint count，则 count 最大；
4. `collection_timestamp` 最早。

当前 CSV 没有 datapoint count。若重复候选在天数和 loss 后仍并列，脚本必须停止，
不能直接跳到时间戳；已观察的5名重复者不存在这种并列。

### 02 Connection QC

将选中的 connection 与 `iglu` 按五字段完整键一对一合并，保留：

- `cgm_days >= 10`；
- QC loss fraction `<= 0.30`；
- `cgm_mean`、`cgm_cv`、`cgm_above_140` 均非缺失。

血糖水平本身不作为 cohort exclusion。根据当前只读审计，预计9,405名 baseline
参与者中7,493人通过，但脚本不为匹配该人数而改变阈值。

### 03 Core phenotype extraction

固定提取 `cgm_mean`、`cgm_cv`、`cgm_above_140`、`cgm_gmi`、
`cgm_in_range_63_140`、`cgm_mage`、`cgm_modd`。保留 raw 值；百分比必须在
0--100，CV/MAGE/MODD/GMI不得为负。非法值设为缺失并写报告，不根据健康程度删人。

### 04 Robust phenotype cleaning

对每个表型独立：排序非缺失值，取覆盖 `ceil(0.95*N)` 个观测的最窄窗口；使用落在
窗口边界内的全部观测计算均值和样本标准差。距离该中心超过8个标准差的值设为缺失，
其余值限制到中心正负5个标准差。raw 列永不覆盖；参数、极端值和温莎化数量全部
写入报告。

### 05 Standardization

使用清洗后非缺失样本的均值和样本标准差产生 Z-score。最终文件每人一行，包含
身份/设备/QC 信息以及7个表型的 raw、clean、z 三轨值。当前没有 medication 输入，
因此 `A10_excluded_flag` 保持缺失，并增加 `A10_exclusion_status=not_evaluated`，不能
把未评估误写为未用药。

## 固定输出

- `outputs/data/01_baseline_cgm_connections.csv`
- `outputs/reports/01_duplicate_baseline_connections.csv`
- `outputs/data/02_cgm_qc_pass_all.csv`
- `outputs/reports/02_cgm_qc_exclusions.csv`
- `outputs/data/03_iglu_core_raw.csv`
- `outputs/data/04_cgm_core_clean.csv`
- `outputs/data/05_cgm_core_phenotypes.csv`
- 每一步对应的 summary/parameter/outlier 报告。

## 失败原则与兼容性

connection-level 表按 `participant_id + cohort + research_stage + connection_id` 连接；
`iglu_daily.array_index` 是逐日数组行号，不属于 connection 键。其余 connection-level
表保留五字段完整键核对。所有关键连接使用 `validate=one_to_one`；缺字段、重复键、
daily coverage 不完整、无法执行既定重复选择优先级、清洗标准差无效或输出 participant
重复时立即停止。
脚本兼容服务器 Python 3.7.16，不使用 Python 3.9+ 内置泛型。
