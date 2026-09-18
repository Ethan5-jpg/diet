# CGM 论文新增17指标：服务器处理包

版本：2026-09-18.1。

## 已完成的服务器运行

2026-09-18用户提供的服务器摘要显示：运行`run_20260918T050730_936213Z_07fcb42c`状态为`completed`，7493人全部匹配，17项Z均成功计算。详细方法、逐项结果和最终服务器路径见[CGM论文新增17指标：处理过程与结果](CGM论文新增17指标_处理过程与结果_2026-09-18.md)。记录依据为服务器截图，本地未下载完整结果表。

## 这次处理什么

把论文中尚未纳入此前关联分析的16类表型补入；其中 Interday SD 的两个候选字段 **SDb、SDdm分别提取**，所以本包涉及 **17个字段**。

- 默认沿用已有7493人的CGM最终表，不重选人、不重选设备连接。
- GMI、MODD在旧表已处理：校验来源后直接保留原raw/clean/z。
- 其余15项追加source、raw、clean、z及状态列。
- 原平均血糖、CV、TAR140、MAGE、TAR180、TBR70、TIR70–180及旧辅助列全部保留。
- 原7个已分析指标 + 17个候选字段 = 24个候选分析指标；不是24个相互独立的生理维度。旧表还可能包含其他辅助表型，因此最终CSV的表型总数不以24作为断言。
- 此包只做数据处理和审计，不运行饮食/菌群关联，不修改已有分析配置。

## 1. 上传位置

把**整个 `cgm论文新增17指标` 文件夹**上传到服务器：

```text
/home/ec2-user/Desktop/HPP/Data/cgm_deal/cgm论文新增17指标/
```

上传后的结构：

```text
cgm_deal/
├── outputs/                         # 服务器已有旧结果
├── cgm新增四指标/                     # 旧扩展代码，可保留原有大小写
└── cgm论文新增17指标/
    ├── README.md
    ├── requirements.txt
    ├── run_preprocessing.sh
    ├── show_summary.sh
    ├── config/outcomes.json
    └── scripts/process_cgm17.py
```

本包自带全部运行代码，不需要导入父目录脚本；输入数据仍读取服务器现有文件。默认路径相对新文件夹解析，不依赖启动终端的位置。

## 2. 运行

使用此前成功运行的 `pheno` Python环境。依赖仅pandas、NumPy；已有环境通常无需安装。

```bash
cd /home/ec2-user/Desktop/HPP/Data/cgm_deal/cgm论文新增17指标
bash run_preprocessing.sh
```

这一条命令执行：检查输入 → 按完整键提取 → 合法性检查 → 清洗 → 每项独立标准化 → 输出核对报告和最终表。

如果希望先只核查：

```bash
bash run_preprocessing.sh --check-only
```

`--check-only`也会计算清洗和标准化的预览参数，但不写正式最终表；无阻断问题后再运行不带`--check-only`的命令。

## 3. 默认输入（已固定，不自动选择“最新”文件）

旧表使用此前截图确认成功的最终表：

```text
../outputs/cgm_extension/data/finalize_20260915T074925_139680Z_b375d22f/06_cgm_extended_phenotypes.csv
```

来源使用：

```text
../../Transfer/cgm/iglu.csv
```

如果实际文件已移动，明确指定路径：

```bash
bash run_preprocessing.sh \
  --base-csv /实际路径/06_cgm_extended_phenotypes.csv \
  --iglu-csv /实际路径/iglu.csv
```

不自动退回早期核心表，以免丢失已经加入的TAR180、TBR70、TIR。默认要求旧表7493行；`--expected-rows`只用于有意改变基准队列或测试，不能为绕过报错随意修改。`PYTHON_BIN`可指定Python，例如`PYTHON_BIN=/path/to/python bash run_preprocessing.sh`。

## 4. 字段与处理规则

| 字段 | 含义 | 处理 |
|---|---|---|
| cgm_median | 血糖中位数 | 连续指标规则 |
| cgm_gmi | GMI，平均血糖的线性换算 | 原raw/clean/z复用 |
| cgm_iqr | 四分位距 | 连续指标规则 |
| cgm_mad | 中位绝对偏差 | 连续指标规则 |
| cgm_mag | 平均绝对血糖变化速率，区别于MAGE | 连续指标规则 |
| cgm_modd | 不同天相同时点血糖差异 | 原raw/clean/z复用 |
| cgm_sd_roc | 血糖变化速率标准差 | 连续指标规则 |
| cgm_sdhhmm | 先按时刻跨日平均，再计算时刻间SD | 连续指标规则 |
| cgm_sdb | 同一时刻跨日计算SD，再取平均 | 连续指标规则；独立保留 |
| cgm_sddm | 每天平均血糖的SD | 连续指标规则；独立保留 |
| cgm_sdwsh | 小时滑动窗口内SD的平均 | 连续指标规则 |
| cgm_hbgi | 高血糖风险指数 | 连续指标规则 |
| cgm_lbgi | 低血糖风险指数 | 连续指标规则 |
| cgm_adrr | 平均每日风险范围 | 连续指标规则 |
| cgm_cogi | 综合CGM评分，0–100 | 合法范围内原值保留 |
| cgm_grade | GRADE风险评分 | 连续指标规则 |
| cgm_grade_eugly | 正常范围对总GRADE的贡献百分比，不是TIR | 合法范围内原值保留 |

### 匹配与原数据保留

使用`participant_id + cohort + research_stage + array_index + connection_id`五字段键。只规范化用于匹配的键副本：去掉首尾空格、将整数形式的array_index统一；不把参与者编号转成数字，也不改旧单元格。基准队列限定`10k`、`00_00_visit`，每人一行。

来源可以包含其他未被选中的连接；重复完整键、空键、缺列或任何旧连接未匹配均停止。CSV重复列名、行宽不一致也停止。旧列内容和行序在内存及写出重读后逐一验证。

### 合法性检查

缺失标记（空字符串、NA、N/A、NaN、null、none等）、非数值和无穷值记为缺失并标明原因。17项均要求非负，COGI和GRADE_eugly另要求≤100。不填0，不把异常值伪装为真实零值。`_source`保留源文件原字符串，`_raw`为通过合法性检查的数值。

### 连续指标规则

参考原`04_clean_cgm_phenotypes.py`：

1. 在该指标合法且非缺失的全基准队列数值中，找包含至少ceil(0.95×N)个值的最窄区间；并列选较低区间，边界同值全部纳入。
2. 用该区间中的均值和样本SD（ddof=1）计算阈值。
3. 超出均值±8SD设为缺失；剩余超出±5SD的值缩尾至边界。
4. 每项单独处理，不因为某人的一个结局缺失而删掉整行。

若非恒定数据的最窄95%区间SD为0（例如99%为0），本次状态为`blocked_review_required`，保留提取表和审计报告，**不生成正式最终表**。不会悄悄改用另一套方法；需根据服务器分布明确后续规则。

如果全缺失、只有一个有效值或整列恒定，保留其合法数值但Z为缺失，明确标记不可估计。其余字段照常处理，最终状态为`completed_with_unavailable_outcomes`；这些不可估计项不能直接进入关联分析。

### 有界指标规则

COGI、GRADE_eugly只保留0–100内的合法原值，不使用5/8SD缩尾。此规则参照之前比例扩展的处理原则，是本次预先规定的选择，不声称与论文每一项处理完全相同。

### 标准化与复用

新Z使用每项clean的全部有效基准队列样本：`z=(clean-mean)/sample_sd`，ddof=1。输出均值、样本SD和有效N并验证Z均值约0、样本SD约1。不使用17项完整样本交集，也不在此步套用饮食/菌群协变量筛选。

GMI、MODD的旧raw与匹配来源进行数值核对，旧Z与旧clean标准化结果核对；一致后保留旧原文本，不重做清洗。报告中的旧项异常处理计数为NA，因为本次没有重跑旧清洗，不能把它理解为0次异常处理。

## 5. 输出与如何截图

每次运行创建唯一目录：

```text
outputs/run_时间戳_随机编号/
├── data/
│   ├── 01_extracted_audit.csv
│   └── 07_cgm_paper_extended_phenotypes.csv   # 仅正式处理通过时生成
├── reports/
│   ├── manifest.json                       # 状态、输入/代码SHA256、匹配及保留验证
│   ├── outcomes_used.json                  # 本次使用的字段及规则快照
│   ├── raw_distributions.csv               # 缺失、0/100值、分位数、均值和SD
│   ├── processing_parameters.csv           # 清洗阈值、处理数量、Z参数和状态
│   ├── value_issues.csv                    # 源值非法/缺失的逐项记录
│   ├── cleaning_actions.csv                # 缩尾或设缺失的逐项记录
│   ├── 运行摘要.txt
│   └── 处理结果.md
└── logs/run.log
```

终端最后直接显示简洁摘要。关闭终端后可重看最近一次运行（包括失败或check-only）：

```bash
bash show_summary.sh
```

指定某一次运行：

```bash
bash show_summary.sh outputs/run_实际目录名
```

把这段摘要截图回来即可：包含STATUS、人数、17项有效N、缩尾/缺失处理数量、Z状态和最终路径。完整细节保存在CSV与日志，不会全部刷到终端。

### 状态说明

- `completed`：已生成正式表，17项Z均可计算。
- `completed_with_unavailable_outcomes`：已生成正式表，但部分结局不能计算Z，详见逐项状态。
- `checked_no_final_table`：只检查和预览，未生成正式表。
- `blocked_review_required`：清洗规则发生退化等阻断问题，审计已保留，无正式表；退出码2。
- `failed`：输入、匹配、复用或验证错误，无可确认成功的正式表；退出码1。摘要说明原因，完整错误见日志。

重复运行不会覆盖任何旧运行。后续关联分析要显式指定成功运行的`FINAL_FILE`，不能随意选择“最新目录”。Z可计算不证明正态性或线性模型适用；本包不重新核查CGM原始曲线、不修改已有药物排除状态、不执行多重比较校正。

## 来源与验证范围

- [HPP官方CGM字段字典](https://knowledgebase.pheno.ai/datasets/017-cgm.html)：上述字段属于连接级`iglu`表，不使用逐日`iglu_daily`。
- [iglu SD计算定义](https://irinagain.github.io/iglu/reference/sd_measures.html)：SDb与SDdm分别保留，尚不替论文指定哪一个叫Interday SD。
- [COGI定义](https://irinagain.github.io/iglu/reference/cogi.html)、[GRADE Euglycemia定义](https://irinagain.github.io/iglu/reference/grade_eugly.html)。
- 本地只使用模拟数据测试；本地Transfer不是服务器完整数据，未据此生成真实7493人的结果。
- 测试位于父目录`tests/test_cgm_paper17.py`，不需要上传测试才能运行。
- 2026-09-18本地14项模拟测试全部通过；包括独立复制上传包的CLI运行、源文件变化阻断及与旧清洗算法逐值对照。另通过shell语法和Python 3.6语法检查；实际测试运行时为Python 3.12。用户随后已回传服务器成功运行摘要，结果记录见本文开头链接。
