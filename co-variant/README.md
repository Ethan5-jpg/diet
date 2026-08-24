# HPP协变量处理

这套程序分为两步：先从服务器原始HPP数据目录重新复制8个parquet文件并转换成CSV，再生成一份“一名参与者一行”的协变量总表。转换时会把parquet的MultiIndex恢复成普通字段，因此不会丢失`participant_id`、`cohort`、`research_stage`、`array_index`和`connection_id`。

## 固定路径

原始HPP数据：

```text
/home/ec2-user/studies/hpp_datasets
```

项目目录：

```text
/home/ec2-user/Desktop/HPP/Data/co-variant
```

## 处理的8个文件

```text
population/population.parquet
sociodemographics/initial_medical.parquet
lifestyle_and_environment/lifestyle_and_environment.parquet
medications/medications.parquet
anthropometrics/anthropometrics.parquet
family_history/initial_medical.parquet
medical_conditions/medical_conditions.parquet
cgm/cgm.parquet
```

不复制HRT问卷，不复制`diet_logging_events.parquet`。CGM虽然以前处理过，本程序仍会从原始HPP目录重新复制和转换。

## 第一步：准备源文件

把整个`co-variant`目录上传到服务器对应位置后运行：

```bash
cd /home/ec2-user/Desktop/HPP/Data/co-variant
conda activate pheno
bash run_prepare_covariate_sources.sh
```

程序只刷新上述8个受管理的raw和CSV文件，不删除目录中的其他文件。已有的同名受管理文件会被新的原始副本和转换结果替换。

## raw目录结构

```text
raw/
├── population/population.parquet
├── sociodemographics/initial_medical.parquet
├── lifestyle_and_environment/lifestyle_and_environment.parquet
├── medications/medications.parquet
├── anthropometrics/anthropometrics.parquet
├── family_history/initial_medical.parquet
├── medical_conditions/medical_conditions.parquet
└── cgm/cgm.parquet
```

## CSV输出

```text
csv/population.csv
csv/sociodemographics_initial_medical.csv
csv/lifestyle_and_environment.csv
csv/medications.csv
csv/anthropometrics.csv
csv/family_history_initial_medical.csv
csv/medical_conditions.csv
csv/cgm.csv
```

每个文件完成后，终端会显示源路径、raw路径、CSV路径、行数、字段数、恢复的索引字段和SHA-256校验结果。

全部成功时终端最后显示：

```text
FILES_PROCESSED=8
PROCESS_COMPLETED=True
```

如果任意一个源文件缺失，程序会在复制开始前停止，不会只生成部分结果。

## 第二步：生成参与者协变量总表

第一步成功后，在同一目录运行：

```bash
cd /home/ec2-user/Desktop/HPP/Data/co-variant
conda activate pheno
bash run_build_covariate_master.sh
```

第二步还会读取两份已经处理好的结果：

```text
/home/ec2-user/Desktop/HPP/Data/cgm_deal/outputs/data/01_baseline_cgm_connections.csv
/home/ec2-user/Desktop/HPP/Data/diet_deal/outputs/05_diet_scores/amed/amed_participant_scores.csv
```

前者用于保证CGM设备和采集日期与血糖处理阶段选定的基线connection完全一致；后者提供hPDI模型需要的日均酒精摄入量。

### 参与者范围

总表以`population.csv`中的10k队列参与者为底表，使用左连接合并所有来源。因此：

- 每名参与者最多一行；
- 协变量缺失时保留为空；
- 不做均值、中位数或其他插补；
- 不在本步骤删除糖尿病、A10用药或协变量不完整者；
- 后续每个模型根据自己的完整性标记进行完整案例分析。

### 14个协变量

| 协变量 | 总表字段 | 主要来源/处理 |
|---|---|---|
| 年龄 | `age_years` | 出生年月与基线CGM日期计算；无CGM时用最早可用基线日期 |
| 性别 | `sex` | `population`，0=女性、1=男性 |
| 教育 | `education_level` | `sociodemographics`，保留为分类编码 |
| 吸烟 | `smoking_status` | `lifestyle_and_environment`，current/former/never |
| 睡眠 | `sleep_duration_hours_day` | `sleep_hours_daily` |
| 身体活动 | `physical_activity_met_h_day` | 步行、中等和剧烈活动换算MET-h/day |
| 维生素使用 | `vitamin_use` | ATC前缀`A11*` |
| 激素使用 | `hormone_use` | ATC前缀`G03*`、`H01*`至`H05*` |
| CGM设备 | `cgm_device_type` | 已选定基线CGM connection |
| 酒精摄入 | `alcohol_intake_g_day` | AMED完整酒精记录；仅hPDI模型需要 |
| BMI | `bmi` | `anthropometrics` |
| NSAID/阿司匹林 | `nsaid_aspirin_use` | ATC前缀`B01*`、`M01*`、`N02*` |
| 糖尿病家族史 | `family_history_diabetes` | 1型或2型糖尿病家族人数大于0；信息不足时保留为空 |
| 心血管家族史 | `family_history_cvd` | `sudden_death_family` |

身体活动的固定公式为：

```text
(3.3 × 步行天数 × 步行分钟
 + 4.0 × 中等活动天数 × 中等活动分钟
 + 8.0 × 剧烈活动天数 × 剧烈活动分钟) / 60 / 7
```

当某类活动天数为0时，后续时长问题的空值按结构性0处理；活动天数缺失，或活动天数大于0而时长缺失时，身体活动结果仍保留为空。当前脚本不会把未收集的剧烈活动天数擅自填成0。

### 两类排除指标

总表同时生成但不立即执行以下排除：

```text
a10_medication_use
known_diabetes
exclude_diabetes_or_a10
```

这些字段使用三状态：`1`表示符合，`0`表示有相应来源记录且不符合，空值表示没有足够来源信息。不能把空值自动当成`0`。

### 模型完整性标记

总表提供AMED和hPDI各自的Model 2、Model 3、Model 4完整性标记：

```text
amed_model2_covariates_complete
hpdi_model2_covariates_complete
amed_model3_covariates_complete
hpdi_model3_covariates_complete
amed_model4_covariates_complete
hpdi_model4_covariates_complete
```

Model 2包括年龄、性别、教育、吸烟、睡眠、身体活动、维生素、激素和CGM设备；Model 3在Model 2上增加BMI；Model 4在Model 2上增加NSAID/阿司匹林、糖尿病家族史和心血管家族史。hPDI的每个模型还要求酒精摄入完整。

### 第二步输出

```text
outputs/data/02_covariate_master.csv
outputs/reports/02_covariate_completeness.csv
outputs/reports/02_source_coverage.csv
outputs/reports/02_categorical_values.csv
outputs/reports/02_build_summary.csv
```

全部成功时，终端最后显示：

```text
COVARIATE_MASTER_COMPLETED=True
```
