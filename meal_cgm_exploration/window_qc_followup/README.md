# 两周CGM/饮食缺失审计与同访视菌群后续分析

版本 **2026.10.04.2**。本文件所在的`window_qc_followup`就是已经准备好的完整文件夹，可直接整文件夹上传，无需解压ZIP。本地只完成合成验证，**尚无新的真实服务器缺失率或关联结果**。

本版复查修正了计划未匹配者的原始读数计数、无法确定窗口时的NA缺失率、整段缺失的互斥分类，以及两周窗口结束边界。入口和默认阈值不变。

## 上传与运行

只上传整个 `window_qc_followup/` 文件夹，放到服务器：

```text
/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/window_qc_followup/
```

它应与 `hppmeal/`、`seven_diet_meals/`、`gut_meal_bridge_v2/`、`quality_timecourse/`、`outputs/` 并列。沿用原pheno环境及原代码。新包核对依赖指纹，不覆盖旧包，不安装或升级服务器软件。

在服务器 `meal_cgm_exploration` 目录运行：

```bash
bash window_qc_followup/run_window_qc.sh --data-root ..
```

默认执行：检查冻结来源 → 两周覆盖审计 → 复现42项原饮食和48项原菌群模型 → 质量筛选后264项预定模型。用时取决于缓存体积与原模型规模。原始CSV不再全量导入，读取已冻结的SQLite缓存；缓存SHA256检查需要读盘。

如希望先只统计缺失，运行：

```bash
bash window_qc_followup/run_window_qc.sh --data-root .. --stage audit
```

`check`只核对输入和身份；`audit`计算缺失、无关联结果；`all`完成两者。每次创建独立运行，不恢复/覆盖旧目录；先audit再all会生成新的目录，保留两次结果。

指定环境（如需要）：

```bash
HPP_MEAL_PYTHON=/你的pheno环境/bin/python bash window_qc_followup/run_window_qc.sh --data-root ..
```

长任务：

```bash
nohup bash window_qc_followup/run_window_qc.sh --data-root .. > window_qc_followup.log 2>&1 &
```

## 回传终端输出

程序会打印 `RUN_DIR=...` 以及第一张覆盖汇总。查看最新尝试的四页：

```bash
python3 window_qc_followup/run_analysis.py --show latest --brief
```

分页便于复制或截图：

```bash
python3 window_qc_followup/run_analysis.py --show latest --page 1
python3 window_qc_followup/run_analysis.py --show latest --page 2
python3 window_qc_followup/run_analysis.py --show latest --page 3
python3 window_qc_followup/run_analysis.py --show latest --page 4
```

1. 两周覆盖、缺失分布、主/敏感性筛选人数。
2. 七饮食评分与餐后净增量、峰值和碳水交互。
3. 四候选菌与餐前、餐后及strict结果，菌群日期审计。
4. 筛选前后/阈值敏感性、整体CGM参照与完成情况。

只回传聚合页即可；不回传private个体表。`latest`指最新尝试，失败时不会回退展示旧成功结果。失败请发错误信息和RUN_DIR。正式写稿时取CSV的完整精度，终端页为舍入显示。

## 本轮缺失率怎么定义

**CGM主阈值严格<5%。** 14×24小时、15分钟一次，共1344槽；有效唯一槽不足的部分算缺失，包括无读数、无效或冲突读数。同值重复只计一次，多连接不拼接，不插值补缺失。另报最长连续缺口、内部/尾部缺失、超出两周的读数与离开15分钟刻度的读数。

**窗口起点有两种证据等级。** 默认用该连接首条可解析时间戳，标记FIRST_RECORDED_POINT_PROXY。与只按首末实际跨度相比，它会补计提前结束产生的尾部缺失；但是第一条记录前的丢失仍未知。官网CGM的collection_timestamp同样描述首数据点，不能当作激活时间。设备采样刻度相位取首条时间戳，与计划起点分开。

如果有已核实的佩戴/计划起点，可在包外准备CSV（不要改原数据），字段：

```text
participant_id,cohort,research_stage,connection_id,planned_start,diet_start_date
```

planned_start必须为有时区时间戳；diet_start_date可选、格式YYYY-MM-DD。传入：

```bash
bash window_qc_followup/run_window_qc.sh --data-root .. --window-starts /实际路径/已核实的窗口.csv
```

显式计划表中未匹配的连接不回退为首记录代理、不进入质量模型。无法确定窗口时，CGM和饮食缺失率记NA，已有原始CGM行数仍如实统计；有明确窗口而没有读数时，CGM缺失率为100%。不要为通过检查而填造计划时间。未提供计划时结果可用于代理窗口质量敏感性，不能声称已验证完整计划佩戴率。

**饮食主阈值默认严格<5%，单位是“无记录日比例”。** 默认取CGM起点的当地日期起连续14日；显式diet_start_date可覆盖这个代理选择。日期既有daily metadata.food_items>0，又有原缓存事件，且时区一致，才记为有记录。少1天=7.14%，所以<5%要求14/14天，<10%至少13天，<20%至少12天。首日可能只是部分CGM日；程序同时报告窗口内完整日覆盖。饮食日历日窗口与CGM336小时窗口可能有部分边界差异，餐模型只取二者内的餐。

**有记录≠饮食无漏记。** 即使每天只有一条记录也不能推断其他餐均记录；此数据无法直接测量漏餐率。total_logging_days可能包括窗口外日期，不拿它作为分母。日表/事件不一致、时区不一致和无法解析的时间均单独审计；未知饮食不填成零摄入。

默认主方案CGM/饮食均<5%；另固定10%/20%重叠样本敏感性。若在看结果之前确定饮食主阈值为10%、CGM仍5%，用：

```bash
bash window_qc_followup/run_window_qc.sh --data-root .. --diet-main-threshold 0.10
```

阈值选择会冻结到analysis_spec.json。不能跑完后根据显著性把宽松结果改称主分析。5%/10%/20%均为本研究操作性规则，不是HPP或Nature规定。

## 菌群与后续研究口径

按本次用户转述的老师确认，把同访视的菌群、饮食和CGM视为研究测量窗口；**菌群记录在CGM之后不作为排除条件**。日期先后、间隔、未知及具体样本身份仍保留审计。单一访视记录匹配不等于已核实测序样本编号。

原四菌模型已完成。新贡献是两周质量规则下的稳健性检验，不再把采样先后描述为“不能做关联”。结果仍是观察性、同队列候选关联，不是菌群基线预测、因果中介、机制验证或独立复制。

七评分全部保留：AHEI、AMED、更正hPDI、rEDIH、EAT13、NOVA4、Carbohydrate_pct。沿用原Z、模型调整、每人餐数/日期数和交互碳水范围规则，不在筛选后重新标准化。评分仍使用上游原记录范围，本轮日期覆盖筛选不等于重新计算严格14日评分。四菌固定T023、T046、T068、T096，使用原CLR-Z。

QC餐要求餐前30分钟起点不早于CGM窗口起点，餐后120分钟终点严格早于窗口结束，日期在饮食参考窗口。时间协变量先在全部原餐中固定，再筛餐；重新检查每人餐数/日期数。整体CGM参照沿用原连接指标，可能包含336h之外读数，不能冒充本轮重算的固定两周指标。它与餐次可能共享读数。

详见PLAN.md：主质量、strict主质量、10%、20%四设置；每组42饮食+24菌群=66模型，共264新规格。CI为逐项95%；沿用原家族FDR，另报每组饮食q42、菌群q24及餐次q12。所有NE保留原因和计划检验分母，不降低门槛。

## 固定来源与输出

默认旧运行：

| 来源 | 运行名 |
|---|---|
| 原餐次与完整缓存 | meal_20260924T143630_752200Z |
| 七评分原模型 | seven_diets_20260925T093822_426339Z |
| 最终评分和整体CGM来源 | nutrition_bridge_20260926T071736_080048Z |
| 已完成四菌模型 | gut_meal_bridge_v2_20260929T044951_469958Z |

搬家时用--meal-run/--seven-run/--nutrition-run/--gut-run指定实际目录，不会自动选其他运行。输入路径变更可用旧--sources-config/--score-sources；仍须符合旧来源指纹。饮食日表默认Transfer/diet_logging/diet_logging.csv；--diary可明确路径。CGM metadata默认Transfer/cgm/cgm.csv，仅当包含身份/队列/访视/连接字段时扩大审计人群，不从文件名猜身份。

新输出：`outputs/window_qc_followup_时间戳/`。

| 文件 | 用途 |
|---|---|
| tables/01_coverage_population.csv、01_coverage_quantiles.csv | 全部可审计目标访视记录的覆盖统计 |
| tables/02_threshold_flow.csv | 记录阈值流失；不同于最终模型人数 |
| tables/03_*_reproduction.csv | 原42+48模型成员/效应复现 |
| tables/diet_models.csv、gut_models.csv | 完整210饮食+144菌群规格，包括原结果参照 |
| tables/04_* | 最终样本和菌群时间审计 |
| tables/05_selection_balance.csv | 保留/排除人群均值、SD、标准化差异 |
| private/connection_quality.csv、diet_window_days.csv | 逐人逐连接/日期质量，只留服务器 |
| private/model_membership.csv、gut_person_inputs_*.csv | 模型成员/个体输入，只留服务器 |
| analysis_spec.json、manifest.json | 固定设计、来源和结果指纹 |

审计人群是原缓存+可用CGM元数据+目标访视饮食日表+可选计划表，**不是全部HPP入组人群**；没有任何这些记录的人不可识别。缓存继承原始时间解析和valid规则，upstream percentage_of_cgm_datapoints_lost_in_qc与固定1344分母缺失率不是同一指标。

状态：CHECK_PASS/AUDIT_COMPLETE仅完成对应阶段；COMPLETE_WITH_LIMITATIONS表示264新模型都可估计且原90模型复现；PARTIAL表示存在NE但保留全部规格；FAILED/INTERRUPTED请查看ERROR.txt。PARTIAL退出码2，失败2、中断130，其余成功0；完成不代表显著。没有复制旧运行private缓存或完整结果到本地，不能提前报告真实缺失率。

本地验证见REVIEW.md；研究交接更新见研究衔接说明.md。官网来源与Nature论证定位在PLAN.md。
