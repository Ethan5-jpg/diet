# HPP 餐前餐后血糖探索

**组员请先阅读：[研究总结与结果说明](RESEARCH_SUMMARY.md)。** 这份总结覆盖 2026-09-24 至 2026-09-29 的八个分析模块及菌群时间核对，介绍研究问题、方法、关键结果、阴性结果、解释边界和复现所需材料，并嵌入三张结果图。

目前主体探索已经完成，进入结果整合与写作阶段。完整证据和逐项数值见 [详细研究报告](research_records/2026-09-29_完整研究盘点与分析报告/完整中文分析报告.md)。本仓库保留代码与聚合证据，完整重跑仍需 HPP 数据权限和服务器前序输出。

---

以下保留 **2026-09-24 首版 `hppmeal` 的运行说明**，供复现起始流程使用；其中“第一次运行”和“首版边界”描述的是该模块，不代表整个仓库目前的研究进度。后续模块入口见 [研究总结](RESEARCH_SUMMARY.md)。

首版运行包可整体上传服务器。所有原始数据只读，输出放入独立运行目录，不修改已有饮食、菌群或中介结果。

**建议第一次只运行到 `build`：核查输入 → 合并餐次 → 对齐 CGM → 计算餐后反应 → 生成汇总和示例图。** 该阶段是默认运行方式。先确认食物时间字段和数据质量，再执行已经写好的个体内模型。

首版分析代码为 **2026.09.24.3 再次复核版**。该版记录见 [再次复核](docs/再次复核_2026-09-24.md)，上轮记录保留在 [上传前复核](docs/上传前复核.md)。

## 1. 上传位置和第一次运行

推荐将整个文件夹上传为：

```text
/home/ec2-user/Desktop/HPP/meal_cgm_exploration/
```

服务器原始和既有处理数据仍在：

```text
/home/ec2-user/Desktop/HPP/Data/
```

使用此前运行 HPP 分析的 Python 环境。在该文件夹打开终端，运行：

```bash
bash run_server.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage check
```

这个快速检查只读文件头，不加载全量数据、不生成 SQLite；它会检查依赖、所有 CGM 文件头、NOVA/协变量/菌群字段，以及两个菌的名称是否唯一匹配。查看 `tables/00_preflight.csv` 和 `SUMMARY.md`，区分“可以构建餐次”和“后续模型输入齐全”。取样日期和实际进餐时间仍需按研究记录核实。

检查后运行正式构建（会自动创建新的运行目录）：

```bash
bash run_server.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage build
```

若服务器项目路径不同，只修改 `--data-root` 后的路径。程序从真实 CSV 字段读取参与者、访视和时间，不从文件名推测这些字段。

若环境提示缺少依赖，在你选定的研究环境安装：

```bash
python3 -m pip install -r requirements.txt
```

如需指定 Python：

```bash
HPP_MEAL_PYTHON=/你的环境/bin/python bash run_server.sh --data-root /服务器/HPP/Data --stage build
```

脚本兼容 Python 3.8+ 语法；不会自动安装或升级服务器环境。Python 3.8 需要由 pip 选择仍支持该版本的依赖，优先沿用已有的 HPP 环境。

较长运行可以使用：

```bash
nohup bash run_server.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage build > meal_build.log 2>&1 &
```

它会打印 `RUN_DIR=...`。SQLite 缓存、餐次表和汇总全部位于该目录。原始 CGM 逐文件、逐块处理；同一运行目录设有进程锁，第二个进程会被拒绝。需为缓存预留磁盘空间，规模取决于原始数据，不宜只按代码包大小估计。

## 2. 默认读取哪些输入

以下路径均相对于 `--data-root`，在 `config/default.json` 中集中配置。

| 输入 | 默认路径 | 首次 build 是否必需 |
|---|---|---|
| 食物记录 | `Transfer/diet_logging/diet_logging_events.csv` | 是 |
| 原始 CGM | `Transfer/cgm/timeseries/*.csv` | 是 |
| NOVA 食物映射 | `diet_deal/outputs/03_score_mapping/new_diet_indicators/new_diet_indicator_food_id_mapping.csv` | 可缺；会缺少 NOVA 分析资格 |
| 协变量 | `co-variant/outputs/data/02_covariate_master.csv` | analyze 时必需 |
| 菌群 CLR-Z | `gut_microbiome_deal/data/08_species_clr_zscore.csv` | 交互分析时需要 |
| 菌群取样日期 | 默认未指定 | 可选；缺少时不得称为基线菌群 |

CGM 必须含 `participant_id, cohort, research_stage, connection_id, collection_timestamp, glucose`。本地只有 glucose 一列的简化示例不能代替服务器全量文件；程序会明确拒绝。

本包限定 `10k` 队列和 `00_00_visit`。它建立新的餐次分析人群，不能直接沿用旧分析的 4,139 人作为结果人数。

协变量表还需包含 `cohort` 和 `cgm_research_stage`（兼容 `research_stage`）；菌群表需包含 `cohort,research_stage`。程序先筛选到同一队列和访视，再检查每人是否只有一条记录、连接餐次。缺少这些字段时，check 会提示后续模型尚未就绪，build 仍可进行，但 analyze 不会猜测访视。应从相应上游输出保留可核实的字段，不能为了通过检查手工填成基线。

取样日期默认未配置，缺失时明确标为未知；如果在配置中指定了日期文件，该文件必须存在且符合格式，路径写错不会被静默忽略。

## 3. 第一次跑完看什么

首先打开运行目录的 `SUMMARY.md`，再按顺序查看：

1. `tables/01_input_audit.csv`：输入行数、时间异常、营养缺失、共同参与者访视数。
2. `tables/02_meal_flow.csv`：食物聚合后有多少餐、多少人逐步满足条件。
3. `tables/02_cgm_window_reasons.csv`：缺基线、缺口、多传感器重叠等排除原因。
4. `figures/02_meal_examples.png`：若存在合格餐，查看真实曲线与食物时间是否合理。示例按顺序选取，不按效应大小挑选。
5. `tables/02_per_person_count_distribution.csv`、`02_connection_quality_distribution.csv`：每人信息量和传感器覆盖。

可以把这些汇总和图的截图发回。本包不自动打包或传输参与者级数据。`private/` 中的餐次表、连接质量、重复餐配对，以及包含来源路径的 manifest，留在服务器。

也可直接在终端显示适合截图的汇总：

```bash
python3 scripts/show_results.py --run-dir outputs/实际目录
```

## 4. 何时运行关联模型

确认所选字段确实代表实际进食时间（包括补记时是否保存实际食用时间），并检查示例图后，将使用的配置中：

```json
"consumption_time_verified": false
```

改为：

```json
"consumption_time_verified": true
```

这表示记录字段语义已核实，不是程序能验证的事实，也不等于所有日志都准确。随后使用第一次运行打印的实际目录：

```bash
bash run_server.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage analyze --run-dir outputs/这里替换为第一次生成的meal目录
```

时间语义这一项可以在已完成 build 后更新；其他分析规则、输入或代码变化时，必须生成新运行。恢复时校验配置、代码和输入内容，避免混用不同版本。若修改的是另外保存的配置文件，每次都加 `--config 该文件.json`。

如语义已确认，也可一次运行全部阶段：

```bash
bash run_server.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage all
```

主要输出：

- `03_cohort_flow.csv`：新的餐次分析人群。
- `03_covariates_scope.csv`、`03_microbiome_scope.csv`：协变量和菌群的队列、访视筛选及排除行数。
- `03_within_person_models.csv`：NOVA4 主模型、峰值次结局和两项敏感性分析。
- `03_taxon_mapping.csv`：两种菌实际匹配到的列，必须核查。
- `03_microbiome_interactions.csv`：两种菌各自的交互及预定检验家族 FDR。

`estimate` 的主暴露单位为 **NOVA4 能量占比增加 10 个百分点**；交互为在此基础上 **菌群原 CLR-Z 增加 1 单位** 的关联差异。净增量面积单位是 mg/dL·min，峰值单位是 mg/dL。

模型同时输出设计矩阵的条件数。对数值上近似共线的设计停止估计，不勉强输出不稳定系数。

## 5. 默认分析规则

- 同一参与者、队列、访视内，条目距该餐开始不超过 30 分钟才合并，避免零食不断串成一顿饭。
- 所有已记录进食事件均参与前后间隔判断，包括零热量饮料；达到 50 kcal 和 5 g 碳水的事件才作为候选分析餐。
- 餐前 30 分钟至少两个有效点，最近点距进餐不超过 15 分钟；基线取这些点的中位数。
- 两小时窗至少 7 个餐后观测点；最大允许间隔 20 分钟；端点可在短间隔内线性插值，不外推、不跨长缺口。
- 主结局为净增量面积；另输出正面积用于核查，两者不是同一个指标。
- 主分析餐距前一事件结束至少 120 分钟，下一事件开始必须晚于餐后 120 分钟。首末事件的未知间隔不当成无限长；只分析这样的餐会改变目标人群。
- 重复可疑食物条目不擅自删除，相关餐暂不进入模型；重复 CGM 同值合并，冲突值失效，多连接涉及同一窗口时排除。
- 某个参与者/访视出现无法解析的饮食时间时，仍可输出其余餐次供核查，但该访视不进入模型，避免把丢失时间的进食当成不存在。
- NOVA4 分母为整餐总能量。主规则要求分类覆盖至少 80%，另看 95%；未分类能量不是“非超加工”。
- 模型需至少 50 人、每人至少 5 个完整可用餐且 NOVA4 范围至少 10 个百分点；阈值只是首版探索规则。

完整方法、解释边界和字段见 [分析规则](docs/分析规则.md)。

## 6. 状态和恢复

| 状态 | 意义 |
|---|---|
| PASS | 该阶段完成；不表示发现显著关联 |
| REVIEW_REQUIRED | 字段含义或必要资料未核实；先查看 reason |
| PARTIAL | 部分预定模型未能估计；查看结果表各行 |
| FAILED | 输入/计算异常；查看 SUMMARY.md 和 logs/error.txt |
| RUNNING | 阶段尚未完成；若进程已退出，说明未正常结束 |
| INTERRUPTED | 接收到键盘中断；本阶段未完成 |

退出码 0 表示请求阶段完成；2 表示失败、待核查或部分模型未完成；键盘中断返回 130。后两种情况不能当作完整成功。不要通过降低样本门槛来追求显著性。修复来源或更改规则后启动新运行，旧输出保留。

只完成 audit 后，可以续跑：

```bash
bash run_server.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage build --run-dir outputs/实际目录
```

build 已完成时，下一步使用 analyze，不能覆盖已完成的 build。中途失败的缓存不作部分续算，重新启动新运行。每次恢复会重新核验原始输入指纹，因此读盘需要时间。

恢复还会核对 CGM 文件集合，新增或删去文件不能沿用旧缓存。原先缺失的 NOVA 映射后来补齐时，也需要重新构建。拒绝无效恢复请求时，不改写已完成运行的配置和摘要。已经核实的时间语义不能在同一运行中撤回；应另建运行。

每阶段开始时摘要即显示 RUNNING；完成最终输入校验后才保存成功结果。中断、计算失败或最终校验失败时，不要读取残留结果表作为完整结果，`show_results.py` 会阻止展示。强制结束进程或断电可能留下 RUNNING，需新建运行；锁随进程退出释放，无须手工删除 `.run.lock`。

## 7. 本地验证与首版边界

测试命令：

```bash
python3 -m pip install pytest
python3 -m pytest tests -q
```

合成数据只在测试临时目录生成，带有 `SYNTHETIC` 标记。包内不附模拟结果冒充真实结果。

首版 `hppmeal` 实现两小时餐后计算、重复餐描述、时段描述和有限个体内模型。该模块未实现四小时曲线、正式重复性 ICC、进餐时段因果模型、每餐 hPDI、因果中介、所有菌扫描或预测模型。后续 `followup` 已另外完成限定场景下的日期留出预测，其他后续工作与当前证据边界见 [研究总结](RESEARCH_SUMMARY.md)，不能将本段首版范围理解为整个项目尚未开展预测。
