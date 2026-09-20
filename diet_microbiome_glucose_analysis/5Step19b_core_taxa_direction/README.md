# Step19b 核心菌方向整合：上传运行包

这一步读取既有18f/18g/19a和三套原始中介结果，补齐a、b、间接效应、原始CI及N，生成核心菌方向汇总。**不重跑关联/中介/Bootstrap，不重算FDR，不修改既有结果。**

状态（2026-09-20更新）：**数值方向整合、主要结果解读和两张汇总图的截图核查完成。** 用户服务器运行显示PASS；已审阅四个三领域菌的90条路径、17个双领域菌的212条路径符号汇总、其中3个代表菌的67条明细，以及真实方向热图和12条示例路径森林图。详见 [服务器截图结果记录](docs/Step19b_服务器截图结果解读_2026-09-20.md)。本地未读取真实服务器CSV；SDw／COGI／GRADE eugly的定义核查、论文版图表完善及其余菌按论文问题的进一步解释仍待完成。

## 1. 上传到哪里

将整个 `5Step19b_core_taxa_direction` 文件夹上传至服务器：

```text
/home/ec2-user/Desktop/HPP/Data/diet_microbiome_glucose_analysis/
├── 4New_cgm/                       # 原代码和真实结果，保留原位
└── 5Step19b_core_taxa_direction/    # 本文件夹
    ├── run_step19b.sh
    ├── requirements.txt
    ├── scripts/
    ├── config/
    ├── tests/
    └── docs/
```

无需下载或复制参与者级数据，也无需把旧脚本复制进来。本包不连接网络、不导入执行旧脚本。

## 2. 直接运行

先使用之前HPP分析所用的Python环境，然后执行：

```bash
cd /home/ec2-user/Desktop/HPP/Data/diet_microbiome_glucose_analysis/5Step19b_core_taxa_direction
bash run_step19b.sh
```

不需要填写时间戳。默认读取同级 `4New_cgm/outputs`。读取的主要文件见 [来源核查说明](docs/输入来源与解释边界.md)。

脚本先核查，全部通过后才产生方向结果，通常远快于重新跑模型。每次结果写入本文件夹 `outputs/step19b_UTC时间戳/`，日志保存在 `logs/`，不覆盖前一次结果。

**完成后，将终端最后从 `=== STEP19B DIRECTION SYNTHESIS ===` 开始的摘要截图发来。** 如果出现FAIL，截取FAIL摘要即可；不要为通过检查修改871等预期数，也不要删除相反方向结果。

如需重新显示最近一次摘要：

```bash
python3 scripts/show_results.py
```

第一张截图看完后，如需查看某个核心菌的逐路径效应和CI，用摘要里的编号替换下面的 `T023`：

```bash
python3 scripts/show_results.py --core T023
```

`--core all` 显示4个核心菌的全部摘要。使用 `--run outputs/step19b_具体时间戳` 可选定某一次运行。

## 3. 环境

Python 3.8及以上，依赖numpy、pandas和matplotlib。不使用Python 3.9新增的 `list[str]` 等类型注解。只读原分析表，不需要statsmodels、scipy或R。

启动器会先检查依赖。可显式指定此前分析环境的Python：

```bash
PYTHON=/你的环境/bin/python bash run_step19b.sh
```

若服务器确实缺少依赖，可单独建立环境（需要可用的软件源）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON=.venv/bin/python bash run_step19b.sh
```

不要把本地macOS虚拟环境上传到Linux。若仅matplotlib缺失，可以先运行 `bash run_step19b.sh --no-plots` 生成全部表格；此时摘要明确显示图未生成。

## 4. 输入选择规则

每一条路径用 `exposure + outcome_field + species完整taxonomy` 匹配。短菌名仅展示；输出另有T001等唯一编号，防止两个同短名菌被合并。

1. 检查18f候选1490条、最高稳健871条、109种菌和4/17/88结构，以及202/598/71的领域路径数。
2. 检查18f→18g→19a逐路径的身份、效应、N、FDR、p与原有显著性/一致性标签，核对19a领域和菌汇总。
3. 枚举primary/strict/protocol原始模型，逐条与18f保存的对应值比较。**不按修改时间选择最新文件。**
4. 每类仅一个候选匹配时自动采用；零个或多个匹配都会停止并保存候选清单。允许显式选取文件，但不能绕过内容核对。
5. 检查全部模型中的a×b、有限值、CI上下界、N和药物协变量。缺失不填0，标签缺失不当False。
6. 在冻结871条上添加方向，不改变任何原标签。效应匹配容差默认为绝对1e-12、相对1e-8；近零符号阈值为1e-12，参数均记录在manifest。

**旧18f没有保存来源哈希manifest。** 本包可以证明候选与18f保留字段内容兼容，不能独立证明其历史文件身份；新运行将保存输入文件SHA256和实际选择。

遇到多个匹配候选时，先把截图发来。如已能根据原运行记录确定文件，可复制 `config/inputs.example.json` 为 `config/inputs.json`，替换其中REPLACE路径，然后：

```bash
bash run_step19b.sh --inputs config/inputs.json
```

该JSON也可显式指定 `18f_all`、`18f_high`、`18g`、`19a`、`19a_species`、`19a_pan`、`19a_cross`。相对路径相对于JSON所在目录。示例文件包含占位符，不能原样运行。

如果旧项目不在默认位置：

```bash
bash run_step19b.sh --source-root /实际路径/4New_cgm
```

只查看文件存在情况：`bash run_step19b.sh --inventory-only`。

完整核查但不生成方向结果：`bash run_step19b.sh --check-only`。

## 5. 输出怎么看

| 文件 | 用途 |
|---|---|
| `19b_SCREENSHOT_SUMMARY.txt` | 适合截图的成功/失败摘要 |
| `19b_CORE_Txxx_SCREENSHOT.txt` | 每个三领域菌一份：逐路径a/b/间接方向、间接效应、95%CI和N |
| `19b_path_direction_details.csv` | 全部871条，保留三模型全部原字段、效应、CI、N、样本哈希、方向 |
| `19b_species_diet_outcome_directions.csv` | 紧凑逐路径a/b/间接方向及三模型符号稳定性 |
| `19b_species_direction_summary.csv` | 全部109种菌的符号分布，不平均混合方向 |
| `19b_species_diet_domain_summary.csv` | 每菌×饮食×领域的描述性分布 |
| `19b_pan_domain_core_details/summary.csv` | 4个三领域菌明细/汇总 |
| `19b_cross_domain2_details/summary.csv` | 17个双领域菌明细/汇总 |
| `19b_core_direction_heatmap.png` | 4个三领域菌×饮食×结局的主分析间接方向；空白表示不在冻结集 |
| `19b_Txxx_path_directions.png` | 每个三/双领域菌逐路径的9列方向：3模型×a/b/间接效应 |
| `19b_core_indirect_forest.png` | 每个三领域菌×领域的一条示例路径，三模型原始95%CI |
| `19b_representative_paths.csv` | 森林图展示路径，可追溯选择 |
| `19b_direction_synthesis_report.md` | 结果说明与解释边界 |
| `19b_direction_dictionary_used.csv` | 本次解释字典快照 |
| `19b_input_audit.csv` / `19b_input_files.csv` | 核查项目、全部输入路径与哈希 |
| `19b_model_candidates.csv` / `19b_input_differences.csv` | 模型匹配与差异明细；淘汰候选的差异也保留，不代表最终匹配失败 |
| `manifest.json` | 最终状态、选择依据、代码/配置哈希、参数、版本及输出清单 |

图中 `+/-` 是数值方向，不表示健康好坏。a和b没有原始置信区间，因此其符号只代表点估计。森林图按每个菌×领域内exposure/outcome字段字典序首条展示，不称为“最佳机制”；全部路径仍在明细表。

## 6. 解释限制与尚待核实事项

- rEDIH在本地代码链上已确认原始取负、缩尾、能量残差调整、Z标准化和列名传递；本包不重新读取参与者级值做验证。
- cgm_sdw的旧显示名Interday SD与IGLU SDw定义有出入，本地17指标配置也未包含该字段；保留原字段/领域并标记来源待核实。
- COGI、GRADE eugly的具体来源实现仍标待核实，不套用“越低越好”。
- 本次先完成数学方向整合，不输出“有益菌/有害菌”结论。
- 109、4/17/88和路径数是冻结显著结果分布，不证明生物学特异性、统计富集或独立重复。
- 这是横断面中介关联；不将三个重叠队列模型当独立验证，不累加不同单菌模型中介比例。

## 7. 合成测试

```bash
python3 -m unittest discover -s tests -v
```

测试在临时目录创建明确的合成数据，不读取真实模型，不写入4New_cgm。包含使用默认预期计数的1490候选/871保留/109菌完整流程；安装matplotlib时同时检查23张图的生成。实际本地验证见 [本地验证记录](docs/本地验证记录.md)。
