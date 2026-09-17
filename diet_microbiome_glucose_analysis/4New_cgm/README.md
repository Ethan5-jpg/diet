# 新四个 CGM 结局：独立关联分析

本目录复用原分析的队列与 OLS 模型过程，分析 **7 项饮食暴露→4 项 CGM** 和 **379 个菌群物种→4 项 CGM**。整个文件夹可上传到服务器同一相对位置，运行代码不依赖父目录的旧模型入口或旧关联结果。

**2026-09-15 进展：服务器真实队列分析及绘图已完成。** 本地依据用户返回的结果截图整理了[详细阶段报告](新四项CGM关联分析_阶段报告_2026-09-15.md)，包含主分析、敏感性检查及七图说明。完整模型 CSV 仍保存在服务器；本地 `outputs/` 不放截图转录表或合成测试产物。截图另存于 `服务器图表截图/` 和 `docs/报告依据截图_2026-09-15/`。

## 1. 目录结构

```text
Data/diet_microbiome_glucose_analysis/4New_cgm/
├── run_analysis.sh                 # 一键检查/运行
├── config/analysis.json            # 五个输入路径和运行设置
├── scripts/
│   ├── run_analysis.py             # 日志、队列检查、模型和汇总入口
│   ├── new_cgm_data.py             # 输入审计、合并及排除规则
│   ├── new_cgm_models.py           # 新结局及 FDR、敏感性、同样本比较
│   ├── new_cgm_visuals.py          # 自动统计、七组图、中文总览；支持单独补图
│   ├── legacy_diet_models.py       # 从旧代码原样提取的数值函数
│   └── legacy_microbiome_models.py # 从旧代码原样提取的数值函数
├── tests/test_analysis.py          # 合成数据测试
├── docs/analysis_plan.md           # 分析规则与范围
├── docs/legacy_provenance.json     # 复用函数、原文件和 SHA256
├── requirements.txt
└── outputs/                       # 所有新运行结果
```

上传时以**整个 `4New_cgm` 文件夹**为单位，放到：

```text
/home/ec2-user/Desktop/HPP/Data/diet_microbiome_glucose_analysis/4New_cgm/
```

不要再套一层 `4New_cgm`。不需要上传旧分析的整个目录。测试和说明不参与正式运行，但建议保留在文件夹内，避免代码与说明分散。

## 2. 一键运行

在服务器已使用的 `pheno` 环境执行：

```bash
conda activate pheno
cd /home/ec2-user/Desktop/HPP/Data/diet_microbiome_glucose_analysis/4New_cgm

# 检查环境（新增绘图库 Matplotlib）
python3 -c 'import sys,numpy,pandas,scipy,matplotlib; print(sys.version); print(numpy.__version__,pandas.__version__,scipy.__version__,matplotlib.__version__)'

# 仅在提示缺少 matplotlib 时执行
# python3 -m pip install "matplotlib>=3.6"

# 先核对五份输入、各表人数、协变量和新结局；不拟合模型
bash run_analysis.sh --check-only

# 输入检查无误后，运行本轮全部关联及预设比较
bash run_analysis.sh
```

每次都会新建运行目录并保存日志，终端最后打印 `OUTPUT_DIR` 和 `LOG_FILE`。无需另加 `tee`。若环境解释器命令为 `python`，可用 `PYTHON_BIN=python bash run_analysis.sh`。

正式运行结束会自动汇总并画图，打印 `VISUAL_REPORT` 和 `FIGURE_DIR`。输入检查阶段也会检查绘图库能否导入，避免拟合完才发现缺少依赖。服务器无图形界面也可运行。

`--check-only` 是独立输入检查，也会保存此次合并队列。正式运行会再次检查输入；不用把检查产生的队列表手工移动到别处。

## 3. 默认输入

配置中的相对路径以 **HPP/Data** 为根，由上传后的目录位置推导，不硬编码本地 Mac 路径。

| 输入 | `config/analysis.json` 中的默认路径（相对于 HPP/Data） |
|---|---|
| 已完成的新 CGM 扩展表 | `cgm_deal/outputs/cgm_extension/data/finalize_20260915T074925_139680Z_b375d22f/06_cgm_extended_phenotypes.csv` |
| 原四个饮食评分 | `diet_microbiome_analysis/data/00_current_four_scores_all_participants.csv` |
| 已冻结的新增饮食暴露 | `diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/15b_new_diet_extension_candidate_master.csv` |
| 物种 CLR-Z 矩阵 | `gut_microbiome_deal/data/08_species_clr_zscore.csv` |
| 原协变量主表 | `co-variant/outputs/data/02_covariate_master.csv` |

这些路径沿用原脚本的输入定义。新增饮食候选表是已经准备好的**暴露数据表**，不是旧关联显著结果。不会读取旧 diet→CGM、microbiome→CGM 的模型输出，也不按旧显著性筛人、筛物种或筛饮食指标。

如果服务器文件实际路径不同，仅修改配置文件对应项，允许绝对路径。不要因为文件名相似自动换成别的协变量版本。若 HPP/Data 根目录不同，也可运行：

```bash
bash run_analysis.sh --data-root /实际位置/HPP/Data
```

默认 `expected_species=379`，与旧流程相同。物种数量不符时停止核查，不静默减少检验数。`minimum_model_n=100` 是本轮的统一最小拟合人数保护；不足时记录失败模型，保留计划检验规模。

## 4. 本轮分析变量

### 血糖结局：只分析新四项

| 显示名 | 使用列 | 角色 |
|---|---|---|
| MAGE | `cgm_mage_z` | 核心结局，复用已有值 |
| TAR180 | `cgm_above_180_z` | 核心结局 |
| TBR70 | `cgm_below_70_z` | 核心结局 |
| TIR70_180 | `cgm_in_range_70_180_z` | 辅助结局，仍纳入四结局校正 |

不分析旧 mean、CV、TAR140。不重做 CGM 清洗，也不在饮食、菌群或协变量子集中重新标准化结局；只核验输入 Z 与原 clean 是否一致。

### 饮食暴露：用户确认的全部七项

| 暴露 | 输入列 | 本轮别名 |
|---|---|---|
| AHEI | `AHEI_z` | AHEI |
| AMED | `AMED_z` | AMED |
| hPDI | `hPDI_z` | hPDI |
| rEDIH | `rEDIH_z` | rEDIH |
| EAT13 | `modified_eat_lancet13_z` | EAT13_z |
| NOVA4 | `nova4_total_energy_pct_cov80_z` | NOVA4_z |
| 碳水供能比 | `carbohydrate_energy_pct_z` | Carbohydrate_pct_z |

NOVA4 沿用旧新增暴露候选表的总能量分母及 ≥80% 映射覆盖率定义，不重新评分；总能量协变量为该表的 `mean_daily_energy_kcal`。hPDI 使用原四评分汇总表中的 `hPDI_z`，不静默改用其他修正版评分文件。

## 5. 复用的队列和模型规则

### 队列

- 旧 CGM 固定为一人一条 baseline connection。本轮跨数据表按 participant_id 一对一合并。
- ID 按文本读取以保留前导零，再沿用旧关联脚本的去两侧空白和尾随 `.0` 规范化；规范化后重复或缺失立即停止。
- 饮食分支使用 CGM 与七项饮食表的可用交集；原四评分与新三暴露先外连接，避免要求两类饮食表同时存在。
- 菌群分支使用菌群与 CGM 的交集，不要求有饮食数据。
- 主人群：仅排除明确 `exclude_diabetes_or_a10==1` 者，未知状态保留并计数。
- 严格人群：仅保留明确 `exclude_diabetes_or_a10==0` 者，运行 Model 2/3 敏感性分析。
- 每个饮食×结局及每个菌群结局使用自己所需的有效样本。不要求七项评分或四项 CGM 同时完整。
- 饮食协变量完整标志沿用原 master 的 amed/hpdi Model 2/3 标志；同时核查实际字段完整性。菌群按实际模型字段筛选，与旧菌群流程一致。
- 发现排除标志与明确糖尿病/降糖药阳性矛盾、重复参与者、CGM 与协变量设备冲突或物种矩阵非法值时停止，输出错误原因。

### 模型

| 模型 | 调整因素 |
|---|---|
| Model 0 | 未调整：结局 Z ~ 饮食评分 Z 或单个物种 CLR-Z |
| Model 2 | 年龄、性别、教育、吸烟、睡眠、体力活动、维生素使用、激素使用、CGM 设备 |
| Model 3 | Model 2 + BMI |

- hPDI 的 Model 2/3 额外调整饮酒。
- EAT13、NOVA4、碳水供能比的 Model 2/3 额外调整总能量，不添加 hPDI 专属饮酒项。
- 菌群 Model 2/3 不加入饮食专属饮酒或总能量项。
- 复用原 OLS、常规标准误、t 检验和 95% CI；连续协变量在该模型所选样本中按旧代码 `ddof=0` 编码。评分、物种和 CGM 使用输入中的 Z，不重新计算。
- 饮食编码及菌群编码保持各自原实现，包括设备参考组。旧菌群设计若只有一种设备或设计矩阵不满秩，会记录该次模型失败，不自动更换协变量设置。

### 同样本比较

除主人群 Model 0/2/3 外，还拟合：

1. Model 0 在 Model 2 的完全相同参与者上重拟合：区分样本筛选变化与协变量调整变化。
2. Model 2 在 Model 3 的完全相同参与者上重拟合：区分 BMI 缺失筛选变化与 BMI 调整变化。

比较同时核验参与者集合 SHA256，而不只检查 N 相同。`analysis_set` 分别为 `primary`、`strict`、`same_m2`、`same_m3`。

## 6. FDR 范围

每个分析人群、模型/比较分别计算，不把 Model 0/2/3 混成一个检验家族。

| 分支 | `FDR_family` | `FDR_global` |
|---|---|---|
| 原四评分 | 4×4=16 次 | 同次模型全部 7×4=28 次 |
| EAT13、NOVA4 | 2×4=8 次 | 同上 28 次 |
| 碳水供能比 | 1×4=4 次，探索家族 | 同上 28 次 |
| 菌群 | 每个 CGM 结局 379 次 | 4×379=1,516 次 |

保持原流程“原评分、新增主要暴露、探索暴露分别校正；菌群按结局校正并报告全局校正”的结构，将原来的三结局检验数更新为四结局。

失败或无法估计的模型仍保留一行，p/q 为空。在 BH 内部用 p=1 占位，保留计划家族大小；占位值不会作为有效 p 值写出。避免某些模型失败后悄悄缩小多重检验规模。TIR 的辅助角色不会令它从预设四结局校正中移除。

## 7. 输出与阅读顺序

```text
outputs/run_<UTC时间>_<后缀>/
├── data/
│   ├── diet_cgm_cohort.csv
│   ├── microbiome_cgm_cohort.csv
│   ├── diet_model_membership.csv
│   └── microbiome_model_membership.csv
├── models/
│   ├── diet_cgm_all_models.csv
│   └── microbiome_cgm_all_models.csv
├── reports/
│   ├── manifest.json
│   ├── cgm_input_audit.csv
│   ├── cohort_summary.csv
│   ├── variable_coverage.csv
│   ├── model_summary.csv
│   ├── model_failures.csv
│   ├── diet_all_coefficients.csv
│   ├── diet_design_parameters.csv
│   ├── microbiome_design_parameters.csv
│   ├── diet_same_cohort_comparisons.csv
│   ├── microbiome_same_cohort_comparisons.csv
│   └── 结果摘要.md
├── visuals/visual_<时间>_<后缀>/
│   ├── 结果总览.html               # 中文总览，打开即可查看统计及全部图
│   ├── 图表说明.md
│   ├── visualization_manifest.json
│   ├── figures/                    # 七组图，各有 PNG / PDF / SVG
│   └── tables/                     # 直观统计、绘图数据、菌种选择清单
└── logs/run.log
```

1. 先看终端或 `run.log`：输入人数、缺失、排除、每个模型 N、失败和 FDR 计数。
2. 再看 `model_summary.csv`、`model_failures.csv`，确认模型完成情况。
3. 打开 `visuals/visual_.../结果总览.html` 查看直观统计与图表；再阅读 `结果摘要.md` 和两张完整模型表。用 `analysis_set=primary, model=2` 查看主调整结果，Model 3 查看额外 BMI 调整。
4. 用 same-cohort 表区分样本变化与调整效应；用 strict 结果检查未知排除状态的影响。

饮食结果 `beta` 表示饮食评分增加一个输入 Z 单位对应的 CGM Z 变化；菌群结果表示物种 CLR-Z 增加一个输入 Z 单位对应的 CGM Z 变化。正负方向按原始指标含义解释，不统一转换“越高越健康”的方向。

默认所有步骤均开启，饮食结果计划 196 行（28 次×7 组模型/比较），菌群计划 10,612 行（1,516 次×7 组）。这是计划检验数，不是参与者人数或显著数。`--check-only` 不会生成模型结果。

状态：

- `inputs_checked_no_models_fitted`，退出码 0：仅输入检查完成。
- `completed`，退出码 0：计划模型均完成。
- `completed_with_model_failures`，退出码 2：保存了结果，但部分模型不能估计，须查看失败表。
- `completed_with_visualization_failure`，退出码 3：模型结果已保存，但绘图失败；查日志及失败模型计数后，可单独补图。
- `failed`，退出码 1：输入或运行错误；该次产物不能当作已完成分析使用。

清单记录输入与代码 SHA256、配置、软件版本、人数、状态；运行末尾复查源文件未变化。旧分析文件和旧结果不写入。

### 自动生成哪些统计与图？

- **直观统计表**：按饮食/菌群、模型、四个结局列出计划检验数、成功/失败数、样本量范围、FDR 显著正/负关联数及全局校正显著数。
- **饮食热图（1 组）**：七项饮食 × 四项 CGM，并排显示 Model 0/2/3。
- **饮食森林图（1 组）**：Model 2/3 的效应值和 95% 置信区间。
- **显著数量图（1 组）**：各模型、各结局正负关联的数量。
- **菌群火山图（2 组）**：Model 2 和 Model 3，各包含四个结局；纵轴是 −log10(家族 FDR)。
- **代表菌种热图（2 组）**：Model 2 和 Model 3，各展示最多 20 个在至少一个结局显著的菌种，按最小家族 FDR 排序。无显著菌种时显示探索性排序并明确标注；完整菌名和选择规则保存在 CSV。

合计 **7 组图、21 个图片文件**。PNG 为 300 dpi，PDF/SVG 可用于后续排版。图中文字为英文，避免服务器缺少中文字体；总览及说明为中文。热图中的星号表示家族 FDR < 0.05，加号表示全局 FDR < 0.05；不可估计的结果显示灰色 NA，不当作零效应。

下载查看时请保留整个 `visual_...` 文件夹，HTML 通过相对路径读取其中的图片。本次运行的显著数量和结论见阶段报告；后续新运行应以该次服务器结果为准。

### 已经跑完模型，只想补图或重新画图

在 `4New_cgm` 目录执行，把路径末尾替换为实际的运行文件夹：

```bash
python3 scripts/new_cgm_visuals.py --run-dir "/home/ec2-user/Desktop/HPP/Data/diet_microbiome_glucose_analysis/4New_cgm/outputs/run_实际运行目录"
```

该命令只读取保存的两张完整模型表，不重新拟合、不重算 FDR、不覆盖已有图表；每次新建一个 `visual_...` 子目录。`--check-only` 的运行没有模型表，不能用于补图。

## 8. 本轮结果的解释边界

用户要求先复用旧分析，因此当前保留原 OLS 和常规标准误。TAR180 的 75.24% 零值集中、三项比例偏态和极端值没有因 Z 标准化而消失；本轮首轮结果不等于已经验证模型适用性，也不作因果或临床解释。后续稳健性、变换或其他模型应另行预设，不能只根据显著与否选模型。

本次入口自动绘制独立关联阶段的统计图。饮食→菌群重算、显著路径交集、桥接候选筛选、中介及其图表属于下游步骤，等新结局独立关联结果出来后再处理。

## 9. 测试

测试使用独立临时目录及合成数据，不向正式 `outputs/` 写入假队列结果：

```bash
python3 -m unittest discover -s tests -v
```

原数值函数及来源指纹见 `docs/legacy_provenance.json`；测试验证七项暴露/四结局、FDR 数量、按结局缺失、人群排除、BMI/饮酒/能量设置、直接 OLS 数值一致性、同样本 ID、失败检验占位及模拟上传后的独立运行。

2026-09-15 本地验证：**21 项测试全部通过**，包含 379 物种规模及 1,516 次全局校正。测试环境为 Python 3.12.14、pandas 2.2.3、NumPy 2.3.5、SciPy 1.18.1、Matplotlib 3.11.2；shell 语法、Python 3.8 语法解析和 JSON 检查通过。服务器沿用原 `pheno` 环境，真实数据运行已由用户返回的完成截图确认；完整模型表未下载到本地。原脚本文件的来源指纹也通过一致性检查，原分析代码未改动。

绘图扩展验证：自动生成七组图的 PNG/PDF/SVG、HTML 内相对链接有效、显著正负计数正确、失败与零效应区分、无显著菌种探索标记、单独补图不改变模型表。已目视检查合成数据生成的热图、森林图、数量图及火山图；合成图仅保存在项目外测试目录，不作为真实队列结果。

火山图显示修正：将各面板的样本量/显著数量说明移到绘图区外，避免遮住右上角的显著点。已用含右上角极显著点的合成示例验证；模型、p 值及 FDR 不变。服务器需更新 scripts/new_cgm_visuals.py 后通过单独补图命令生成新版图。
