# 日级个人内饮食—菌群—全天CGM分析

当前上传版本：**2026-10-11.3**（修复CGM连接筛选后空数据块的时间转换错误）。

**真实运行更新（2026-10-11 19:45–19:46）：**[132/132项拟合及重点结果核对均已完成](research_records/2026-10-11_1945_日级重点结果核对/README.md)。主分析NOVA4×T096的SD、Mean正交互通过相应校正，对酒精处理较稳定，但并非全部敏感性保留显著性；营养背景设定中SD估计减弱，同时样本缩小。逐项结果、斜率区间和日期核对已归档，本轮可以收尾。用户确认各测量属于同一时间窗，窗口内采样先后不作为本模块的主要限制。保留所有服务器输出，目前无需再次上传或运行；以下原始构建命令保留作溯源。

这是研究第三步的独立运行包：**同一个人某天比自己平时摄入更多NOVA4或纤维时，当天整体血糖是否不同；这种个人内关联是否随T068/T096的原CLR-Z变化？**

原中介与整体交互结果保留在上级模块。本包不依赖上级Python代码或旧餐次运行目录，但读取项目已有原始数据、NOVA食物映射、菌群/协变量及最终CGM连接表。本地只有示例和截图记录，真实队列是否满足新日级要求，须由服务器运行确认。

## 上传与完整运行命令

将**整个 `daily_within_person` 文件夹**上传到：

```text
/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person/
```

在服务器执行：

```bash
conda activate pheno &&
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person &&
python -c 'from daily import VERSION; print("CODE_VERSION=" + VERSION); assert VERSION == "2026-10-11.3", "Please upload version 2026-10-11.3"' &&
PYTHON=python bash run_analysis.sh \
  --data-root /home/ec2-user/Desktop/HPP/Data \
  --stage all
```

路径和运行脚本均支持从任意工作目录启动。默认Data根是本文件夹上两级，Linux和macOS的前缀不写死在代码里；命令中的`--data-root`可以明确覆盖。具体输入相对路径集中在 `config/analysis.json`。**不自动挑“最新”旧输出**，最终CGM表沿用已经核实的固定运行路径；若服务器该表位置不同，应修改配置中的`inputs.connections`为真实路径。

程序会先验证输入，再分块读取、生成日级面板、检验可行性和运行全部预设模型。末尾只显示三屏汇总和`RESULT_DIR=...`，将这三屏截图发回即可。中间DIET/CGM/DAILY/MODELS信息仅为进度。

若此前遇到`Cannot cast DatetimeArray to dtype float64`：重新上传本文件夹的新版同名代码，保留服务器已有`outputs/`。直接按上面的`--stage all`新建运行；失败的部分CGM缓存不能作为`BUILD_COMPLETE`续算，不需要删除旧失败目录。原始截图、报错复现和修复记录见[13:30失败记录](research_records/2026-10-11_133052_CGM读取失败/README.md)。

不自动下载、传输或压缩文件。`private/`中的个人和每日记录、SQLite缓存全部留在服务器。输出按新运行目录保存，不覆盖原始文件或既有分析结果。程序不自动安装依赖；若明确提示缺包，在当前环境执行：

```bash
python -m pip install -r requirements.txt
```

## 输入与连接规则

所有路径相对于Data目录：

| 输入 | 路径/作用 |
|---|---|
| 食物条目 | `Transfer/diet_logging/diet_logging_events.csv`，含身份、时间、食物ID和营养字段 |
| 原始CGM | `Transfer/cgm/timeseries/*.csv`，每个文件必须含参与者、cohort、research_stage、connection_id、collection_timestamp、glucose |
| NOVA映射 | `diet_deal/outputs/03_score_mapping/new_diet_indicators/new_diet_indicator_food_id_mapping.csv` |
| 菌群 | `gut_microbiome_deal/data/08_species_clr_zscore.csv`，精确匹配T068/T096物种列 |
| 协变量 | `co-variant/outputs/data/02_covariate_master.csv`，核对cohort/访视和糖尿病/A10三态排除标记 |
| 唯一CGM连接 | `cgm_deal/cgm论文新增17指标/outputs/run_20260919T030229_482714Z_c1a0f480/data/07_cgm_paper_extended_phenotypes.csv` |
| 可选菌群日期 | `Transfer/gut_microbiome/gut_microbiome.csv`，身份不足或多次取样无法区分时记录未知 |

不从文件名猜参与者，不只按日期合并，不把不同连接拼成一个传感器。限定10k、00_00_visit；协变量中的连接/设备信息若与最终CGM表冲突，程序停止。缺少必要身份列的本地简化CSV不能替代服务器完整数据。

## 主要分析和质量要求

- NOVA4：每天NOVA4能量/已分类NOVA能量×100，要求分类覆盖至少80%；单位为10个百分点。另保存以总能量为分母的描述值，不混用两个分母。
- 纤维：每日g/1000 kcal，单位为5 g/1000 kcal。任何正能量条目纤维缺失时，该日纤维总量不参与分析；不以零填充。
- CGM：固定Asia/Jerusalem当地日，按15分钟格的有效观测中位数计算日Mean、样本SD和CV。没有插值；夏令时按真实23/24/25小时日计算分母。
- CGM主要求：覆盖≥90%、最大有效点间隔含两端≤60分钟、四个等时长区段各覆盖≥75%。首尾不完整日同样应用这些规则，不假设每人都应有14个完整日。
- 饮食主要求：能量完整且500–5000 kcal、至少3次相距规则下的有能量进食、记录跨度≥8小时、无可疑完全重复条目/日期冲突。这是可观察的记录质量代理，不能认证当天没有漏餐；人群因此限于这类记录日。
- 每个模型至少200人、每人至少5个完整日，个人内NOVA4范围≥10个百分点或纤维密度范围≥2 g/1000 kcal。先按模型字段完整性筛日，再计算天数与个人内变化。
- 4项主要检验：2种饮食×2菌×日SD。全部12项交互（再含Mean、CV）另做BH校正。主分析加6组敏感性，总计84个拟合；不可估计项仍保留在预设校正分母中。

主模型是个人固定效应、参与者聚类CR1标准误。控制当日能量、酒精、星期和观察日，并允许这些变量的关联随菌群不同。年龄、性别等恒定特征由个人固定效应吸收；基线因素是否对应不同饮食斜率另有敏感性检验。细节见 [METHODS.md](METHODS.md)。

六组敏感性分别为明确无糖尿病/A10、严格CGM/NOVA覆盖、至少7天、营养背景、基线因素的饮食斜率差异、每人等权。重叠样本不叫独立复现；CV和SD也不是两个独立的波动验证。

## 三屏和状态怎么读

1. 数据与可行性：连接人数、匹配日、各主要组合实际人数/天数，以及质量合格日中的饮食、酒精和菌群缺失日数（原因可重叠）。
2. 全部12项主分析：原单位交互系数、逐项95%区间、qSD4、qAll12，包括阴性及不可估计行。
3. 六组敏感性摘要、主要日SD的Q25/Q75饮食斜率、时间和支持限制。条件斜率完整区间留在表中。

| 状态 | 含义 |
|---|---|
| `PREFLIGHT_PASSED` | 仅文件头检查通过，不代表日级数据或模型通过 |
| `BUILD_COMPLETE` | 日级构建完成，尚未运行模型 |
| `COMPLETE` | 84项估计完成且未触发既定支持提示，不代表有显著结果 |
| `COMPLETE_WITH_LIMITATIONS` | 计算完成，但有不可估计模型或支持提示；逐行保留 |
| `NO_ESTIMABLE_PRIMARY_MODELS` | 4项主要模型均未估计；仍保存可行性、失败原因与预设检验行，退出码2 |
| `FAILED` / `INTERRUPTED` / `RUNNING*` | 结果未完成；不要读取残留系数作为结论 |

`*`提示菌群尾部人数不足、菌群IQR为零，或单人占去均值后交互列平方和超过10%；这是设计支持诊断，不是证明因果可识别。仅有方向一致而未通过校正，不写成显著。

第一屏的`QCdays`是质量/资格合格但尚未要求模型字段完整的日数；`MissDiet`、`MissAlc`、`MissGut`分别是其中的暴露、酒精和菌群缺失日数。先因字段缺失筛日，再因不足5天或个人内变化不足筛人，最终损失人数可能更大。本地少量食物样例已有酒精/纤维缺失，因此不能保证真实服务器最终N达到200；程序会如实报告，不自动填零或放宽门槛。没有同人同日交集也保留完整的零匹配报告。

## 可选：分阶段运行

只核对依赖和文件头：

```bash
PYTHON=python bash run_analysis.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage check
```

只构建日级面板并查看可行性：

```bash
PYTHON=python bash run_analysis.sh --data-root /home/ec2-user/Desktop/HPP/Data --stage build
```

然后把下面路径替换为该次打印的完整`RESULT_DIR`，运行模型：

```bash
python run_analysis.py --stage analyze --run-dir '/这里替换为BUILD_COMPLETE的完整RESULT_DIR'
```

续跑前会核对代码、配置、解释器/依赖版本、原始输入和已构建面板指纹。不能混用不同批次。修改输入、参数或代码后应新建运行；中断的缓存不做部分续算。

之后重看某一屏：

```bash
python show_results.py --run-dir '/这里替换为已完成的完整RESULT_DIR' --screen 2
```

`show_results.py`核验已保存结果指纹，不重读数GB原始CGM。不需要手工寻找或复制CSV回本地。

## 本地合成验证

```bash
PYTHON=python bash run_analysis.sh --demo --stage all
python -m pytest tests -q
```

`--demo`仅生成虚构输入，输出每屏明确显示SYNTHETIC；合成演示把人数门槛降为30，**真实运行仍为200**。数值核对测试另外需要pytest和statsmodels，它们不是服务器分析依赖。验证记录见 [docs/LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md)。

所有真实截图与解释保存在本模块的 [research_records/](research_records/README.md)。真实日级修订分析及重点核对已完成；结果属于同次访视观察性证据，不能证明饮食或菌群的因果作用。
