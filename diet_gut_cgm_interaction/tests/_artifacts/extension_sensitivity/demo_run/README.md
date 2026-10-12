# 整体饮食 × 菌群 × CGM 交互探索

新模块位置：`Data/diet_gut_cgm_interaction/`。分析单位是参与者，每人一条整体CGM记录；使用既有七项饮食、四个候选菌和整体Mean/CV/SDw。无需餐次数据。

这是2026-10-10冻结的第一轮探索。所有四菌来自既有全项目筛选；本模块完成不等于获得独立验证或因果机制。

## 服务器结果记录

本部分的截图转录、结果解释和后续核对统一保存在 [research_records](research_records/README.md)。当前批次见 [2026 年 10 月 10 日服务器结果](research_records/2026-10-10_服务器结果/README.md)。

## 扩展敏感性补充运行

新增入口 `run_extension_sensitivity.sh` 为全部 60 项扩展补齐六组敏感性。为沿用完整 84 项校正，每个设定均拟合七饮食的全部 84 项，共 588 次拟合；扩展占 420 行。全部未显著和失败项保留，不只追踪原 19 项阳性。原第一轮入口和输出仍然保留。

把更新后的本目录**合并上传**到服务器同名目录，保留服务器原有 `outputs/`。补充运行需要新增的两个 `run_extension_sensitivity` 文件、`interaction/extension_sensitivity.py`、`interaction/extension_reporting.py` 以及原有运行文件；README 和 EXTENSION_METHODS 一并上传。无需上传本地 `tests/`、`research_records/`、`outputs_extension_sensitivity/` 或缓存。

上传后，在服务器终端整段运行：

```bash
conda activate pheno && \
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction && \
PYTHON=python bash run_extension_sensitivity.sh \
  --reference-run outputs/run_20261010T085601_096554Z_4d43e0
```

程序会核对旧运行、核心代码和输入文件指纹，重算后检查原主分析及前五组重点敏感性共 204 项能否复现。共同样本改为七饮食全部完整，明确标记 `common_all7_sample`，不冒充旧两饮食共同样本。若输入或模型复现不一致，程序停止并记录原因。

每组 `q_all84` 都按完整 84 项校正；原 19 项仅用于稳定性追踪。结束时打印 `EXTENSION 1/3`、`2/3`、`3/3`，把这三屏发回即可。完整 β、区间、q、条件斜率、样本流和全部扩展森林图保存在新的 `outputs_extension_sensitivity/` 运行目录。

补充分析方法与解释见 [EXTENSION_METHODS.md](EXTENSION_METHODS.md)。这是看过主结果之后开展的同队列敏感性，不能称独立验证。**本地代码与合成测试不代表服务器结果；补充真实结果仍待运行回传。**

本地软件演示可以运行 `bash run_extension_sensitivity.sh --demo`，始终标记 SYNTHETIC，不读取真实数据。旧 `show_results.py` 的默认指针仍指原分析；查看补充结果时须用 `--run-dir` 指定新的 RESULT_DIR。

## 上传与运行

把整个`diet_gut_cgm_interaction`目录上传到服务器HPP的`Data`下面。代码默认把自身上一级目录作为Data根目录，与服务器用户名、HPP所在绝对路径无关。不要单独上传一个Python脚本。

在服务器的新目录打开终端，运行：

```bash
bash run_analysis.sh
```

运行默认完成输入核查、84项基础模型、重点24项的六组敏感性、汇总表和图。末尾直接打印三屏摘要，并显示`RESULT_DIR`。运行前不需要把结果或参与者数据下载到本地。

若要先检查文件路径（不需要科学计算依赖）：

```bash
python3 run_analysis.py --list-inputs
```

若要先核查文件内容、人数和完整病例，不拟合模型：

```bash
bash run_analysis.sh --check-only
```

可从任何工作目录调用脚本；若数据不在代码上一级目录，显式指定实际Data目录：

```bash
bash /实际代码位置/diet_gut_cgm_interaction/run_analysis.sh --data-root /实际数据位置/Data
```

切换Python解释器：

```bash
PYTHON=/实际环境/bin/python bash run_analysis.sh
```

所需环境为Python 3.8+及NumPy、pandas、SciPy、Matplotlib。缺包时脚本会报出安装命令；代码不会自动联网安装。可在服务器允许的环境中执行：

```bash
python3 -m pip install -r requirements.txt
```

## 输入来自哪里

完整路径在`config/analysis.json`，均默认相对Data。

| 内容 | 来源 |
|---|---|
| AHEI、AMED、rEDIH | 原`00_current4_diet_cgm_cohort.csv`及原脚本的列名优先规则 |
| hPDI | `hpdi_correction_audit/corrected_scores/hpdi_participant_scores.csv`中的`hpdi_score_energy_adjusted_z` |
| EAT13、NOVA4、碳水供能比 | `15d0_new_diet_cgm_cohort.csv`；与`15b_new_diet_extension_candidate_master.csv`逐人核对 |
| NOVA4正式定义 | `nova4_total_energy_pct_cov80_z`，总能量分母、分类能量覆盖至少80% |
| 菌群 | `gut_microbiome_deal/data/08_species_clr_zscore.csv`，精确匹配四菌，保留原CLR-Z |
| 协变量 | `co-variant/outputs/data/02_covariate_master.csv` |
| 整体CGM | 已有餐次/整体衔接模块所指向的`cgm论文新增17指标/outputs/run_20260919T030229_482714Z_c1a0f480/data/07_cgm_paper_extended_phenotypes.csv` |
| 药物敏感性 | 原`15f2_protocol_abxppi_cohort.csv`中的protocol窗口抗生素与PPI标记 |

**CGM路径固定到已有已知版本，不按文件修改时间自动挑选latest。** 如果你服务器上使用另一个经核查的最终CGM版本，只修改配置中`inputs.cgm`的相对路径。其他输入发生目录变动时同样修改相应条目，不需要改Python代码。显式绝对路径也支持，但迁移时更推荐相对Data的路径。

输入检查包括：ID保留为字符串、前导零保留、重复ID、基线访视范围、必需字段、非数值/无穷值、CGM clean/Z对应关系、设备冲突、排除标记矛盾。每个输入文件记录SHA256，结束时再次确认未改变。

protocol默认`if_available`：文件缺失时标记相应48项为SKIPPED，主分析仍运行，状态明确为有局限。将配置改为`required`可要求缺失就终止；`disabled`表示主动不运行这组敏感性。不存在“缺失药物标记自动当作未用药”。

## 冻结的问题和模型

重点24项：`hPDI/NOVA4 × T023/T046/T068/T096 × Mean/CV/SDw`。

扩展60项：`AHEI/AMED/rEDIH/EAT13/Carbohydrate_pct × 同4菌 × 同3结局`。

四菌为GGB6012_SGB8640、Clostridium phoceensis、Eubacterium rectale、Ruthenibacterium lactatiformans。整体模型为`Y = Diet + Microbe + Diet×Microbe + 原Model3协变量`。没有把中介a×b的符号直接当成交互。

主样本继续排除明确糖尿病/A10阳性，允许原标记未知；strict只保留明确阴性。缺少整个协变量记录的人始终不进入模型。每个组合采用完整病例，每项输出真实N与样本集合哈希。

重点24项另执行：strict；protocol相同人未调整；protocol相同人加抗生素/PPI；加Diet²/Microbe²；加年龄/性别/BMI与Diet/Microbe的交互；两项重点饮食、四菌和三结局的共同完整样本。共84+24×6=228项预定拟合。这里没有228个独立发现或独立人群。

## 怎样看结果

- `SCREEN_1.txt`：人群、状态和总体结果。
- `SCREEN_2.txt`：全部24项重点结果，无论阳性或阴性。
- `SCREEN_3.txt`：敏感性、共同支持和诊断。
- `tables/all_interactions.csv`：全部228项，包括不可估计原因、N、交互、区间、p/q、原单位换算、双IQR对比、诊断。
- `tables/conditional_diet_slopes.csv`：菌Q25/Q50/Q75下饮食的条件斜率；不是把人群硬分三组。
- `tables/cohort_flow.csv`：逐模型完整病例与缺失流失。
- `tables/fdr_families.csv`：每个校正家族的预定分母、实际可估计数。
- `figures/`：重点结果的PNG/PDF森林图和全部24项条件斜率面板。
- `manifest.json`：数据、配置、代码哈希、软件版本、状态与限制。
- `aggregate_results.zip`：全部汇总产出，不包含参与者矩阵、ID列表或逐人拟合值。

再次查看最近成功的真实输入运行：

```bash
python3 show_results.py --screen 1
python3 show_results.py --screen 2
python3 show_results.py --screen 3
```

三屏摘要足够第一轮截图反馈，详细结果保留在服务器。若需要读旧运行，用`--run-dir outputs/对应运行目录`。真实和合成运行的指针分开，默认不会误读合成结果。

## 状态与解释

`INPUT_CHECK_PASSED`仅表示接口检查通过；`COMPLETE`表示预定拟合和所请求输出完成；`COMPLETE_WITH_LIMITATIONS`表示存在跳过、不可估计或有限共同支持；`NO_ESTIMABLE_FOCUS_MODELS`及`FAILED`需先看原因，退出码为2。输入变化、错误列、重复ID或真实程序错误不会被伪装成阴性结果。

`q_focus`是每个模型设定内固定24项重点校正，主分析`q_all`是完整84项校正。敏感性只包含24项，故该设定的两种q相同。不可估计项以p=1保留在校正分母，其自身q留空。95%CI是逐项区间，不是同时区间。

负交互只代表斜率差，不能直接写“该菌保护”或“这种饮食有益”。CV原单位是百分点，Mean/SDw是mg/dL。更完整的模型和解释见`METHODS.md`。

2026-10-10复核修订（版本.2）：零四分位距不再误删可估计模型，改为单独标记双IQR对比不可用；糖尿病/A10原始标记存在时核对完整三态逻辑；全部模型不可估计或仅核查输入时也保留可读取的结果表头。

## 本地演示

```bash
bash run_analysis.sh --demo
```

合成输入在临时目录生成并在结束后删除，输出始终标记`SYNTHETIC`，manifest记录随机种子与注入的交互。它只验证软件，不能用于论文结果。上传不需要`tests/`、`outputs/`或`__pycache__/`；运行必需的是根目录脚本、`interaction/`、`config/`、README、METHODS和requirements。
