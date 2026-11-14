# Experiment 2 v4 · 纤维 × 菌群交互模型和叠加比较

沿用 v2/v3 的 3,343 人、3,059 个家庭、人均纤维密度、人均 D_Glu 和两套家庭五折。所有留出人员和折内处理后的 D_Glu 都与 V11 原记录逐折核对。研究问题是，在相同人群和相同结局上，哪种纤维 × 菌群交互能在共同基线上增加留出预测，以及加了交互之后，纤维 10→15 g/1,000 kcal 的预测关联在不同人之间有多大差异。 [v2 方法](../v2/README.md) · [v3 三菌评分定义](../v3/README.md)

## 一次运行的路线

| 路线 | 纤维交互采用的特征或拟合方法 |
| --- | --- |
| `gut_shannon` | 肠道 Shannon 线性交互，固定参照 |
| `oral_gut_ecology`、`oral_gut_ecology_spline` | 两部位 Shannon 和优势度，与线性或样条纤维基底交互 |
| `gut_linear_ridge`、`gut_spline_ridge`、`oral_spline_ridge` | 单部位 CLR 主成分和生态特征，比较线性与样条纤维基底 |
| `both_spline_ridge`、`both_smooth_ridge` | 两部位联合，分别用原菌群特征和训练分位点确定的非线性菌群特征 |
| `both_spline_elastic`、`both_spline_spectral` | 两部位联合，分别用弹性网和对弱特征方向加大惩罚的谱收缩 |
| `both_site_shrink`、`both_pc5_spline`、`both_projected_ridge` | 口腔和肠道分配不同罚项、只取每部位前五个主成分、在训练集去除主效应背景后拟合交互 |
| `ecology_hinge_curve`、`oral_gut_tensor_curve` | 允许生态指标两端出现不同斜率，以及在纤维交互中加入口腔与肠道主成分的组合项 |
| `both_huber_curve`、`both_adaptive_group_curve` | Huber 稳健拟合，以及按每个菌群特征的一组交互系数共同调整罚项 |
| `fermentation`、`shannon_fermentation_joint`、`shannon_fermentation_curve`、`fermentation_smooth_curve` | 三菌评分单独交互、与 Shannon 联合交互，以及让修饰随三菌评分非线性变化 |

核心背景有 17 个单模型家族，含共用零修正时有 47 个候选。三菌评分可用时，补充背景扩展到 21 个家族、58 个候选。各候选只通过内层家庭折选取。每个背景还报告以下组合比较。

- **`selector`** 用内层家庭折选一个候选，可以选零修正。
- **`stack_core` / `stack_panel`** 把各家族的非零入选候选按内层留出预测拟合非负权重。总权重不超过 1，允许把所有权重收缩到零。
- **`stack_equal`** 等权叠加非零入选候选。内层预测未超过共同基线时选零修正。
- **`selector_stable` / `stack_stable`** 只用内层合计误差下降、且在多数内折分别下降的候选。分别比较单模型和加权组合是否更稳。三内折时至少需两折下降。

补充背景另算一个只组合核心候选的 `stack_core`。用相同补充基线上的 `stack_panel_minus_stack_core` 衡量把三菌交互候选加入组合池后的额外增量。核心背景的基线含宿主、饮食及口腔和肠道菌群主效应，补充背景的基线还含三菌评分主效应。两种背景都允许年龄和性别与菌群进入同一树路径。每个背景的所有交互路线复用同一个外折已拟合基线。两个背景的 ΔR² 各用自己的基线，直接路线差在相同背景内计算。

每个外折在自己的训练家庭中选择候选和权重。训练修正项的残差来自训练角色内的分家庭交叉拟合。样条、标准化、收缩和特征基底均在相应训练角色拟合，全部留出修正以纤维基底乘菌群特征回放。家庭置信区间按固定的外折预测配对重抽 1,999 次。两轮是同一发现集的重分折。各候选路线的区间是逐项探索性区间。

## 结果怎么看

| 输出 | 要看的结果 |
| --- | --- |
| `EXP2_V4_EFFECT` | 原四个口腔和肠道指标各自的 Q75−Q25 修饰差、四指标校正区间和 10→15 剂量覆盖 |
| `EXP2_V4_JOINT` | Shannon 和三菌评分同时入模时的两项条件修饰差，单位 mg/dL。附家庭区间、标记相关和秩诊断。剂量覆盖逐个指标计算 |
| `EXP2_V4_PREDICT` | 每条路线的留出 `delta_r2`、区间、启用折数和 10→15 额外预测差的标准差与四分位数 |
| `EXP2_V4_HIGHLIGHTS` | 每轮的 Shannon、自动选模、叠加 ΔR² 及关键配对差，便于快速查看 |
| `EXP2_V4_STABILITY` | 同一批人在两套家庭划分中得到的预测增量、交互预测差的相关，以及按两轮较低 ΔR² 排列的前六条路线 |

主读数为核心背景的 `stack_core` 相对共同基线的 ΔR²。再看 `stack_core_minus_gut_shannon` 和 `stack_core_minus_selector`，判断叠加是否优于固定 Shannon 或单路选择。两条路线之间的配对 ΔR² 直接在同一批留出人的平方误差上计算。跨轮相关用于检查同一批人的模型预测是否稳定。稳定性排行读取了两轮评估结果，只用于描述。固定指标和联合模型的 mg/dL 区间用于解读关联，其他路线的预测差四分位数用于描述模型产生的个体差异。若三菌评分无法完整计算，程序继续运行核心路线并输出缺失原因。

## 服务器运行

将整个 `v4` 上传到已运行 v2/v3 的同一服务器目录。入口校验 v4 文件、v2/v3 的固定依赖和现有 V12 解释器环境。预检使用生成数据。

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiment2_Effect_Modification/v4 &&
set -o pipefail &&
bash server_run.sh --preflight &&
bash server_run.sh --threads 2 2>&1 | tee "v4_server_summary_$(date +%Y%m%d_%H%M%S).log"
```

正式入口默认跑完两套家庭五折。优先回传本次生成的 `v4_server_summary_*.log`。文件名包含运行开始时间，便于保留不同运行的结果。如果发截图，可以先截 `EXP2_V4_HIGHLIGHTS`、`EXP2_V4_JOINT` 和 `EXP2_V4_COMPLETE`。稳定性与全路线结果保存在同一日志。输出只包含汇总结果。若中止，回传 `EXP2_V4_STOP`。

本地使用服务器清单中的包版本验收。生成数据通过两轮共十个外折，并回收了已知的相反方向交互。无植入交互与缺少三菌评分的两套生成数据分别通过第一轮。两套检查分别用于核对无植入交互时的输出，以及评分缺失时核心路线仍能完成。
