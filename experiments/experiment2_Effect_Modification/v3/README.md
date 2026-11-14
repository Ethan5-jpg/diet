# Experiment 2 v3 · 纤维发酵相关菌群评分与肠道 Shannon

研究问题是，同样比较纤维密度 10→15 g/1,000 kcal，哪个肠道菌群指标对应更明显的人均 D_Glu 关联差。V3 沿用 v2 的 3,343 人、3,059 个家庭、V11 原人均结局及两套家庭五折。两轮使用同一发现集，r0 为主分析，r1 为换折检查。

## 固定评分

选用 *Ruminococcus bromii*、*Bifidobacterium adolescentis* 和 *Eubacterium rectale*（也接受 *Agathobacter rectalis*）。人体抗性淀粉干预观察到前两种菌与底物降解相关，*E. rectale* 与丁酸产出较高的人群相关。选菌依据来自干预和培养研究。三菌总丰度及下列变换为本轮事先固定的 HPP 评分定义。 [Venkataraman 等，2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4928258/) · [Baxter 等，2019](https://pubmed.ncbi.nlm.nih.gov/30696735/) · [Ze 等，2012](https://academic.oup.com/ismej/article/6/8/1535/7587723) · [物种同义名](https://lpsn.dsmz.de/species/eubacterium-rectale)

```text
score = log(1 + (relative_abundance_Rb + relative_abundance_Ba + relative_abundance_Er) / 0.0001)
```

相对丰度以 0–1 的比例输入。0.0001 是固定的丰度尺度（0.01%）。评分使用冻结的 MetaPhlAn4 物种表，允许完整分类路径末端的物种标签与裸物种名。每种菌必须唯一匹配。保留归档中的零丰度，单个样本没有检出目标菌时评分为零。缺列、重复映射或冻结样本存在无法评分的记录时停止并报告。评分只反映三种菌的相对丰度，人均总纤维暴露也没有区分抗性淀粉与其他纤维。HPP 中实际的发酵活性和丁酸产量需要另外测量。

## 配对比较

- **关联量。** 共同背景包含 V11 饮食与宿主特征、既有菌群特征和两个菌群指标的主效应。两个模型各加入一项线性纤维交互，分别报告各自 Q75−Q25 在 10→15 对应的修饰差（mg/dL）、家庭聚类区间及尾部剂量支持。用同一批家庭的影响向量估计有符号差 `difference`。另报 `absolute_difference`，计算 |发酵评分修饰差| − |Shannon 修饰差|，区间由两指标同时区间构造。
- **预测增量。** 每折用一个共享基线。年龄和性别允许与菌群共同进入树路径，固定年龄和性别时保留饮食与菌群的分块约束。基线含 Shannon 和新评分主效应。两个模型各自在这一已拟合基线上添加对应交互，使用相同的内折罚项网格和零交互候选。输出各自的留出 ΔR²，并直接用相同家庭的误差计算两条路线的配对差及区间。预测区间条件于已拟合外折预测。
- **判断标准。** 绝对修饰差的配对比较和预测 ΔR² 的直接差分别回答“关联差更大”和“留出预测更准”。如果区间覆盖零，报告当前证据不足。两指标的 Q75−Q25 是各自尺度上的典型跨度。主分析是观察性发现集估计。

## 服务器运行

上传整个 `v3` 到服务器交接包的 `experiment2_Effect_Modification` 下，与已更新的 `v2` 并列。v3 复用 v2 的两个已固定源码文件和 V12 的已安装解释器。服务器端只需执行下面的命令。

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiment2_Effect_Modification/v3 &&
bash server_run.sh --preflight &&
bash server_run.sh --availability &&
bash server_run.sh --threads 2
```

`--preflight` 校验文件、依赖版本并运行生成数据检查。`--availability` 只读已登记的物种字典和丰度矩阵，报告归档发现集中目标菌的列名、检出人数与评分分布。正式运行的 `EXP2_V3_PANEL` 报告冻结评估人群的对应汇总。回传两轮 `EXP2_V3_EFFECT`、`EXP2_V3_PREDICT` 及 `EXP2_V3_COMPLETE`。若中止，回传 `EXP2_V3_STOP`。

本地验收使用与服务器清单相同的六个 Python 包版本。用生成数据重建交接目录，完成两轮共十个外折，核对冻结行、结局、配对区间和已植入交互的恢复。隐藏生成的原始结局文件后，可用性检查仍能运行。HPP 冻结物种字典及真实样本覆盖由服务器输出确认。
