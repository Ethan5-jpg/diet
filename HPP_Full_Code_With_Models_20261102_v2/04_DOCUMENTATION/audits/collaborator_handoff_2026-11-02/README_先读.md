# HPP 合作者：数据与预处理工作目录

2026-11-02交接。按Haochen最新要求，本包仅含数据、预处理、读取示例和核验；不含预测模型、模型结果、预测值、已学菌群评分或评分交互。合作者在experiments内开展自己的建模。

## 研究背景与目标

研究相同饮食水平下，口腔/肠道菌群特征不同的人，其饮食—代谢关联是否不同。可视为连续暴露与两个高维组成数据视图的交互学习；没有真实个体异质性标签。观测关联不直接等于个体因果获益。

人级范围：纤维密度、UPF × 九结局（CGM血糖、总脂、VAT、六肝超声指标：弹性/速度/黏度/衰减/频散/声速）；各结局独立准入。日级为纤维—CGM。可先从纤维×D_Glu接口开始。

## 开始工作

服务器项目根：`/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021`。

本目录：项目根下 `collaborator_handoff_20261102_v1/`。

```text
README_先读.md                    本说明
DATA_DICTIONARY.md                字段与行对齐
inputs/discovery_v2/               人级发现输入、双部位丰度、九结局
inputs/daily/                     发现组CGM/food日表和无评分共同日表
preprocessing/folds/fold0–fold4/   折内CLR/PC/饮食基函数与转换参数
preprocessing/folds/full/         全发现参考转换
preprocessing/code/               纯菌群/表型预处理转换器
examples/load_fiber_fold.py        读取和预处理示例，无模型
examples/check_package.py          包哈希核验
manifests/                       来源/哈希/schema/环境/实际核验
experiments/                     合作者新实验工作区
```

在本目录执行：

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python examples/check_package.py
/home/ec2-user/miniconda/envs/pheno/bin/python examples/load_fiber_fold.py
```

两条命令只核对文件、读取和预处理，不训练预测模型。示例load_fold(0)返回主索引、原协变量、双部位CLR/PC、饮食基函数、折内处理Y和训练/验证掩码，可自行接模型。所有数据仅在TRE内使用；本包独立复制，非链接，原文件与权限不变。新实验在experiments下独立命名，不覆盖输入。

## 队列与划分

候选池6305人/5699家庭或同住单位；当前70/30：发现4427人/3989家庭，确认保留1878人/1710家庭。完整角色表只有角色元数据，无确认Y。发现固定单层家庭五折cv_fold=0–4，家庭键split_unit_id。

纤维×D_Glu实际CV3343人/3059家庭，各折训练2666–2682人；4427是角色人数而非每个模型训练人数。九结局分别准入，不要求共同完整。日级共同有效子集3028人/2782家庭/21279日。

发现期已有方法开发，不能称独立确认。确认结局继续不参与选型，W22/W23正式确认未启动，55位置Holm预算保留。未新增嵌套CV、任意稳定性淘汰或强制功效前置实验。

## 饮食清洗和暴露

处理后食物事件匹配有效日索引，营养素负值/非有限/缺失不补零。事件能量<0或>3000 kcal、重量<0或>2400 g时，该整日退出主要分析；边界保留，缺失重量只标记。日能量500–4000 kcal。个人均值以下2.5 SD的低能量日只标记，不自动删除。

每种人级暴露至少3个相应有效日。纤维日需能量/纤维均有效；日密度=1000×fiber_g/energy_kcal，个人暴露为有效日等权平均，能量协变量按对应窗口平均。不是累计分子分母再求比。纤维u=(密度−训练中位数)/5，单位每5 g/1000 kcal，缓存保留训练拟合的限制立方样条基函数。

UPF未知FoodID不能算非UPF、不能用上下界中点。既有预处理使用类别概率及1024组共享FoodID抽样，非线性整套基函数积分。包中有上下界、食品证据和折内UPF基函数；fraction单位0–1。按本次仅数据/预处理范围，不附食品概率分类器代码；若改变UPF校准，应回原来源另行登记，不把单一期望当摄入真值。

## 菌群、表型和协变量

已锁定同访视双部位样本；完整相对丰度口腔737、肠道2088列，不是折内保留数。训练参与者中检出率≥5%保留；物种训练正丰度第1百分位的一半替精确零；重新闭合、自然对数CLR、训练均值/总体SD标准化；常量列移除。PC只训练拟合。验证固定应用；缺失整份样本不能补全零。

fold0–fold4分别训练拟合并对齐发现主索引；full只供全发现参考，不能用于五折验证。unsupported行可能在缓存中数值置零，必须应用supported掩码，不能当有效菌群。

人级每结局在相应训练角色有效原Y中，用最短95%连续区间估计均值/样本SD；>8 SD设缺失，其余±5 SD缩尾，保留原单位。95%区间用于估计而非删掉外侧5%。示例调用纯转换器复现现有处理顺序；不能以全发现Y阈值计算五折成绩。其余结局替换原Y文件和endpoint标签，分别计算准入。

原协变量：年龄、源性别编码、吸烟（缺失显式UNKNOWN）、对应窗口能量。源sex_code当前准入0/1，不自行推断标签；其他结局单位以Y_unit为准。建模需要的标准化/编码由合作者在自己的训练折拟合。

## 日级数据

提供发现组来源CGM日表、来源food日表，以及去掉所有评分/预测/评分交互的共同有效日表。来源表先按发现IDs、10k、00_00_visit过滤；来源表仍需相应有效性和至少3共同日筛选。共同日表3028人/2782家庭/21279日，每人3–13日、中位7日，person_weight=1/n_i，每人总权重1。

按participant_id/cohort/research_stage/date规范化源日期严格相交，不平移。人级CGM为同访视至少3合格CGM日汇总，不要求饮食逐日重合。日期一致不等于实际进食时刻已确认，不能当餐后分钟级反应标签。日级不套人级8/5 SD规则。新模型需自行处理人内重复、家庭依赖和时间变化混杂。

## 原来源与记录

人级输入：`data_qc/w20_simplified_20261101_v2/`；折转换：其`results/fold_preprocessing/`。

food日表：`data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet`。

CGM日表：`analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet`。

共同日来源：`data_qc/w20_targeted_20261101_v3/results/attempt49b/daily_analysis_TRE_ONLY.parquet`；本包显式只选择无评分字段，不复制整张来源表。

纯转换器：`scripts/preprocessing_20261030_v1/preprocessing_v1_20261030.py`。食物规则代码来源同目录`07_food_qc_amendment.py`，本包附食物rules.json；详细步骤见上文。

较早`data_qc/preprocessing_20261030_v1/analysis_ready_v1/`使用旧50/50划分，不直接沿用其角色/全训练转换做当前CV。

本包构建日志/PID/实际退出：`/home/ec2-user/multisystem_runtime_20261021/collaborator_handoff_20261102_v1/`。新实验记录输入哈希、代码/配置/种子、实际人数/家庭/日数。挂载盘不保证seek/append，NPZ先在BytesIO生成，再以新文件顺序写入。
