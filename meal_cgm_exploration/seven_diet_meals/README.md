# 七项饮食变量 × 餐次血糖

版本 2026.09.25.2。此包回答两个新问题，补充原七饮食变量×两周CGM汇总指标研究：

1. 个人整体饮食评分，是否与合格餐的平均餐后净增量、峰值升幅相关？
2. 同一个人不同餐次碳水变化对应的血糖斜率，是否随其个人饮食评分不同？

7项全纳入：AHEI（项目7组分改良版）、AMED（8组分适配版）、更正hPDI、原反向rEDIH、modified EAT-Lancet13、个人NOVA4供能比例、个人碳水供能比。后两项不是整体健康评分。保留原标准化单位和方向，不将个人评分假装成餐次评分。旧餐次NOVA4结果与此处个人NOVA4是不同问题。

## 上传一个文件夹

仅将本地 `seven_diet_meals/` 整个文件夹上传到服务器现有 `meal_cgm_exploration/` 下。启动脚本就在该文件夹内。本包不依赖 followup，保留服务器旧的 hppmeal、scripts、outputs。

## 服务器运行

在原 pheno 环境，粘贴：

```bash
cd /home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration
HPP_SEVEN_LOG="seven_diets_$(date +%Y%m%d_%H%M%S).log"
nohup bash seven_diet_meals/run_seven_diets.sh --data-root .. --run-dir outputs/meal_20260924T143630_752200Z > "$HPP_SEVEN_LOG" 2>&1 &
tail -n 50 -F "$HPP_SEVEN_LOG"
```

默认先核查全部输入再分析，一次执行即可。仅想核查时，加 `--stage check`（不会估计模型）。成功显示 `SEVEN_DIETS=COMPLETE`；部分规格因样本/共线性不能估计时显示 COMPLETE_WITH_LIMITATIONS，并保留明确原因。出现 ERROR 则发最后的日志。强制杀进程可能留下 RUNNING，不代表成功。

日志打印新的 `outputs/seven_diets_...`，不覆盖 meal 或 repeated 结果。完成后 Ctrl+C 退出日志查看，运行下列命令查看摘要（替换为实际输出目录）：

```bash
python3 seven_diet_meals/run_analysis.py --show outputs/seven_diets_实际时间戳
```

## 输入接口

复用指定原运行的 private/02_meals.csv，核查原 manifest、BUILD餐次哈希、hppmeal/scripts指纹、原协变量哈希。输入或原代码改变会拒绝混用。

从 Data 根目录读取以下既有表（与前期全七评分分析一致）：

- `diet_microbiome_glucose_analysis/outputs/data/00_current4_diet_cgm_cohort.csv`：AHEI_z、AMED_z、rEDIH_z。原 hPDI 列即使存在也不使用。
- `diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/hpdi_correction_audit/corrected_scores/hpdi_participant_scores.csv`：hpdi_score_energy_adjusted_z。
- `diet_microbiome_glucose_analysis/outputs/new_diet_extension/data/15d0_new_diet_cgm_cohort.csv`：EAT13_z、NOVA4_z、Carbohydrate_pct_z、mean_daily_energy_kcal。
- 原配置指向的 `co-variant/outputs/data/02_covariate_master.csv`：人口、行为、BMI、CGM设备、饮酒和排除标记。

默认的 current4/new3 是前期已经与CGM汇总表匹配、并保留至少一个相应饮食变量和主要CGM结局的导出表，覆盖范围继承该筛选；并非所有原始饮食记录参与者。本轮不重新改变原Z值或用其他来源填补未匹配者。`tables/00_score_coverage.csv` 分别列出每项评分在合格餐参与者中“源表没有此人”“源表有此人但评分缺失”“评分可用”的人数及可匹配餐数；后续协变量及餐次支持筛选见 `01_sample_flow.csv`。

若评分表已搬家，在此包外复制 sources.json 并修改三个路径，运行时加 `--sources-config 你的新配置.json`。不要修改已校验包内文件；列名差异应先核查来源，而不是任意选择相似列。列缺失或重复ID报错；不会自动平均重复记录、补零或替换更正hPDI。评分源是规范化个人baseline导出；若表缺少cohort/visit字段，报告中明确不能由这些列独立验证访视，不凭个人ID认定时序。

## 统计口径

A：每人合格餐的平均净增量和平均峰值升幅，每人一行，HC3。M1调整年龄、性别、教育、吸烟、通常睡眠、通常活动、维生素、激素、BMI、CGM设备，另有平均餐前基线/趋势、时刻正余弦、前餐间隔、观察日、餐次时长；hPDI另调饮酒，EAT13/NOVA4/碳水另调日能量。M2再加平均餐次碳水、碳水平方均值和平均餐能量。M1/M2同一完整样本，至少5餐/3日期。

B：个人固定效应；碳水/10g及平方、餐能量、餐前基线/趋势、时刻正余弦、前餐间隔、观察日、餐次时长；焦点为个人评分×碳水。加入年龄/BMI/性别及该评分特有饮酒或日能量×碳水的斜率控制。个人不变主效应被吸收。每人总权重1、按人聚类CR1、吸收固定效应计入自由度。至少10餐/3日期/碳水范围30g。

三个设置：各评分可用的primary、明确排除标记为0的strict、七评分+饮酒+日能量共同完整的primary_common7。每个模型至少50人、30残差自由度。不放宽门槛追求显著；常量控制项记入诊断，非恒定共线设计明确不能估计。主设置与敏感性样本高度重叠。

每设置42规格：A的2模型×2结局×7评分=28，B的2结局×7评分=14。总126规格。FDR家族：A-M2+B净增量共14项为主；峰值14项为次；A-M1两结局14项为背景家族。未估计项保留计划分母。各项95%CI是未作同时校正的点区间。

样本不足或评分无法匹配时，仍保留全部126个规格及其原因，系数和描述表保留表头；空表表示没有相应可估计数据，不能解释为效应为零。

净增量为原net_iAUC/120，单位mg/dL；峰值升幅也是mg/dL。B交互为每原饮食Z、每10g碳水的斜率差。调整能量后的碳水斜率不能描述为自由添加碳水的作用。该线性交互允许碳水总体二次关系，但未检验任意非线性交互，不将它称为生理碳水耐受性。

这是从现有数据提出的新探索；评分与餐次记录窗可重叠，不称长期先验预测。A不是完全同餐条件下的比较；B不报告不可识别的个人饮食主效应。既有菌群和20项CGM结果不改写。本包不重复扩展所有菌。

## 看结果

先看SUMMARY.md中的两张全七项表；随后看各评分的主/strict/common7对照、峰值结果、效应大小与区间。不要按显著数量排列“哪个评分最好”。若背景模型与餐次调整模型不同，属于同样本条件调整后的变化，不直接命名为机制解释比例。

汇报使用 tables/、figures/ 和摘要。private/ 包含人级分析输入及模型餐次成员，manifest含原始路径，保留服务器。所有图若使用合成数据会明确标识；本地测试不是研究结果。方法设计和文献链接见 PLAN.md；上传前验证见 REVIEW.md。
