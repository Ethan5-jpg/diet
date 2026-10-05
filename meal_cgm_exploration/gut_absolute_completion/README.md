# 绝对餐后血糖：补齐三个餐数门槛

版本2026.10.04.1。上传整个`gut_absolute_completion`到服务器`meal_cgm_exploration`下，与`gut_absolute_robustness`同级。旧目录与旧outputs全部保留。本轮只补算32个新规格，复用40个共同调整参照，并独立保留上轮72行原模型结果。

## 运行

在既有pheno环境、meal_cgm_exploration目录执行：

```bash
python3 gut_absolute_completion/run_analysis.py
```

默认来源是已完成的两次运行：

- `outputs/gut_meal_bridge_v2_20260929T044951_469958Z`：原个人汇总、餐次成员、样本与48项模型。
- `outputs/gut_absolute_robustness_20261004T145431_649081Z`：上轮A24、B72、样本/选择描述、私有成员与文件指纹。

这两个来源的`manifest.json`、`tables/`和`private/`需要保留。无需重传数据、不读取原始CGM、不依赖两个window_qc目录。现有gut_absolute_robustness须为2026.10.04.2；程序核对依赖文件指纹。更换来源路径时用`--source 实际原目录 --previous 实际上轮目录`；搬迁允许，数据内容必须一致。不自动选用其他运行。

## 只回传三段

运行末尾打印KEY 1/3、2/3、3/3。需要重新查看时：

```bash
python3 gut_absolute_completion/run_analysis.py --show latest --page 1
python3 gut_absolute_completion/run_analysis.py --show latest --page 2
python3 gut_absolute_completion/run_analysis.py --show latest --page 3
```

1. 两个来源核验、六组人数/餐数、人群差异、40复用/32新拟合及图件状态；原Net主模型的校正结果单列。
2. 主样本的Pre、Net、Abs，全部四菌×三个门槛。
3. 严格样本的对应结果。

仍只需截图回传，无需下载结果。原始p/SE/CI/q与详细NE原因在服务器CSV保留。摘要的q接近0.05时会增加显示位数；0*表示数值下溢，不是理论概率为零。最新失败时`latest`不会回退显示旧成功结果。

## 固定的比较

门槛同时要求至少5餐/3日、10餐/5日、20餐/7日。只筛人，保留入选者全部原合格餐及冻结均值和CLR-Z；每人等权。

共同调整采用原餐前模型的背景、饮食、餐营养和餐情境协变量，不含餐前血糖及斜率。三个结局为Pre、Net、Abs（0–120分钟平均绝对血糖，等于同餐Pre+Net）。每样本×门槛固定4菌×3结局12项BH，`q_panel12`。高门槛Pre的系数/CI/p可以复用旧B，但它在这个新面板中的q必须重新归入12项家族，不能沿用旧q_family。

共同调整结果总72行：旧A24行、旧B高门槛Pre16行、新拟合Net/Abs32行。旧B72行保留原调整集与Net4/Pre+Peak8的q_family、q_meal12。合并展示144行中包括两块共用的24个Pre估计，不是144个独立发现。新Net不替换原调餐前的主Net模型；32个新规格也不都是统计独立的。

这是在看到既有结果后确定的探索/敏感性分析。strict与门槛样本重叠，不称独立验证。24项系数恒等式只作代数核对，不算中介比例。各组BH不宣称控制跨所有门槛和样本的总FDR。所有候选、阴性、NE保留，不依结果调整门槛。

## 输出与状态

新目录：`outputs/gut_absolute_completion_时间戳`。

| 文件 | 内容 |
|---|---|
| tables/common_models.csv | 72行共同调整结果、估计来源、CI/p/q_panel12 |
| tables/original_models.csv | 上轮72行原模型，原q_family/q_meal12保留 |
| tables/all_meal_models.csv | 上述144展示行，标明block及共用Pre |
| tables/source_models_48.csv | 最初48项，仅作来源参照 |
| tables/support_flow.csv、selection_balance.csv | 六组人数、餐数和保留/排除者差异 |
| tables/reuse_audit.csv、algebra_check.csv、not_estimated.csv | 40行复用、24项恒等式、NE原因 |
| figures/Figure1_common_adjustment.* | 新共同调整三结局×两套样本，三个门槛，PNG/PDF/SVG |
| figures/Figure2_original_models.* | 原模型的同类完整展示，PNG/PDF/SVG |
| RESULTS_en.md、METHODS.md、FIGURE_LEGENDS.txt（在figures中） | 数值结果草稿、方法、图注 |
| manifest.json、analysis_spec.json、pages/ | 来源/软件/结果指纹、分析方案、三页摘要 |
| private/ | 仅服务器保留的个人成员及成员指纹 |

`COMPLETE_WITH_LIMITATIONS`：共同调整72项全部估计，图件完整。`PARTIAL`：存在NE，保留原门槛与全部计划行。`COMPLETE_ANALYSIS_FIGURES_UNAVAILABLE`：分析完整但Matplotlib或其依赖不可导入，完整表保留，错误见FIGURES_ERROR.txt。`FAILED`：来源/验证或运行错误，回传错误与RUN_DIR。部分完成/缺图退出码为2。

Python>=3.8语法、numpy/pandas/scipy；图件使用Matplotlib。程序不自动安装或更改环境。图按服务器全精度数据生成；本地测试只用合成数据并明确标记，不属于真实HPP新结果。Methods/Results自动稿还需结合回传数据进行科学表述复核。
