# 四候选菌：餐后绝对血糖与餐数稳健性

版本2026.10.04.2。按照2026-10-04确认方案新增两项补充分析；旧主模型和全部原结果保留。只上传本文件夹，不需要重传数据、不需下载结果。默认打印三段汇总。

## 上传和运行

把整个`gut_absolute_robustness`文件夹放在服务器`meal_cgm_exploration`下，与`gut_meal_bridge_v2`、`hppmeal`、`seven_diet_meals`、`outputs`同级。

在原pheno环境、`meal_cgm_exploration`目录运行：

```bash
python3 gut_absolute_robustness/run_analysis.py
```

默认读取`outputs/gut_meal_bridge_v2_20260929T044951_469958Z`，具体需要其中的`manifest.json`、`tables/models.csv`、`tables/sample.csv`、`private/person_inputs.csv`、`private/meal_membership.csv`。不要删除这些原文件。原运行搬家时用`--source 实际原运行目录`，不自动选择其他来源。不使用刚完成的`gut_same_visit_complete_*`导出目录作为输入，因为那里没有所需的个人输入表。

运行顺序：核对新包/依赖/来源哈希与成员 → 列出三个门槛人数 → 复现旧48项效应与p/q → 运行A与B → 验证原文件未变 → 保存新目录及三段摘要。

新结果为`outputs/gut_absolute_robustness_时间戳`。成功为`COMPLETE_WITH_LIMITATIONS`；有无法估计的规格为`PARTIAL`（退出码2，NE保留，不降低门槛）。输入损坏、来源复现不符为`FAILED`；请发错误和RUN_DIR，不修改旧文件绕过检查。

## 截图回传，只需三段

程序末尾已打印三段。若被前面日志挤出屏幕，依次查看：

```bash
python3 gut_absolute_robustness/run_analysis.py --show latest --page 1
python3 gut_absolute_robustness/run_analysis.py --show latest --page 2
python3 gut_absolute_robustness/run_analysis.py --show latest --page 3
```

也可用`--show latest --brief`一次显示三段。`latest`指最新尝试，失败不会回退展示旧成功。

1. 来源复现、六组门槛人数/餐数、保留比例及保留/排除者描述差异。摘要最大SMD不含用于筛选的餐数/日期数；完整表仍保留这些字段。
2. A：相同调整条件下四菌与餐前Pre、净增量Net、绝对餐后Abs的关联。
3. B：原模型在5餐/3天、10餐/5天、20餐/7天的效应、95%CI和原家族q。

摘要是舍入显示，正式精度在CSV。接近0.05的q会自动增加显示位数，保留其位于0.05哪一侧的信息。NA/NE不能解释成零效应。若q显示0*，只表示数值下溢。只回传终端聚合页，`private/`留在服务器。

## A与原模型的关系

Abs为0–120分钟曲线的平均绝对血糖，由同批合格餐的个人平均Pre+Net计算；不是120分钟单点，不是把旧模型系数相加。

A的三个结局均使用原餐前模型的共同协变量：背景、个人饮食、餐营养及餐情境；不含餐前血糖和餐前斜率。A中的Net因此是**新调整条件下的诊断性比较**，不能替代原来调整餐前水平/斜率的主Net模型。8个A-Pre系数复现原参照，但新q_panel12按每组4菌×3结局12项校正，与旧q不同。

直接拟合各自HC3区间，再核对同X下βAbs=βPre+βNet。这个恒等式只是程序核对，不是机制、中介或解释比例，也不把三项当独立发现。

## B与选择问题

三个门槛同时要求合格餐数和不同日期数；仅筛人，不截取前10餐或前20餐，不重新求Z，不把不合格餐纳入。保留者所有原合格餐及冻结均值不变。

每个门槛、每个样本均保持原调整集、Net4项/Pre+Peak8项FDR家族；另报q_meal12。原5/3组复用已核对的24项，10/5和20/7共48项新规格。不同门槛人群变化用保留/排除者均值、SD和SMD描述；这不是偏差已被消除的证明。比较系数和区间，不将显著性跨线解释成正式效应差异。

当两组某特征各自无变异、但取值不同，SMD分母为零，记为未定义并标注`ZERO_VARIANCE_SEPARATION`；摘要优先提示该完全分离情况。相同常数、组人数不足则分别注明原因，不将这些情况当作正常有限SMD。

菌群仍按同访视，采样先后不筛选；不加入两周5%缺失筛选。严格样本和三个门槛样本均重叠，不是独立验证。A/B为看过既有结果后新增探索/敏感性，未声称原研究预注册；各组q不控制跨全部门槛/样本的总FDR。

## 服务器完整输出

| 文件 | 内容 |
|---|---|
| tables/source_reproduction.csv、source_models_48.csv | 48项来源复现与旧结果 |
| tables/absolute_models.csv | A的24项，含8项原Pre参照；效应、SE、CI、p、q_panel12 |
| tables/support_models.csv | B的72项，含原24项及新增48项；q_family和q_meal12 |
| tables/support_flow.csv、selection_balance.csv | 门槛流失和人群变化 |
| tables/algebra_check.csv、not_estimated.csv | 代数核对与NE原因 |
| private/participant_membership.csv、group_membership_hashes.csv | 服务器私有成员和指纹 |
| analysis_spec.json、manifest.json、METHODS.md、pages/ | 冻结设计、来源/软件/结果指纹、方法与摘要 |

Python>=3.8、numpy/pandas/scipy；复用现有原模型代码，不安装软件，不依赖window_qc_followup、window_qc_diagnosis或gut_same_visit_completion。新包的合成验证不构成新的真实研究结果，真实数值由用户上传运行后产生。
