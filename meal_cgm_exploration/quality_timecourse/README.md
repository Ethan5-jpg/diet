# 七项饮食：记录覆盖与餐后时间分段

整体上传本 `quality_timecourse` 文件夹，放到服务器现有 `Data/meal_cgm_exploration/` 下，与 `seven_diet_meals`、`hppmeal` 同级。启动脚本已经在本文件夹里。原 `followup`、七评分代码、原运行输出均保留。

## 这轮能增加什么结论

- 记录覆盖更充分的日期/参与者中，七项饮食的平均净增量、峰值与碳水斜率关联是否保留。
- 饮食评分的关联更集中在餐后0–60分钟，还是60–120分钟；直接估计晚期减早期差异。
- 把上一轮三套样本中，营养调整前后和峰值结果整理为同样本对照，报告系数变化。

不以显著性选评分、结局或阈值。保留全部七项和未估计规格及原因。它可能支持、削弱或限制原结论；不保证产生显著结果。

## 输入要求

1. 原成功的 `meal_...` 输出，尤其 `private/02_meals.csv`、`private/cache.sqlite`、manifest/code_fingerprint。
2. 成功的 `seven_diets_...` 输出及其原评分/协变量来源；版本2026.09.25.2。
3. `Data/Transfer/diet_logging/diet_logging.csv`，使用身份、访视、日期、food_items、total_logging_days和timezone。
4. 现有 pheno 环境。无新依赖；代码采用Python >=3.8语法。新包会核对旧代码、原数据与缓存哈希。不同来源不能悄悄混用。

## 服务器运行

在 `meal_cgm_exploration` 中逐行执行：

```bash
conda activate pheno
cd /home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration
HPP_QUALITY_LOG="quality_timecourse_$(date +%Y%m%d_%H%M%S).log"
nohup bash quality_timecourse/run_quality_timecourse.sh --data-root .. --meal-run outputs/meal_20260924T143630_752200Z --seven-run outputs/seven_diets_20260925T093822_426339Z > "$HPP_QUALITY_LOG" 2>&1 &
tail -n 40 -F "$HPP_QUALITY_LOG"
```

校验大缓存时可能暂时没有新进度。正式运行每50人打印曲线/覆盖进度，再复现原42模型和运行新分析。`QUALITY_TIMECOURSE=COMPLETE`为全部294规格已估计；`COMPLETE_WITH_LIMITATIONS`表示运行完成但部分规格样本不足等，需看原因。`FAILED`应先查看错误，不能当成功。

输出到新的 `outputs/quality_timecourse_日期时间/`，不会覆盖旧结果。日志打印确切 `RUN_DIR` 和 `SUMMARY`。结束后按Ctrl+C退出日志查看，然后将下面路径替换成该次RUN_DIR：

```bash
python3 quality_timecourse/run_analysis.py --show outputs/quality_timecourse_实际日期时间
```

可加 `--stage check` 只做包/源/日表检查，不做曲线重算或模型复现；检查通过不等于分析完成。可用 `--diary-file` 指定日表路径。不要同时改动原包或输入。换路径必须仍为同一份哈希验证通过的数据。

## 样本与分析定义

仅扩展原 `analysis_ready` 餐。复用原CGM合格窗口规则与连接归属，逐餐复现基线、斜率、净iAUC和峰值；不跨缺口或不同传感器拼接。不从CGM反推漏餐。原所有合格餐在筛选前计算 observation_day，筛选后不重置时间起点。

完整参考日来自该传感器首末观测覆盖的本地午夜到午夜（末条加一个15分钟采样间隔）；边界不完整日、多连接覆盖日不进入参考分母，内部无观测日期仍计入。有效覆盖=存在有效读数的15分钟格数/实际本地日格数，夏令时23/25小时日自动处理，重复记录不增加覆盖。使用原配置本地时区；旅行等其他时区的餐不纳入日对齐敏感性，仍留在原主样本和原时间分段分析。

`food_items>0`仅表示该日期有记录，不是餐数或无漏记的证明。未知时区、不匹配日期分别影响对齐；整个人不在日表中时，日志覆盖率标为未知，而非0摄入。`total_logging_days`与实际日期数的分歧单独报告，不直接拿汇总字段作分母。

固定六套质量样本及阈值见PLAN.md。每项/分支筛选后仍需满足原餐数/日期数/碳水变异支持条件。匹配人群对照使用joint保留的同一批人和这些人的全部原合格餐，帮助描述人群选择与日期选择；它不是因果分解。

A为每人等权的平均反应HC3模型，B为每人总权重1的个人固定效应、参与者聚类CR1交互模型。沿用原评分、方向、协变量、样本门槛及原Z单位。营养调整模型控制平均餐次碳水、碳水平方及总能量，不能称为标准餐实验。

早/晚段精确分割原分段线性曲线；使用原餐前基线，不在60分钟重新设基线。可为负值。两段均值平均必须等于原两小时净iAUC/120。直接拟合晚减早；不把早晚显著性不同当作两者差异显著，也不直接解释成胰岛素分泌、清除或机制。

每套质量样本42规格，六套共252（含复现42）；每套沿用主14/峰值14/背景14三个FDR家族。原主样本和joint各7评分×3时间指标=21项一个家族，合计42。总294。未估计的规格仍保留；置信区间为逐项95%区间，不是多重校正区间。这些是重叠样本探索，不是独立复制。

所有公众表均为汇总；个人ID/日期/连接/成员在private/，源路径在manifest。汇报使用SUMMARY、tables、figures。

## 核验

本地验证记录见REVIEW.md。合成数据不代表真实研究结果；真实数据全部通过须以服务器新运行输出为准。

官方资料：[HPP饮食日志](https://knowledgebase.pheno.ai/datasets/005-diet_logging.html)、[HPP CGM](https://knowledgebase.pheno.ai/datasets/017-cgm.html)。本轮90%、80%、7日是固定的操作性敏感性阈值，不是从官网引用的完整记录标准。
