# 上传包：碳水交互量级与日期分离可行性

版本 `2026.09.27.2`（复查修正版）。请上传这个完整文件夹，放在服务器：

`/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/response_followup/`

与原有`nutrition_bridge/`、`quality_timecourse/`、`seven_diet_meals/`、`hppmeal/`并列；保留旧目录和旧结果。沿用当前`pheno`环境，Python≥3.8，依赖原环境已有numpy/pandas/scipy/matplotlib/pytz。无需安装新依赖。

若之前已上传本包旧版，只更新整个`response_followup/`文件夹，包括`MANIFEST.json`；其他分析目录保持原样。

## 运行

在服务器`meal_cgm_exploration`目录内，运行这一行：

```bash
bash response_followup/run_response_followup.sh --nutrition-run outputs/nutrition_bridge_20260926T071736_080048Z --data-root ..
```

新结果写入`outputs/response_followup_时间戳/`。终端会显示进度、RUN_DIR及状态。`COMPLETE_WITH_LIMITATIONS`是正常的完成标签：日期分离只做可行性计数，尚未确认CGM新窗口重算定义，也没有拟合新效应。

可选仅检查输入：在上面命令末尾加`--stage check`，但完成本次任务请用默认完整运行。若报SHA256不一致或缺文件，只发报错截图，不改旧文件、不跳过检查。查看特定运行时，用输出中的确切RUN_DIR替换`latest`。如果最新运行失败或未完成，`latest`会明确报错，不会悄悄显示更早的成功结果。

## 只回传这3份关键摘要

```bash
python3 response_followup/run_analysis.py --show latest --brief
```

1. **状态与日期分离**：成员核对、可换算对比数、前后日期窗口保留人数与关键排除计数。
2. **原样本关键结果**：七项评分的观测参照、支持人数，以及净增量/峰值/正增量的差异、95%CI和原q。
3. **joint关键结果**：同样的关键指标，用于核对覆盖筛选后的变化。

每份约一屏，通常3张截图即可。若一次输出不方便截图，把命令中的`--brief`换成`--page 1`、`--page 2`或`--page 3`，单独查看对应摘要。无需继续回传明细页，也无需下载CSV或压缩包。只有发现具体异常时才定点查看相关明细；`private/`里的个体信息仅留服务器。

若已经完成旧版运行，新版查看命令可直接把其已保存、经哈希核对的汇总表压缩成3份摘要，不修改旧结果。摘要会注明“旧版”，旧版共同能量范围边界判定仍是原值；要应用本次修正需用新版运行命令生成一个新结果目录。

## 本包算什么

1. 核对14个B样本（七评分×原/joint）的精确成员及哈希，保留42个已有系数的解释性结果。
2. 根据实际评分、碳水和能量共同支持，计算“高评分与低评分之间，碳水由参照升到高值时，反应变化相差多少mg/dL”。它是原交互系数线性换算，**不是新的绝对血糖曲线，也不是碳水总斜率**。
3. 用冻结原始CGM缓存和日期覆盖，检查前窗餐、后窗CGM的日期分离是否剩下足够资料；特别排除终点插值仍用后窗读数的餐。报告新保留和排除人群；不运行新CGM效应模型。

评分Q25/Q75和餐营养分位数由数据支持决定；门槛详见[PLAN.md](PLAN.md)。七项都报告，包括不支持/不可展示项。原q家族和单位不变；CI是逐项区间且条件于观测参照。不同评分/样本使用不同观测参照，不能直接按换算值排名。通过支持初筛不代表所有协变量组合都有充分观测。

## 输出

- `tables/support.csv`、`descriptive.csv`：参照、观测范围、各支持单元人数/餐数。
- `tables/contrasts.csv`：42项原β/q及支持时的差异/CI；`curves.csv`：有支持的相对变化曲线点。
- `tables/membership.csv`：原成员核对摘要和哈希。
- `tables/feasibility.csv`、`specifications.csv`、`window_distributions.csv`、`selection.csv`：日期分离诊断。
- `figures/relative_changes_*.png/pdf`：两套样本全部七评分的相对变化图。
- `pages/01.txt`–`03.txt`：仅3份关键摘要，无个体标识，读回时校验哈希。
- `private/person_windows.csv`、`first_window_members.csv`：内部窗口审计，不回传。
- `manifest.json`、`analysis_spec.json`：来源、状态、固定规则。

## 本地成果与测试

`plot_local.py`只读取累计截图JSON，生成已完成的本地图表；服务器主命令不调用它。科研图中的英文轴标签配中文说明，保留PNG/PDF/SVG和逐行出处。

复查结果：19项测试通过，详见[REVIEW.md](REVIEW.md)。在项目目录运行`python3 -m pytest response_followup/tests -q`；端到端测试另以`HPP_FOLLOWUP_SYNTHETIC_RUN`指向明确标记SYNTHETIC的nutrition_bridge测试运行，旧版读回测试以`HPP_FOLLOWUP_LEGACY_RUN`指向旧版合成运行。没有相应变量时跳过对应测试。不要将测试输出视为真实结果。

后续真实日期分离分析需先审阅这次计数，并确认原iglu计算/清洗、窗口缩放和统计家族。AMED四固定时间段探索留在下一阶段。
