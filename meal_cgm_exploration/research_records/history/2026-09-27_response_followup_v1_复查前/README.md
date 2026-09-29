# 上传包：碳水交互量级与日期分离可行性

版本 `2026.09.27.1`。请只上传这个完整文件夹，放在服务器：

`/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/response_followup/`

与原有`nutrition_bridge/`、`quality_timecourse/`、`seven_diet_meals/`、`hppmeal/`并列；保留旧目录和旧结果。沿用当前`pheno`环境，Python≥3.8，依赖原环境已有numpy/pandas/scipy/matplotlib/pytz。无需安装新依赖。

## 运行

在服务器`meal_cgm_exploration`目录内，运行这一行：

```bash
bash response_followup/run_response_followup.sh --nutrition-run outputs/nutrition_bridge_20260926T071736_080048Z --data-root ..
```

新结果写入`outputs/response_followup_时间戳/`。终端会显示进度、RUN_DIR及状态。`COMPLETE_WITH_LIMITATIONS`是正常的完成标签：日期分离只做可行性计数，尚未确认CGM新窗口重算定义，也没有拟合新效应。

可选仅检查输入：在上面命令末尾加`--stage check`，但完成本次任务请用默认完整运行。若报SHA256不一致或缺文件，将报错截图发回，不改旧文件、不跳过检查。若此前运行过多个后续包，请用输出中的确切RUN_DIR替换`latest`。

## 第一批回传这四页

```bash
for page in 1 2 3 12; do python3 response_followup/run_analysis.py --show latest --page "$page" || break; done
```

- 1：运行状态/目录；2–3：原/joint观测参照；12：日期分离筛选人数。
- 第二批支持和原样本：`for page in 4 5 6 7; do python3 response_followup/run_analysis.py --show latest --page "$page" || break; done`
- 第三批其余差异：`for page in 8 9 10 11; do python3 response_followup/run_analysis.py --show latest --page "$page" || break; done`
- 第四批窗口诊断：`for page in 13 14 15 16; do python3 response_followup/run_analysis.py --show latest --page "$page" || break; done`
- 第五批选择比较/原分布：`for page in 17 18 19 20; do python3 response_followup/run_analysis.py --show latest --page "$page" || break; done`
- 有支持不足时补第21页：`python3 response_followup/run_analysis.py --show latest --page 21`

仍按每次3–4页截图回传，不需要下载CSV或压缩包。`private/`里的个体信息仅留服务器。

## 本包算什么

1. 核对14个B样本（七评分×原/joint）的精确成员及哈希，保留42个已有系数的解释性结果。
2. 根据实际评分、碳水和能量共同支持，计算“高评分与低评分之间，碳水由参照升到高值时，反应变化相差多少mg/dL”。它是原交互系数线性换算，**不是新的绝对血糖曲线，也不是碳水总斜率**。
3. 用冻结原始CGM缓存和日期覆盖，检查前窗餐、后窗CGM的日期分离是否剩下足够资料；特别排除终点插值仍用后窗读数的餐。报告新保留和排除人群；不运行新CGM效应模型。

评分Q25/Q75和餐营养分位数由数据支持决定；门槛详见[PLAN.md](PLAN.md)。七项都报告，包括不支持/不可展示项。原q家族和单位不变；CI是逐项区间且条件于观测参照。通过支持初筛不代表所有协变量组合都有充分观测。

## 输出

- `tables/support.csv`、`descriptive.csv`：参照、观测范围、各支持单元人数/餐数。
- `tables/contrasts.csv`：42项原β/q及支持时的差异/CI；`curves.csv`：有支持的相对变化曲线点。
- `tables/membership.csv`：原成员核对摘要和哈希。
- `tables/feasibility.csv`、`specifications.csv`、`window_distributions.csv`、`selection.csv`：日期分离诊断。
- `figures/relative_changes_*.png/pdf`：两套样本全部七评分的相对变化图。
- `pages/01.txt`–`21.txt`：无个体标识的短页，读回时校验哈希。
- `private/person_windows.csv`、`first_window_members.csv`：内部窗口审计，不回传。
- `manifest.json`、`analysis_spec.json`：来源、状态、固定规则。

## 本地成果与测试

`plot_local.py`只读取累计截图JSON，生成已完成的本地图表；服务器主命令不调用它。科研图中的英文轴标签配中文说明，保留PNG/PDF/SVG和逐行出处。

测试：在项目目录运行`python3 -m pytest response_followup/tests -q`。端到端测试另以环境变量`HPP_FOLLOWUP_SYNTHETIC_RUN`指向已完成的、明确标记SYNTHETIC的nutrition_bridge测试运行；没有变量时跳过该项。不要将测试输出视为真实结果。

后续真实日期分离分析需先审阅这次计数，并确认原iglu计算/清洗、窗口缩放和统计家族。AMED四固定时间段探索留在下一阶段。
