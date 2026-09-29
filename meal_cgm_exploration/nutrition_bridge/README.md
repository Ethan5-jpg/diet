# nutrition_bridge 上传与运行

把整个 `nutrition_bridge` 文件夹上传到服务器现有 `meal_cgm_exploration` 内，与 `quality_timecourse`、`seven_diet_meals`、`hppmeal` 并列。运行脚本已经在此文件夹内。旧包和旧结果保留不动。

Python >=3.8。沿用 pheno 环境中的 numpy / pandas / scipy / matplotlib。所有读取均只读；输出为新的 `outputs/nutrition_bridge_时间戳/`，不覆盖旧结果。

## 服务器操作

先逐条执行：

```bash
conda activate pheno
cd /home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration
```

启动（下面是一整条命令；自动创建独立日志）：

```bash
nohup bash nutrition_bridge/run_nutrition_bridge.sh --data-root .. --quality-run outputs/quality_timecourse_20260925T130549_263963Z > "nutrition_bridge_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

查看最新日志（一整条命令）：

```bash
tail -n 35 -F "$(ls -t nutrition_bridge_*.log | head -n 1)"
```

出现 `NUTRITION_BRIDGE=COMPLETE` 或 `COMPLETE_WITH_LIMITATIONS` 后，按 Ctrl+C 退出日志查看，然后看第1页：

```bash
python3 nutrition_bridge/run_analysis.py --show latest --page 1
```

第1页有页码目录。把最后的 `1` 改为 `2`、`3` 等即可，每条命令只打印一页。先回传第2、3、6、7、9、10、14、15页，其余按需要补充。`--page 30` 列未估计原因。输出不会包含逐人身份。`latest` 仅选择本包最近完成的输出；需指定旧次结果时，用目录路径替换 latest。

`COMPLETE` 只表示计算完成；是否有新的可靠结论，要看效应、区间、样本与诊断。`COMPLETE_WITH_LIMITATIONS` 表示固定规格中有无法估计的项目，并非所有分析失败。若出现 `ERROR`，截图日志最后约35行即可，不必重跑原有包。

## 输入与结果口径

从已完成 quality 输出追溯 meal、seven、评分、协变量、日记和原CGM缓存；核对来源哈希、旧模型及成员。默认整体CGM来自原已归档路径：

`Data/cgm_deal/cgm论文新增17指标/outputs/run_20260919T030229_482714Z_c1a0f480/data/07_cgm_paper_extended_phenotypes.csv`

若此文件被移动，运行时增加 `--cgm-file "新路径"`；不能指向另一个版本来代替。程序不会自动选最新文件。当前哈希可追溯本轮输入，但原记录缺少历史哈希，不能证明与过去的数值绝对一致。

完整固定方案在 PLAN.md。主要汇总：

- `tables/02_reproduction.csv`：原/joint共84模型复现。
- `tables/04_adjustment_models.csv`：同样本分步调整系数、区间/q、VIF和影响诊断。
- `tables/04_score_nutrient_correlations.csv`：原始及背景残差相关。
- `tables/05_positive_models.csv`、`05_positive_net_comparison.csv`：新增正增量及原净增量并列；系数差仅描述。
- `tables/05_response_distribution.csv`：零值、缺失、分布和正净面积差。
- `tables/06_bridge_flow.csv`、`06_bridge_models.csv`：同连接匹配损失、原20项衔接及配对人数。
- `tables/07_coefficients.csv`：完整回归系数。`figures/`：分步调整图。`pages/`：54页短表；31–50页补全调整模型各步区间，51–54页补充评分相关及正增量分布。

`private/` 有成员和逐人分析数据，`manifest.json` 有来源路径；均留在服务器。模型不把漏记饮食补成进食、不填补CGM缺口。日记覆盖也不能证明每餐记录完整。

诊断不自动删人。正增量缺失的规格保留不可估计状态，不静默改换样本；整体衔接按每个结局的配对可用人群分析，配对人数和成员哈希逐行记录，三项餐特征可能因缺失人数不同。

本地核验仅使用合成数据，见 REVIEW.md；尚未产生真实服务器的新结果。
