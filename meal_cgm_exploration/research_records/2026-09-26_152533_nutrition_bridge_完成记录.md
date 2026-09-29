# nutrition_bridge：真实服务器完成记录

证据：用户提供2026-09-26 15:18:42启动日志和15:25:33结果第1页截图。原图已复制至evidence/，未下载服务器输出CSV或目录。

- 运行ID：`nutrition_bridge_20260926T071736_080048Z`。
- 服务器结果目录：`/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/outputs/nutrition_bridge_20260926T071736_080048Z`。
- 第1页显示：`DATA=REAL`、`STATUS=COMPLETE`、复现`84/84`、扩展估计`302/302`。
- 启动日志显示：`INPUTS ready meals=183709, CGM people=7493`。这是输入阶段计数，各模型的实际有效人数和餐数尚未展示。
- 第1页共有54页目录。计划家族：每套样本调整70项、正增量21项、整体CGM衔接60项；逐项95%区间。
- 本地交付版的复现检查涵盖原/joint各42项旧模型的状态、人数、餐数、成员哈希、系数、标准误、区间、p和q。当前截图显示程序检查完成，尚未人工逐项查看这轮数字。
- 302是模型规格数，不是302项显著结果或独立科学发现；含旧A背景和完整调整的重复规格。
- 当前新增可确认事实仅为真实运行完成与程序复现通过，尚不能确认营养调整解释、正增量敏感性或整体CGM衔接是否提供了新证据。

## 下一批截图

先展示第2、3页（原/joint平均净增量分步调整），再看正增量第6、7、9、10页，以及整体CGM衔接第14、15页。后续按解释需要补各步区间、影响诊断、分布及完整20项，不按显著性筛选结果。

```bash
python3 nutrition_bridge/run_analysis.py --show outputs/nutrition_bridge_20260926T071736_080048Z --page 2
python3 nutrition_bridge/run_analysis.py --show outputs/nutrition_bridge_20260926T071736_080048Z --page 3
```

所有新结论继续带运行ID及截图来源，按显示精度转录；未展示值保持未知。合成核验与真实研究结果分开归档。
