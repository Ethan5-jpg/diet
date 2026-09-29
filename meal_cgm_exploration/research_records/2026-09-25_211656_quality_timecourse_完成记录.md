# quality_timecourse：服务器完成记录

证据日期：2026-09-25 21:16:56，用户提供的服务器日志截图。

- 运行ID：`quality_timecourse_20260925T130549_263963Z`。
- 服务器目录：`/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/outputs/quality_timecourse_20260925T130549_263963Z`。
- 截图显示：`VERIFY all source files unchanged`，随后为`QUALITY_TIMECOURSE=COMPLETE`及SUMMARY路径。
- 状态：真实服务器运行已完成，尚待读取具体结果摘要和表格。
- 依交付版run_analysis.py的状态逻辑，COMPLETE要求294个固定规格全部ESTIMATED且末尾输入核验通过；这不是294项显著结果，也不是已经人工核对全部估计值。
- 原42模型复现是程序继续完成的前置检查；本截图没有展示复现表，具体核查记录仍待读取03_model_reproduction.csv。
- 本地仅更新运行状态及来源，尚未接收到该次完整SUMMARY、CSV或图；不新增关于效应稳健性、质量筛选损失或早晚差异的结论。
- 此次截图图像文件本地复制：已复制至evidence/。

读取命令（服务器项目目录内）：

```bash
python3 quality_timecourse/run_analysis.py --show outputs/quality_timecourse_20260925T130549_263963Z
```

下一步按SUMMARY及tables核对：原结果复现、日表交集、样本流失、质量筛选后估计、早晚及晚减早差异、营养调整前后和峰值结果。全七项均保留，逐项记入结论总览；缺少的数值继续保持未知。
