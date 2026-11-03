# W20 targeted v3：已实际运行并验收的接续

日期：2026-11-01。当前聊天通过新版 Chrome control-chrome 入口，实际操作 5bc05bbe… Amazon WorkSpaces；未操作用户截图中的 14054… 其他项目标签。无 SSH/API/个体数据下载，不刷新或重启会话。Chrome 用户标签保持打开。当前临时 tab ID 不作为未来接续默认，重新从实际清单识别。

## 当前完成状态

- 47 预检、48 方案登记和四项算法检查实际退出0。
- 首次49 worker2256因 npz直接写入挂载目录 seek失败，实际退出1，失败源码/日志/不完整文件保留。
- 49b runner3509/worker3516，实际退出0，24.84秒；只修复IO和新输出目录，统计规则不变。
- 50只传输未运行。50b runner4466/worker4473，实际退出0，1.30秒；固定参数家庭出折评分的日级分析完成。
- 初次51因 npz内对象型分组字符串被allow_pickle=False拒绝，实际退出1，日志保留；51b改为只读数值数组、从已核对的同序Parquet取家庭列，worker6835实际退出0。18预测公式和10评分公式独立读回，实际日期/值、个人权重、WLS和家庭协方差独立重算通过。
- 52封存时 append在挂载盘不支持，实际退出1；52b改写新FINAL_REPORT.md，worker8872，实际退出0，最终43个产物登记。52b的ps只见当时本身封存进程，没有旧模型/日级工作进程。
- 65旧v1保护文件、197旧v2产物哈希未变；9当前源码哈希通过。角色70/30和发现单层5折保持；确认Y未用于新模型或选型；正式55 Holm、W22/W23未开始。

## 科学结果

发现角色4427人/3989家庭；纤维×D_Glu实际CV与全发现拟合3343人/3059家庭；折训练2666–2682人、2446–2450家庭。
匹配null R²0.0782722301。v2 ΔR²+0.0011317449；残差CLR+0.0018225266（λ10）；PC5+0.0004972022（λ1）；PC10+0.0039876591（λ1）。条件配对家庭区间全部跨0，均是参与调参的开发指标。没有据此替换正式模型。

连续纤维u按+5g/1000kcal表达，共享非线性背景保留。新模型是明确的正则残差变系数投影，不是二元处理R-learner或因果效果。它包含与斜率系数绑定的-e(C,Z)*Z主效应分量，预测增益本身不等于纯斜率修饰。详细公式与限制见targeted_spec.md和报告。

日级：3028人/2782家庭/21279真实共同有效日，最少3、中位7、最多13天；每人权重和1；个人固定效应、家庭CR1。家庭出折评分超参数在本轮新拟合前固定为k5、主λ1、交互λ10；每人的家庭结局不参与其评分学习，方向为更高的学得饮食关联斜率；不用全发现Y对齐符号。3名日级分析者没有人级CV Y，说明没有把人级结局排除硬套给日级。
口腔交互+0.0487895 mg/dL per fiber5 per scoreSD，95%CI[-0.0334063,0.1309852]；肠道+0.0887650，CI[-0.0058419,0.1833720]；联合探索性P0.0934725。区间条件于已学评分，不包含学习/适应不确定性，不是独立确认。无明确菌群修饰证据。

## 精确远程路径

项目：/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021
源码：scripts/w20_targeted_20261101_v3
登记：data_qc/w20_targeted_20261101_v3/registration.json、targeted_spec.md
成功结果：data_qc/w20_targeted_20261101_v3/results/attempt49b
最终报告：该目录FINAL_REPORT.md
核验：verified_summary.json；seal_summary_final.json；final_artifact_manifest_TRE_ONLY.json
模型：model_comparison.csv、model_summary.json、score_stability.csv、各折和full预测器npz
日级：daily_summary.json、daily_effects.csv、daily_input_receipts.json、daily_analysis_TRE_ONLY.parquet、daily_regression_TRE_ONLY.npz
日志/PID/实际退出：/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3
所有历史失败状态均保留；不要将49/51/52失败日志误认为49b/51b/52b仍未完成，或盲目重启。

## 本地交付及下一步边界

主报告：reports/HPP_W20_Targeted_V3_Daily_CGM_2026-11-01.md。
本目录存自行生成的通用源码、传输命令、源码哈希和可见聚合截图，不能当作个体输入替代TRE。
本轮先完成纤维×血糖与日级分析，已评估推广：当前小幅开发增益不足以确定正式赢家；保留v2、残差CLR和PC对照的全部结果，不追加搜索/稳定性硬闸。UPF和九结局正式范围仍保留，其余组合尚未扩展，不自动变成55项正式主模型，也不依据确认Y选型。

已将通用浏览器接续和IO验收经验更新到hpp-workspaces-browser技能2.0.4；首次52追加写入失败及52b新文件封存成功亦应遵循顺序写入路径。所有浏览器动作后显式等待至少1.5秒并观察。
