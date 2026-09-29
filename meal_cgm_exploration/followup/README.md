# 重复餐与餐前背景：后续分析

版本：2026.09.25.3（单文件夹上传版）。此模块读取已经完成的 meal 运行，不重新扫描原始 CGM，不修改原分析结果。当前真实运行是 `outputs/meal_20260924T143630_752200Z`。

## 上传

只上传本地 `meal_cgm_exploration/followup/` 整个目录，启动脚本已放在里面。服务器目标结构为：

```
/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/
  followup/
    __init__.py
    io_utils.py
    repeats.py
    prediction.py
    reporting.py
    run_repeated.py
    run_repeated.sh
    MANIFEST.json
    README.md
    tests/
```

服务器现有 `hppmeal/`、`scripts/`、`outputs/` 必须保留。旧源文件未修改，新入口依赖旧模块并校验其指纹。仅同步 `followup/` 这一个文件夹即可；不要用“删除目标多余文件”的同步方式清掉服务器 outputs。若此前上传过根目录的旧启动脚本，本次使用下方的新路径即可。

沿用 `(pheno)` 环境及原 requirements.txt，无新增依赖。支持 Python 3.8 语法；本地验证所用解释器和范围见 VALIDATION.md。

上传前复查细节见 REVIEW_2026-09-25.md；本版对核对后到读取前的文件变化也会拒绝，并在结束时核对本模块及其依赖的旧代码。

## 启动（在服务器终端粘贴）

```bash
cd /home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration
HPP_REPEAT_LOG="meal_repeated_$(date +%Y%m%d_%H%M%S).log"
nohup bash followup/run_repeated.sh --data-root .. --run-dir outputs/meal_20260924T143630_752200Z > "$HPP_REPEAT_LOG" 2>&1 &
echo "日志文件：$HPP_REPEAT_LOG"
tail -n 50 -F "$HPP_REPEAT_LOG"
```

默认固定随机种子 20260925、1000 次 bootstrap。可用 `--bootstrap 2000` 增加区间的数值精度；不要按区间是否跨零选择种子或抽样次数。每次启动都创建新的输出目录，不覆盖前次结果。

日志开始处会显示 `RUN_DIR=.../outputs/repeated_...`。出现 `FOLLOWUP=COMPLETE` 或 `FOLLOWUP=COMPLETE_WITH_LIMITATIONS` 表示流程完成；后者需查看哪些区间或模型因样本不足而未估计。`FAILED` 或 `INTERRUPTED` 输出不可作为完成结果。

`tail -F` 会继续等待日志，即使计算已经完成。看到最终状态后按 Ctrl+C 退出日志查看，不会取消已经完成的分析。若尚在运行，Ctrl+C 也只停止这个日志查看；后台任务仍继续。

查看摘要（把目录替换成日志打印的真实 RUN_DIR）：

```bash
python3 followup/run_repeated.py --show outputs/repeated_实际时间戳
```

也可直接打开新目录的 SUMMARY.md。不要将命令中的示例时间戳当成真实目录。

## 分析规则

### 人群和配对

1. 原运行必须成功 BUILD，配置记录已接受自报进食时间的操作性解释，且餐次 SHA256 和原代码指纹一致。该标记不代表日志准确性被独立验证。
2. 协变量表与原分析记录的 SHA256 一致，继续按同队列/访视匹配。变化后不会静默使用新表；应保留原表或明确建立新研究版本。
3. 重新生成原互斥配对并核对保存的配对表。同人、同访视、同食物编号和整克舍入后的份量；不同日期；当地时刻圆周差 ≤1 小时；每餐最多使用一次。不同传感器连接的数量另行报告，不默认为零。
4. `all_descriptive` 使用全部合格配对；`primary` 要求原协变量完整并排除明确糖尿病/A10 标记为 1 者，允许标记未知；`strict` 进一步要求标记明确为 0。协变量源中不存在的参与者不能归入 primary。
5. 本问题不要求 NOVA 覆盖或 NOVA 个体内变化，人数不应强制等于第一轮 5,211 人。四个指标使用共同完整的餐对，缺失数写入流量表。

### 差异与区间

四个指标均用 mg/dL：餐前基线、餐后两小时平均绝对血糖、平均净增量、峰值升幅。平均净增量=原净 iAUC/120；平均绝对血糖=平均净增量+基线。差异方向为后一次减前一次。

按参与者整组 bootstrap，默认 1000 次，取 2.5% 和 97.5% 分位作为探索性区间。每次抽到一个人，带入其全部餐对；不把餐对当成独立受试者。至少 50 人才给区间，人数不足保留描述量并标记 DESCRIPTIVE_ONLY。

- `pair_median_absolute_difference`：按餐对计的绝对差中位数，与此前截图对应。
- `pair_p90_absolute_difference`：绝对差第 90 百分位；其置信区间和 P90 本身不同。
- `pair_mean_signed_difference`：餐对有符号差的均值。
- `person_mean_of_median_absolute_difference`：先算每人的绝对差中位数，再将各人的中位数等权平均。
- `person_mean_of_mean_signed_difference`：先算每人的有符号差均值，再按人等权平均。

完整食物模板表保留全部重复记录，按不同日期数列出 ≥2、≥3、≥4 日期的结构；没有施加上述两餐时刻卡钳，因此不能与互斥餐对混用分母。此版本正式估计的是重复差异及聚类区间，**没有计算 ICC、可靠性分类或个人稳定性分数**。约一半人只有一对，先弄清重复结构比直接输出一个混合模板 ICC 更可解释。

这些区间不包含日志误差、CGM 仪器误差或样本选择偏差；不证明因果。多个指标均保留，没有做显著性筛选或提供多重检验后的确认性结论。

### 后续日期验证

目标是：**已知参与者、已在训练中重复出现的食物模板，在后续日期的两小时平均绝对血糖**。食物签名来自整餐日志，所以是已知餐食记录条件下的回顾性验证，不是可实时部署的餐前工具，也不预测新参与者或未知食物模板。

- 每个人/访视按当地有效日期排序，前约 70% 日期训练、其余测试；至少 2 个训练日期和 1 个测试日期。同一天不拆散，切分不看结局值。各人的日历日期可能不同，是个人内前后日期验证，并非全队列共同日历切点。
- 所有模型使用共同特征完整样本；切分后要求每个食物模板在训练中出现至少 2 个不同日期，测试只保留这些已见模板。
- 三个嵌套模型：训练中同人同模板均值；再加当地时刻正余弦/前餐间隔（封顶12小时）；再加餐前基线/餐前趋势。实际餐次持续时间在进食开始时未知，未放入预测特征。
- 只用训练数据估计模板均值、模板内特征中心及尺度。固定 ridge 惩罚 10，不在测试集调参。无变化特征标明 CONSTANT_IN_TRAIN，系数为零。
- 附加简单对照 `premeal_persistence`：直接将餐前基线作为餐后均值预测值。
- 至少 50 名训练及测试参与者，否则 NOT_ESTIMATED，不自动放宽阈值。训练/测试及模板支持的损失见 04_prediction_flow.csv。
- 主要误差为每人 MAE 再等权平均；同时保留餐次加权 MAE、RMSE。报告相同测试餐上模型之间的每人等权 MAE 改善及参与者 bootstrap 区间。
- 预测区间以本次已训练模型为条件，没有每次重新训练；不代表全部模型开发过程的不确定性。改善为负也照常保留。

餐前信息可能只是在预测背景血糖的延续；不能把其预测增益解读为对“食物引起反应”的机制解释。没有同期事件数据就不声称已调整当天睡眠或活动。没有新增菌群筛选或因果中介分析。

## 文件与异常

先读 SUMMARY.md、tables/01_cohort_flow.csv、03_pair_differences.csv、04_prediction_status.csv、04_prediction_gains.csv。

`private/` 包含身份、完整模板、训练/测试分配和逐餐预测；保留服务器。汇报只需 SUMMARY.md、tables/ 和 figures/。manifest.json 包含输入路径与哈希，也优先留在服务器。

若原代码/餐次/协变量指纹不一致，程序失败而不改写原结果。若没有足够重复餐或预测支持，输出明确状态，不伪造零效应。中断或失败的目录保留用于排查；重新运行时另建新目录即可。
