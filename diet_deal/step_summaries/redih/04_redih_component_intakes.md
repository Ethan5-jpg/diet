# 第4步：rEDIH 组件 g/day 摄入量

## 状态

脚本已在本地完成并通过合成测试；必须等服务器第3步 `review_required=0` 后运行。

脚本：`Data/diet_deal/scripts/redih/04_build_redih_intakes.py`

版本：`2026-08-08-redih-intakes-v2`

## 汇总方法

1. 分块读取 `diet_logging_events.csv`。
2. 固定筛选 `cohort=10k` 和 `research_stage=00_00_visit`。
3. 按参与者、记录日和日期汇总17个 HPP 可映射组件的 `weight_g`。
4. 按参与者实际观察日计算平均每日克数；未摄入某组的观察日为0克。
5. wine 进入摄入输出，并以论文权重 `-0.165` 进入第5步的17组件加权和。
6. 已映射事件缺 `weight_g` 时不插补、不虚构为已知0；主分析只累加其余可观测重量并保留原观察日分母，同时保存缺重量事件数和严格完整标志。

## 硬性检查

- 映射表不得包含 `review_required`、重复 food_id 或无效组件。
- 目标事件中的非缺失 food_id 必须全部出现在冻结映射表。
- 事件状态必须完整分为 mapped、not applicable、标签全缺失、映射外 ID 和缺 food_id。
- 17项参与者级平均每日克数必须非缺失且非负。
- 输出17个当前计分组的总体重量完整标志，并保留每组缺重量事件审计列。

## 输出

- `redih_daily_component_intakes.csv`
- `redih_participant_component_intakes.csv`
- `redih_intake_qc.csv`

## 服务器运行

映射冻结后：

```bash
grep 'SCRIPT_VERSION' scripts/redih/04_build_redih_intakes.py
python3 scripts/redih/04_build_redih_intakes.py 2>&1 | tee /tmp/redih_04_v2.log
```

服务器需要核对目标事件数、125,374条预期基线参与者日附近的结果、9,737名预期参与者、各组件事件数、缺重量事件数、受影响参与者和严格完整人数。最终数字以本次 rEDIH 映射下重新计算的结果为准。
