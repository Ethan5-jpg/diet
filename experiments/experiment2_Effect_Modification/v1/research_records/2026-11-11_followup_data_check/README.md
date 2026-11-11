# 后续实验数据核对

## 第三轮核对（通过）

把 [server_daily_link_check.sh](server_daily_link_check.sh) 全文粘贴到 HPP 服务器终端，回传从 `EXP2_DAILY_LINK` 到 `EXP2_DAILY_LINK_DONE` 或 `EXP2_DAILY_LINK_STOP` 的截图。命令保留数值身份键，按同一参与者、访视和日期关联，并核对冻结人表的纤维天数和均值。结果包括共同有效日、纤维日间变异、W20 共同日的重叠以及菌群登记日期间隔。已在本地模拟数据中重现上一条命令的故障，并验证了新命令的无损匹配、核对失败及重复日处理。

服务器已完成核对，恢复了冻结人群内 W20 的 21,262 个共同日。 [第三轮截图与结果](2026-11-11_174005_daily_link_verified/record.md)

## 第二轮核对

[诊断指令](server_daily_diagnosis.sh) 核对了交接表行数、身份键匹配和有效性筛选。服务器发现原先的文本 ID 匹配没有接上。文件行数及错误记录见 [第二轮截图与解读](2026-11-11_173123_id_mismatch/record.md)。

## 首轮核对

把 [server_data_check.sh](server_data_check.sh) 全文粘贴进 HPP 服务器终端。命令复用 V12 的解释器，从已经完成的 101 条路线运行核对 3,343 名参与者的行索引，再读取现有表做汇总。

- `DAILY` 和 `FIBER_WITHIN` 汇报同一人、同一访视、同一日的饮食与 CGM 重叠，以及至少三个共同有效日者的纤维密度人内变异。
- `SAMPLE_DATES` 按口腔和肠道分别汇报锁定样本的日期覆盖、与共同有效日的绝对间隔。7 天和 30 天是描述性分组。登记日期的采样语义仍待核实。
- `MEAL` 核查 2026-09-24 历史构建表在当前发现组中的合格餐次、纤维完整性及进餐时间标记。找不到时会报告已知路径未找到。

服务器只需回传从 `EXP2_DATA_CHECK` 到 `EXP2_DATA_CHECK_DONE` 的终端截图。命令只输出汇总，所有关联键留在服务器。它在 PyArrow 15.0.2 的本地生成数据上通过了同访视匹配、重复日期拒绝、CGM 有效性字段可选、菌群日期缺失和旧餐次表缺失的检查。

历史日级分析已见 [W20 日级报告](../../../../../HPP_Full_Code_With_Models_20261102_v2/04_DOCUMENTATION/reports/HPP_W20_Targeted_V3_Daily_CGM_2026-11-01.md)。本次核对的分母为已完成的 Experiment 2 人群。

[首轮截图与解读](2026-11-11_172242_first_response/record.md)
