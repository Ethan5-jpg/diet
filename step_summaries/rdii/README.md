# rDII 复现计划与综述

## 当前状态

rDII 尚未开始计算。现有日级营养汇总只能提供部分宏量营养素，严格复现前需要补齐微量营养素数据、全球参考均值和标准差以及各成分的炎症效应分数。

## HPP 可用组成部分

论文补充材料显示 HPP 可使用 21 项：酒精、维生素 B12、维生素 B6、碳水化合物、胆固醇、能量、总脂肪、膳食纤维、铁、镁、MUFA、烟酸、蛋白质、PUFA、核黄素、饱和脂肪、硫胺素、维生素 A、维生素 C、维生素 E 和锌。

## 计算框架

1. 使用全球参考均值和标准差计算每项摄入量的 Z-score。
2. 将 Z-score 转为百分位数，再计算 `2 × percentile - 1`。
3. 乘以对应 inflammatory effect score 后求和得到 DII。
4. 按论文方向反转为 rDII，使高值代表更健康、抗炎的饮食方向。

## 当前数据限制

当前饮食事件表仅直接包含能量、碳水化合物、总脂肪、蛋白质、钠、酒精和膳食纤维，无法覆盖论文所用的 21 项。开始计算前必须查找 HPP 的扩展营养素来源或外部食物营养数据库，并确认 global reference 参数和反向编码公式。

## 计划文件

- `Data/diet_deal/scripts/rdii/03_prepare_rdii_mapping.py`
- `Data/diet_deal/scripts/rdii/04_build_rdii_intakes.py`
- `Data/diet_deal/scripts/rdii/05_calculate_rdii_scores.py`
- 对应结果分别写入 `outputs/03_score_mapping/rdii/`、`outputs/04_score_intakes/rdii/` 和 `outputs/05_diet_scores/rdii/`。
