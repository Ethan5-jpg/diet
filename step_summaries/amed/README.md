# AMED 复现综述

## 当前状态

AMED 的食物映射、参与者摄入量汇总和八项评分框架已经完成。七项食物成分可在 9,737 名饮食参与者中评分；酒精记录完整且可计算八项总分的参与者为 7,192 人。酒精缺失问题暂时保留，后续单独处理。

## HPP 可用组成部分

标准 AMED 有 9 项。HPP 缺少单不饱和脂肪与饱和脂肪比，因此使用水果、蔬菜（不含土豆）、全谷物、坚果、豆类、鱼类、红肉及加工肉和酒精共 8 项。

## 已完成步骤

1. [`03_amed_food_mapping.md`](03_amed_food_mapping.md)：建立 AMED 专属 `food_id` 映射。
2. [`04_amed_component_intakes.md`](04_amed_component_intakes.md)：计算参与者日级和平均每日成分摄入量。
3. [`05_amed_scores.md`](05_amed_scores.md)：计算七项食物分、性别特异酒精分和 0–8 分总分。

## 对应脚本与结果

- 脚本：`Data/diet_deal/scripts/amed/`。
- 映射：`Data/diet_deal/outputs/03_score_mapping/amed/`。
- 摄入量：`Data/diet_deal/outputs/04_score_intakes/amed/`。
- 评分：`Data/diet_deal/outputs/05_diet_scores/amed/`。

## 待处理问题

- 2,545 名参与者存在无法可靠解析的酒精饮料记录，AMED 总分保留缺失。
- 当前食物摄入量使用 `weight_g` 的克/日代理值，尚未转换为论文未公开的标准份量。
- 当前饮食参与者为 9,737 人，比论文报告的 9,616 人多 121 人，需要后续核对额外饮食 QC 条件。
