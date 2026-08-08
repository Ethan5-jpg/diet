# AHEI 复现计划与综述

## 当前状态

AHEI 尚未开始专属映射和评分。公共日级饮食汇总、食物字典和 population 性别表可以沿用，但 AMED 映射结果不能直接用于 AHEI。

## HPP 可用组成部分

标准 AHEI 有 11 项。HPP 缺少长链 n-3 脂肪酸 EPA + DHA，因此计划使用 10 项：蔬菜、水果、全谷物、含糖饮料和果汁、坚果和豆类、红肉和加工肉、反式脂肪、PUFA、钠和酒精。每项连续计 0–10 分。

## 计分要点

- 蔬菜、水果、全谷物、坚果和豆类、PUFA 为正向计分。
- 含糖饮料和果汁、红肉和加工肉、反式脂肪、钠为反向计分。
- 全谷物和酒精存在性别特异阈值，需要使用 `population.csv`。
- 食物类成分需要 servings/day 或明确的克数换算；PUFA 和反式脂肪需要相应营养字段及能量占比。

## 当前数据限制

- 当前饮食事件表没有直接提供反式脂肪和 PUFA 字段。
- 当前没有统一的食物标准份量换算表。
- 在补齐以上信息前，可以先建立食物类成分映射，但不能声称完成全部 10 项 AHEI 的严格复现。

## 计划文件

- `Data/diet_deal/scripts/ahei/03_prepare_ahei_mapping.py`
- `Data/diet_deal/scripts/ahei/04_build_ahei_intakes.py`
- `Data/diet_deal/scripts/ahei/05_calculate_ahei_scores.py`
- 对应结果分别写入 `outputs/03_score_mapping/ahei/`、`outputs/04_score_intakes/ahei/` 和 `outputs/05_diet_scores/ahei/`。
