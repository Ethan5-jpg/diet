# 第二步：构建并完善食物字典

使用 `scripts/shared/02_build_food_dictionary.py` 将重复出现的饮食事件按照 `food_id` 汇总，建立唯一食物编号与标准食物名称、产品名称和食物类别之间的对应关系；随后使用 `scripts/shared/02_finalize_food_dictionary.py` 对可根据名称判断的缺失类别进行人工审核和补充。

## 生成文件

1. `food_id_dictionary.csv`：每个 `food_id` 一行，包含标准名称、产品名称、原始食物类别、出现次数、累计重量及名称冲突标记。
2. `food_category_summary.csv`：汇总 34 个原始食物类别的事件数、唯一食物数量、累计重量和事件占比。
3. `food_dictionary_qc.csv`：记录唯一 `food_id` 数量、名称缺失、类别缺失和映射完整性等质量控制结果。
4. `food_id_conflicts.csv`：记录同一 `food_id` 对应多个名称或类别的情况；当前结果为空，说明未发现冲突。
5. `food_category_overrides.csv`：保存 12 个缺少类别但可根据食物名称确认类别的 `food_id` 及人工审核结果，共覆盖 3,382 条饮食事件。

## 处理结果

最终建立了 7,770 个唯一食物编号，食物类别覆盖率由 99.42% 提高至 99.548%。剩余 368 个完全缺少名称和类别的 `food_id` 暂时标记为未映射。

本步骤仅完成食物字典构建和食物分类准备，尚未计算任何饮食评分。
