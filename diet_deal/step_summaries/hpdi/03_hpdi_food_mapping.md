# 第 3 步：建立并冻结 hPDI 食物映射

## 目的与当前结论

本步骤使用 `Data/diet_deal/scripts/hpdi/03_prepare_hpdi_mapping.py`，将公共食物字典中的每个 `food_id` 对应到 hPDI 的 18 个食物组之一，或明确标记为不适用、原始标签缺失。

截至 2026-08-08，服务器最终映射已经完成：

- 待人工审核的 `review_required` 已清零；
- 论文规定的 18/18 个 hPDI 食物组均有映射记录；
- 最终映射覆盖 7,770 个唯一 `food_id` 和 2,640,959 条饮食事件；
- 可以进入第 4 步参与者摄入量汇总。

## 最终映射结果

| 映射状态 | food_id 数 | 事件数 | 全部事件占比 | 后续处理 |
|---|---:|---:|---:|---|
| `mapped` | 6,879 | 2,514,775 | 95.2220% | 进入相应 hPDI 食物组 |
| `not_applicable` | 523 | 114,242 | 4.3258% | 不进入18个食物组 |
| `unmapped_missing_labels` | 368 | 11,942 | 0.4522% | 无法归类，保留缺失状态 |
| `review_required` | 0 | 0 | 0% | 已全部处理完成 |
| **合计** | **7,770** | **2,640,959** | **100%** | |

`mapped + not_applicable` 共覆盖 99.5478% 的事件。剩余 0.4522% 不是尚未审核，而是原始记录本身没有足够身份信息。

## 18 个 hPDI 食物组

### 健康植物性食物：正向计分

1. `whole_grains`
2. `fruits`
3. `vegetables`
4. `nuts`
5. `legumes`
6. `vegetable_oils`
7. `tea_coffee`

### 较不健康植物性食物：反向计分

8. `fruit_juice`
9. `refined_grains`
10. `potatoes`
11. `sugar_sweetened_beverages`
12. `sweets_desserts`

### 动物性食物：反向计分

13. `animal_fat`
14. `dairy`
15. `eggs`
16. `fish_seafood`
17. `meat`
18. `miscellaneous_animal_foods`

## 为什么不能只按原始 `food_category` 直接分类

论文和补充材料定义的是18个标准食物组，但项目数据库中没有现成的 hPDI 对照表。原始 `food_category` 中还包含大量混合大类，例如 `Soups and sauces`、`Others`、`Snacks`、`Oils and fats`、`Canned veg and fruits`、`fruit juices and soft drinks` 和 `Industrialized vegetarian food ready to eat`。这些大类内部可以同时包含多个 hPDI 组或完全不属于 hPDI 的项目。

例如：

- `Snacks` 可能是坚果、全谷物爆米花、薯片、精制谷物脆片或甜点；
- `Soups and sauces` 可能是蔬菜汤、鸡汤、蛋黄酱、番茄酱、植物油酱汁或只作为调味品使用；
- `fruit juices and soft drinks` 可能是100%果汁、含糖饮料、无糖饮料、咖啡或茶；
- 工业化素食可能是豆类/蔬菜，也可能是无法按配料拆分的仿肉复合食品。

因此最终映射结合了原始类别、短名称和商品名称；只有稳定且能够对应论文定义的项目才进入18组。

## 前期自动规则如何处理

映射脚本经历了多轮规则校正。早期版本先进行类别级映射，之后逐步加入名称级高可信规则和误匹配保护。主要处理包括：

1. 明确类别直接映射，例如水果、蔬菜、乳制品、鸡蛋、鱼类等。
2. 对混合类别使用具体名称判断，而不是把整个大类强制归入同一组。
3. 将 `Processed meat products` 强制归入 `meat`。最终共有 105 个加工肉 `food_id`、15,088 条事件使用该规则。
4. 根据补充材料 Table 11 修正薯片和玉米片：potato chips、corn chips、tortilla chips、Doritos、Cheetos 和 Apropo 归入 `potatoes`；爆米花仍归 `whole_grains`。
5. 对 pizza、chowder/cream soup、独立 mayonnaise 和 creamy salad dressing 等符合定义的复合食品，保守归入 `miscellaneous_animal_foods`。
6. 只有商品名称足够具体时才从商品名回退判断；短名称存在时优先使用短名称。
7. 对鸡蛋关键词使用完整单词匹配，避免把 `eggplant`、`veggie` 等误识别为 `eggs`。
8. 补充 cranberry juice cocktail、lemon nectar、potato chips、vinaigrette、black olives、kimchi、pickled ginger、wasabi peas、chicken nuggets 等高可信名称规则。
9. 肌酸、维生素、矿物质、胶原粉、无法确定主要食物组的蛋白粉、添加剂和其他无可靠 hPDI 对应关系的项目标记为 `not_applicable`，不强行归类。

## 各版本审核量变化

以下数字来自服务器每轮 `03_prepare_hpdi_mapping.py` 的 QC 输出。早期版本属于规则开发过程，数量变化反映了错误纠正和分类边界调整，并非简单把待审核项逐条填满。

| 版本 | mapped food_id | not_applicable | review_required | missing labels |
|---|---:|---:|---:|---:|
| v5 | 6,376 | 189 | 837 | 368 |
| v6 | 6,364 | 189 | 849 | 368 |
| v7 | 6,414 | 211 | 777 | 368 |
| v8 | 6,433 | 242 | 727 | 368 |
| v9 | 6,476 | 263 | 663 | 368 |
| v10 | 6,498 | 276 | 628 | 368 |
| v11 | 6,539 | 294 | 569 | 368 |
| v12，人工审核前 | 6,541 | 294 | 567 | 368 |
| v12，最终冻结 | 6,879 | 523 | 0 | 368 |

## 567 个待审核 food_id 如何完成

待审核清单按事件数从高到低完整输出，字段包括：

- `food_id`；
- 事件数；
- 自动建议组件；
- 原始 `food_category`；
- `short_food_name`；
- `product_name`。

在查看全部 567 个项目后，使用 `scripts/hpdi/03b_finalize_hpdi_review.py` 生成最终人工覆盖和审核记录。脚本首次运行曾在 `if invalid.any()` 处遇到 pandas Series 真值歧义错误；修正为单一布尔条件后重新运行成功。

最终人工决策为：

| 人工决策 | food_id 数 | 事件数 |
|---|---:|---:|
| 映射至18个组件之一 | 338 | 8,674 |
| 标记为 `not_applicable` | 229 | 6,797 |
| **合计** | **567** | **15,471** |

其中 63 个中等可信度项目、1,862 条事件因没有唯一且稳定的 Table 11 食物组对应关系而采用保守排除，没有为了消除缺失而强行分类。

人工决策写入：

- `outputs/03_score_mapping/hpdi/hpdi_component_overrides.csv`：最终覆盖表；
- `outputs/03_score_mapping/hpdi/hpdi_review_resolution_audit.csv`：审核审计表。

随后重新运行 `03_prepare_hpdi_mapping.py`，567 个 `review_required` 全部消失，映射表正式冻结。

## 368 个 `unmapped_missing_labels` 是什么

这 368 个 `food_id` 并不是没有 ID。它们有 `food_id`，而且共有 11,942 条事件，但食物身份字段缺失：

- `short_food_name` 缺失；
- `product_name` 缺失；
- `food_category` 也无法提供分类信息。

已经检查过以下最原始的 Transfer 文件：

- `Data/Transfer/diet_logging/raw_diet_logging_events.csv`；
- `Data/Transfer/diet_logging/diet_logging_events.csv`。

原始文件中这些 ID 同样没有名称或类别；不是公共食物字典合并时把名称弄丢，也不能从另一张 Transfer 表恢复。部分记录仍有重量、能量和营养素数值，但仅凭这些数值不能可靠反推出具体食物，所以不能主观放入任何 hPDI 组件。

当前处理决定是：

1. 不把这些事件归入18个 hPDI 食物组；
2. 不把它们改成 `not_applicable`，继续保留明确的 `unmapped_missing_labels` 状态；
3. 不因此删除整名参与者；
4. 第4步仍将这些事件的可用能量计入参与者总能量代理，但不会计入任何组件克数；
5. 在结果限制中报告其占全部事件 0.4522%，并保留后续敏感性分析的可能性。

这种处理可能使少数参与者的某些 hPDI 组件摄入量略低估，但比依据营养素猜测食物类别更可审计。

## `not_applicable` 的含义

`not_applicable` 不是数据缺失，也不是仍需审核，而是有足够信息判断该条目不应进入 hPDI 18组，或无法按论文定义把复合食品可靠分给某一个组。它们不计入组件克数，但可用的能量仍属于参与者总能量摄入。

最终 523 个 `not_applicable` 包括：

- 自动规则已明确排除的项目；
- 567 个审核项目中人工判为不适用的 229 个；
- 其中一部分是补充剂、添加剂、调味粉、蛋白粉或无法稳定拆分的复合食品。

## 最终输出文件

第3步输出位于 `outputs/03_score_mapping/hpdi/`：

- `hpdi_component_definitions.csv`：18个组件、三大类和计分方向；
- `hpdi_food_id_mapping.csv`：最终 food_id 映射；
- `hpdi_mapping_qc.csv`：映射状态、事件覆盖率和组件分布；
- `hpdi_review_candidates.csv`：最终为空；
- `hpdi_component_review_template.csv`：人工审核模板；
- `hpdi_component_overrides.csv`：人工覆盖决定；
- `hpdi_review_resolution_audit.csv`：人工审核审计记录。

## 当前参与者范围决定

hPDI 不包含酒精组件，因此当前计划在全部 9,737 名基线饮食参与者中计算，不沿用 AMED 因酒精不完整而只剩 7,192 人的范围。

论文主分析报告 9,616 名饮食参与者，比当前 9,737 人少 121 人。论文正文和补充材料尚未给出能够完全复现这 121 人排除的明确逐条条件。当前决定是先对 9,737 人计算并完整保留 QC，后续若确认论文额外筛选条件，再从参与者级结果细化到相同样本。

## 下一步

映射步骤已经结束，不再继续修改食物规则。下一步依次为：

1. 将新版 `04_build_hpdi_intakes.py` 同步到服务器，运行第4步，确认参与者数、18组摄入完整性和能量覆盖率；
2. 将新版 `05_calculate_hpdi_scores.py` 同步到服务器，计算18项五分位、18–90原始 hPDI、0.5%/99.5%温莎化分数和总能量残差校正分数；
3. 对照论文 hPDI 分布、均值和五分位人数进行 QC。

论文补充材料明确规定18组摄入量按克/日五分位计分；主文又将五种饮食模式称为能量校正分数。论文引用的 Wang 等 2023 方法是先获得饮食模式总分，再把总分对总能量摄入做残差法校正。因此新版脚本会同时保留原始分和能量校正分，避免丢失可审计的中间结果。

## 方法来源

- 项目论文：`2026.05.04.26352208v1.full.pdf`。
- Wang P, et al. *Optimal dietary patterns for prevention of chronic disease*. Nature Medicine. 2023;29:719–728. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10294543/>

