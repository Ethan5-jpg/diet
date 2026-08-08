# hPDI 复现计划与综述

## 当前状态

hPDI 的第3步食物映射已经完成并冻结。最终覆盖 7,770 个 `food_id` 和 2,640,959 条事件：6,879 个已映射，覆盖 95.2220% 的事件；523 个明确不适用，覆盖 4.3258%；368 个因原始名称和类别全部缺失而无法映射，覆盖 0.4522%；`review_required` 已清零，18/18 个组件均已出现。

567 个最终待审核 `food_id` 已通过 `03b_finalize_hpdi_review.py` 全部处理，其中338个映射至组件、229个保守标记为不适用。详细规则、版本变化、缺失来源和审核记录见 [`03_hpdi_food_mapping.md`](03_hpdi_food_mapping.md)。

第4步v3已在服务器真实数据上成功运行并通过最终双轨QC：2,059,935条基线目标事件汇总为125,374条日级记录和9,737名参与者；9,737人的18项摄入代理全部可用且平均每日能量均大于0。共有476条已归类事件缺少重量（占已归类事件0.024301%），影响359人；9,378人18组事件重量严格完整。正式采用双轨记录：主分析用可观测重量摄入代理保留9,737人，同时保留严格完整标志供9,378人敏感性分析。

第5步v3已在服务器真实数据上成功运行并通过最终QC。9,737人均获得原始和能量校正hPDI；原始分范围27–82，均值54.055、标准差7.873；温莎化范围35–75；能量校正分范围34.083–79.646，均值54.057、标准差7.684。能量校正五分位人数依次为1,948、1,947、1,948、1,947、1,947。当前9,737人hPDI主分析评分已完成。

分步骤说明见 [`04_hpdi_component_intakes.md`](04_hpdi_component_intakes.md) 和 [`05_hpdi_scores.md`](05_hpdi_scores.md)。

hPDI 不涉及酒精，当前已在全部9,737名基线饮食参与者中完成计算。论文报告9,616人，额外排除的121人条件仍待确认，后续可在参与者级结果上细化并重新评分。

以下内容保留映射设计和评分结构说明。

v11 根据原始 hPDI 定义，将具体名称为 pizza、chowder/cream soup、独立 mayonnaise 或 creamy salad dressing 的条目归入 `miscellaneous_animal_foods`。该规则只按具体名称识别，不把 `Others`、`Soups and sauces` 等混合大类整体归入，也不会把 tuna salad、egg salad、potato salad 等仅因名称含 mayonnaise 就整份改为第18组。v11 继续对原规则保留的 `review_required` 使用具体名称补充高可信映射，例如 cranberry juice cocktail、lemon nectar、potato chips、vinaigrette、black olives、kimchi、pickled ginger、wasabi peas 和 chicken nuggets；肌酸、仿肉制品、调味粉及其他无法对应18组的项目则标为不适用。对短名称信息不足的项目优先使用限定商品名规则。短名称为空时一般规则才回退到商品名，从而处理缺短名称的食物，同时限制误匹配范围。规则不会覆盖已有可靠类别映射。鸡蛋候选词使用完整单词匹配，避免 `Eggplant` 和 `Veggie` 被误提示为 `eggs`。脚本同时输出逐个 `food_id` 的审核明细和模板，并支持人工组件覆盖表。

v12 按论文补充材料 Table 11 的示例定义修正薯片和玉米片：potato chips、corn chips、tortilla chips、Doritos、Cheetos 和 Apropo 统一进入 `potatoes`，而不是 `refined_grains`。该规则在宽泛的原始数据库类别规则之后执行，避免同一种玉米片因被标成 Bread、Cereals 或 Snacks 而进入不同组件；popcorn 仍为 `whole_grains`，普通 corn 和 sweet potato/yam 仍为 `vegetables`，pretzels、croutons 等仍为 `refined_grains`。

## 18 个食物组

### 健康植物性食物

全谷物、水果、蔬菜、坚果、豆类、植物油、茶和咖啡，共 7 组。

### 较不健康植物性食物

果汁、精制谷物、土豆、含糖饮料、甜食和甜点，共 5 组。

### 动物性食物

动物脂肪、乳制品、鸡蛋、鱼和海鲜、肉类、其他动物性食物，共 6 组。

## 计分方法

每个食物组按队列摄入量五分位数计 1–5 分。健康植物性食物正向计分；较不健康植物性食物和动物性食物反向计分。18 项相加，理论范围为 18–90 分。

## 第一步：食物映射

`03_prepare_hpdi_mapping.py` 将每个 `food_id` 映射至18个 hPDI 食物组之一。明确类别直接映射；饮料、罐装果蔬、油脂和谷物等混合类别结合食物名称进行保守判断；无法可靠拆分的复合食物保留为 `review_required`。脚本同时保留 `mapped`、`not_applicable` 和 `unmapped_missing_labels` 状态，避免把不确定食物强行归类。

论文及补充材料规定了18个组件、代表性食物和计分方向，但没有提供本项目数据库全部 7,770 个 `food_id` 的现成对照表。因此 food_id 级映射需要在论文定义约束下根据本数据库类别和名称建立，并保留 QC、待审核状态和人工覆盖记录，不能声称逐条分类直接来自论文。

脚本生成：

- `hpdi_component_definitions.csv`：18个食物组、三大类型和五分位计分方向；
- `hpdi_food_id_mapping.csv`：每个 `food_id` 的类别、映射结果、候选结果和判断依据；
- `hpdi_mapping_qc.csv`：映射状态、组件覆盖和待审核类别汇总。

上述结果写入 `outputs/03_score_mapping/hpdi/`。

## 第二步：18组摄入量

`04_build_hpdi_intakes.py` 分块读取全量饮食事件，按参与者实际记录日生成日级及参与者级18组平均每日克数代理。脚本会拒绝仍含 `review_required` 的映射、缺少任一组件的映射、重复 `food_id`、映射表外事件，以及组件内缺失或负的食物重量。

输出：

- `hpdi_daily_component_intakes.csv`；
- `hpdi_participant_component_intakes.csv`；
- `hpdi_intake_qc.csv`。

## 第三步：五分位评分

`05_calculate_hpdi_scores.py` 使用平均秩生成并列值安全的队列五分位：相同摄入量不会被拆到不同组。健康植物食物正向计 1–5 分，其余食物组反向计分。只有18项全部非缺失时才计算总分，并硬性检查组件分为 1–5、总分为 18–90。

输出：

- `hpdi_quintile_thresholds.csv`；
- `hpdi_participant_scores.csv`；
- `hpdi_score_qc.csv`。

第3步映射、第4步18组摄入汇总和第5步主分析评分均已在服务器完成。后续可使用能量校正连续分或五分位进入统计分析；仍待完成的可选工作是9,378人严格重量完整敏感性评分，以及在明确论文额外121人排除条件后重建9,616人分析样本。

## 计划文件

- `Data/diet_deal/scripts/hpdi/03_prepare_hpdi_mapping.py`
- `Data/diet_deal/scripts/hpdi/04_build_hpdi_intakes.py`
- `Data/diet_deal/scripts/hpdi/05_calculate_hpdi_scores.py`
- 对应结果分别写入 `outputs/03_score_mapping/hpdi/`、`outputs/04_score_intakes/hpdi/` 和 `outputs/05_diet_scores/hpdi/`。
