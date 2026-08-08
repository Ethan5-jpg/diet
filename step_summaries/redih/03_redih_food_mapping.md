# 第3步：rEDIH food_id 映射

## 状态

本地脚本已经完成并通过合成测试；服务器全量映射和人工审核尚未运行。

脚本：`Data/diet_deal/scripts/redih/03_prepare_redih_mapping.py`

版本：`2026-08-08-redih-mapping-v2`

## 输入

- `outputs/02_food_dictionary/food_id_dictionary.csv`
- `outputs/02_food_dictionary/food_category_overrides.csv`
- `outputs/03_score_mapping/redih/redih_component_overrides.csv`，首次运行后创建，后续由人工维护

脚本复用公共 food_id 字典和公共类别修订，但不读取 AMED 或 hPDI 的评分专属映射。

## 映射定义

定义表保存全部18个标准 EDIH 食物组及论文权重。margarine 标记为 HPP 不可用且不计分；wine 与其余16组均标记为 HPP 可映射并进入当前17组件评分。

自动映射只处理高置信度规则，例如明确的 red meat、processed meat、poultry、eggs、whole fruits、低能量饮料，以及名称明确的 coffee、wine、cream soup、French fries、butter、tomatoes 和部分 other fish。以下情况保守进入 `review_required`：

- 低脂或高脂等级不能确定的乳制品；
- other fish 与深色鱼/复合鱼类菜品边界不清；
- 饮料是否低能量或高能量不清；
- 番茄或绿叶蔬菜只作为复合菜的一部分；
- 宽泛类别中的肉类、咖啡、奶油汤或其他候选。

margarine 名称强制标记为 `not_applicable`，并记录 HPP 未测量的原因。短名称、商品名和最终类别全部缺失的 food_id 单独标记为 `unmapped_missing_labels`。

## 输出

- `redih_component_definitions.csv`
- `redih_food_id_mapping.csv`
- `redih_mapping_status_summary.csv`
- `redih_component_summary.csv`
- `redih_review_candidates.csv`
- `redih_component_overrides.csv`
- `redih_mapping_qc.csv`

## 人工 override

override 的必要字段为：

- `food_id`
- `mapping_status`
- `redih_component`
- `review_notes`

已确认食物使用 `mapping_status=mapped` 并填写17个 HPP 可用组件之一；不适用食物使用 `not_applicable` 且组件留空。脚本拒绝重复 food_id、无效状态、margarine 作为 mapped 组件、非 mapped 行填写组件以及字典外 ID。

## 服务器运行与完成标准

```bash
grep 'SCRIPT_VERSION' scripts/redih/03_prepare_redih_mapping.py
python3 scripts/redih/03_prepare_redih_mapping.py 2>&1 | tee /tmp/redih_03_v2.log
```

服务器运行后需要记录7,770个 food_id 的状态和事件覆盖率。只有 `review_required=0`、映射状态完整分区且17个可用组均得到合理处理后，才可冻结映射并进入第4步。目前这些全量 QC 数字仍待服务器结果，不能从 hPDI 复制。
