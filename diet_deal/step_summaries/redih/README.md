# rEDIH 复现计划与当前状态

## 当前状态

rEDIH 已完成本地方法审计和第3至第5步脚本实现，但服务器食物映射尚未运行，因此还没有全量分数结果。

当前脚本版本：

- `03_prepare_redih_mapping.py`：`2026-08-08-redih-mapping-v2`
- `04_build_redih_intakes.py`：`2026-08-08-redih-intakes-v2`
- `05_calculate_redih_scores.py`：`2026-08-08-redih-scores-v2`

本地已通过11个合成与文件级单元测试、3个脚本的编译和命令行检查，以及9,737行压力检查。服务器必须先运行第3步并完成所有人工审核，才能运行第4和第5步。

## 当前项目决定

- 食物组摄入使用论文补充方法文字所写的 `g/day`。
- 标准 EDIH 有18组；HPP未测量 margarine，因此可映射17组。
- wine 使用论文权重 `-0.165` 进入17组件 EDIH；margarine 是唯一因 HPP 未测量而排除的标准组。
- 全部9,737名基线参与者保留在输出中；仅 `alcohol_complete=true` 者计算正式评分，当前预期约7,192人，其余评分留空。
- 以后补齐酒精缺失后必须重新运行第5步，并在更新后的合格样本内重算温莎界值、能量校正和五分位。
- `EDIH = Σ(组件平均每日克数 × 论文权重)`；`rEDIH = -EDIH`。

## 17个 HPP 可映射组

17个计分组为 red meat、low-energy beverages、cream soups、processed meat、poultry、butter、French fries、other fish、high-energy drinks、tomatoes、low-fat dairy、eggs、wine、coffee、whole fruits、high-fat dairy 和 green leafy vegetables。wine 的 `included_in_current_score=True`；margarine 保留在标准定义表中并标记 `available_in_hpp=False`。

## 三步脚本

1. `Data/diet_deal/scripts/redih/03_prepare_redih_mapping.py`：从公共字典建立独立 rEDIH 映射、审核候选、override 模板和映射 QC。
2. `Data/diet_deal/scripts/redih/04_build_redih_intakes.py`：分块读取基线事件，汇总17组日级和参与者级 g/day，保留 wine 和重量完整性标志。
3. `Data/diet_deal/scripts/redih/05_calculate_redih_scores.py`：合并 AMED 第4步的酒精完整性标记，对完整者计算17组件原始 EDIH/rEDIH、0.5/99.5百分位温莎化、能量残差校正和五分位，不完整者留空。

详细说明：

- [`03_redih_food_mapping.md`](03_redih_food_mapping.md)
- [`04_redih_component_intakes.md`](04_redih_component_intakes.md)
- [`05_redih_scores.md`](05_redih_scores.md)

## 下一步

在服务器确认脚本版本并运行第3步：

```bash
conda activate pheno
cd /home/ec2-user/Desktop/HPP/Data/diet_deal
grep 'SCRIPT_VERSION' scripts/redih/03_prepare_redih_mapping.py
python3 scripts/redih/03_prepare_redih_mapping.py 2>&1 | tee /tmp/redih_03_v2.log
```

首次运行会创建 `outputs/03_score_mapping/redih/redih_component_overrides.csv`。根据 `redih_review_candidates.csv` 审核并填写 override，重复运行，直到 `review_required=0`。在此之前第4步会主动拒绝运行。

## 当前限制

补充方法写食物组使用 g/day，但补充权重表把系数标为 `weight per serving`。本项目根据当前明确决定采用 g/day，并在参数和输出列名中保留这一事实；它属于论文公开文字口径的复现实现，不应声称等同于作者未公开代码。
