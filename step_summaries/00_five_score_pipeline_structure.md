# 五种饮食评分的数据处理结构

本研究复现论文中的五种饮食评分：AMED、AHEI、hPDI、rDII 和 rEDIH。为避免重复处理和文件混杂，饮食流程分为“公共数据层”和“评分专属层”。

各评分的独立方法、处理状态和限制分别记录在 `amed/`、`ahei/`、`hpdi/`、`rdii/` 和 `redih/`；公共数据步骤记录在 `shared/`。总索引见 [`README.md`](README.md)。

## 公共数据层

### 01_daily_summary

按参与者和日期汇总原始饮食记录，生成日级营养摄入、参与者记录天数统计和数据质量控制结果。该结果可以被五种评分共同使用。

### 02_food_dictionary

以 `food_id` 为单位整理食物名称、产品名称和原始食物类别，并保存人工修订和质量控制结果。该字典是 AMED、AHEI、hPDI 和 rEDIH 食物分组的共同基础；rDII 主要使用营养素摄入，但仍可用该字典辅助追溯异常记录。

## 评分专属层

每个处理阶段均按 `amed`、`ahei`、`hpdi`、`rdii` 和 `redih` 分开保存：

- `03_score_mapping/<score>/`：保存该评分的成分定义、食物或营养素映射和映射 QC。
- `04_score_intakes/<score>/`：保存该评分的日级及参与者级成分摄入量和摄入量 QC。
- `05_diet_scores/<score>/`：保存各成分得分、总分、阈值和评分 QC。

## 脚本目录

- `scripts/shared/`：保存五种评分共同使用的日级汇总和食物字典脚本。
- `scripts/amed/`、`scripts/ahei/`、`scripts/hpdi/`、`scripts/rdii/` 和 `scripts/redih/`：分别保存每种评分自己的映射、摄入量汇总和计分脚本。

不同评分只共用公共输入表，不共用评分映射结果。每种评分的映射规则和输出文件均在自己的目录中维护。

## 五种评分分类

- 膳食模式或指南型评分：AMED、AHEI、hPDI。
- 炎症或代谢机制型评分：rDII、rEDIH。

AMED 已完成。下一项建议优先处理 hPDI，因为它主要依赖现有食物字典和食物类别，不需要先解决 AMED 酒精缺失，也比需要更多营养素字段的 rDII 更适合当前数据条件。
