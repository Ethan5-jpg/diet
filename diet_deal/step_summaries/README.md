# 饮食评分复现综述目录

本目录记录饮食数据公共处理和五种饮食评分的独立复现过程。每完成一个步骤，都应在对应评分目录中补充处理方法、生成文件、关键 QC、当前限制和下一步。

最新新对话交接文档：[`HPP_新对话交接说明_2026-08-08_hPDI完成.md`](../HPP_新对话交接说明_2026-08-08_hPDI完成.md)。

## 公共步骤

- [`shared/01_daily_diet_summary.md`](shared/01_daily_diet_summary.md)：日级和参与者级饮食汇总。
- [`shared/02_food_dictionary.md`](shared/02_food_dictionary.md)：唯一 `food_id` 字典和人工类别修订。

## 五种评分

- [`amed/README.md`](amed/README.md)：AMED，当前已完成方法框架复现。
- [`ahei/README.md`](ahei/README.md)：AHEI，等待食物份量和脂肪类型字段确认。
- [`hpdi/README.md`](hpdi/README.md)：hPDI，第3步食物映射、第4步18组摄入汇总和第5步9,737人主分析评分均已完成。
- [`hpdi/03_hpdi_food_mapping.md`](hpdi/03_hpdi_food_mapping.md)：hPDI 最终映射、567项人工审核、368项原始标签缺失及覆盖率。
- [`hpdi/04_hpdi_component_intakes.md`](hpdi/04_hpdi_component_intakes.md)：hPDI 18组摄入和总能量汇总，最终双轨QC已通过；主分析9,737人，严格重量完整敏感性样本9,378人。
- [`hpdi/05_hpdi_scores.md`](hpdi/05_hpdi_scores.md)：hPDI 原始分、温莎化和能量残差校正方法及最终分布；9,737人均有能量校正分。
- [`rdii/README.md`](rdii/README.md)：rDII，等待微量营养素和全球参考参数。
- [`redih/README.md`](redih/README.md)：rEDIH 使用 g/day；wine 纳入17组件评分，酒精不完整者保留记录但评分留空。第3至第5步 v2 脚本和本地测试已完成，等待服务器运行映射、审核和全量评分。

总体目录和数据流见 [`00_five_score_pipeline_structure.md`](00_five_score_pipeline_structure.md)。方法依据来自项目论文和 [`../HPP_to_analysis_data_processing.md`](../HPP_to_analysis_data_processing.md)。
