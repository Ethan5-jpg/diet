# date_split_bridge上传交付与固定方案

2026-09-28。真实后续日期分离模型尚未运行；本轮交付代码，不新增真实研究结论。

## 为什么需要新包

已完成response_followup只换算既有交互并检查日期分离可行性；没有后窗CGM重算和新的β/CI/q。本包冻结上轮切分点与前窗餐成员，完成3种前窗餐反应×4种后窗CGM×2样本共24项主分析，并提供同人同尺度的24项全期对照和24项配对系数差。三类家族各固定24项BH，后两类为次要。

本地未找到旧iglu导出的完整生成定义，使用明确命名的新窗口规则：完整日原始覆盖>=90%、日内15分钟网格、两支撑点间隔<=45分钟、禁止跨日借值/外推/跳过无效读数、网格覆盖>=90%、至少3日。Mean/CV/SDw按固定公式；MODD15为相邻非夏令时变更日同钟点的绝对差均值，至少2日对。旧全期Z仅做一致性诊断。

全期对照与后窗共用人群、前窗X、背景协变量和参考尺度；系数差CI由同人Y差值模型计算，包含配对协方差。日期分离关联不等于外部验证或因果；与全期之差也不能全部归因于共享读数。

## 上传哪一个目录

上传整个 [date_split_bridge](../date_split_bridge/) 到服务器 meal_cgm_exploration/，与response_followup、nutrition_bridge、outputs同级。14个文件，含13个SHA256冻结文件及MANIFEST本身。旧包保留，不上传本地tmp、research_records或模拟结果。

[运行说明](../date_split_bridge/README.md) · [固定计划](../date_split_bridge/PLAN.md) · [操作合同](../date_split_bridge/CONTRACT.json) · [测试核查](../date_split_bridge/REVIEW.md)。

在服务器原pheno环境、meal_cgm_exploration目录运行：

```bash
bash date_split_bridge/run_date_split_bridge.sh --followup-run outputs/response_followup_20260927T155611_920880Z --data-root ..
python3 date_split_bridge/run_analysis.py --show latest --brief
```

只回传3份摘要：状态/指标核对、原样本12项、joint12项；完整表和个人数据留服务器。失败时只回传末尾错误，不用再发几十页。

## 本地验证证据

19项测试全部通过，含SYNTHETIC完整端到端运行；最终冻结版本另经shell入口启动和72项模型运行验证。解析公式、读数分离、夏令时、缺日/长间隔、配对HC3、FDR失败项、哈希防篡改、最新失败运行不得回退、Python3.8语法及bash语法均有检查。本地实际解释器3.12.14，未在3.8解释器实际执行。

新包14文件加旧来源59文件，共73个包文件哈希验证通过。更新前核对263项原归档长度/SHA256，当前入口文档与旧manifest存history；截图、真实转录JSON及6张图不改动。

上轮真实候选4435/4136仅是可行性结果；本包真实最终人数与新关联仍需服务器运行确认。模拟输出不是研究结论，不回填旧真实结果。
