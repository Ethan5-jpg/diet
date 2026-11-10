# V7 全部模型结果截图

来源：2026-11-08 01:18:36（Asia/Shanghai）用户截图，执行 `report.py --run-id main_v7_20261107 --all`。原图 full_results.png；完整人工核对数值 visible_results.json、visible_transcript.txt；OCR 辅助文本未修订，不能直接使用其中错误的 0/9、模型名或末位数字。

截图明确显示 RUN=main_v7_20261107 STATUS=COMPLETE、10/10 外折、两轮各 N=3343。完整包含 30 条新管线及 3 条参考。与上一张完成屏重叠的 11 行均一致；新增展示了 22 条新管线。

两条 FM 均有 W，双站点 FM 两轮负 R²。具体警告及退出原因、融合权重仍未读，不能断言所有折均未收敛或 FM 已拖累融合。模型最大单轮 R²为 ridge_stack 的 0.119128，三条显式交互结论与先前一致。

README、中文 SUMMARY 及状态索引已更新。SUMMARY_before_full_results.md 保留上一版截图摘要；derived_family_comparisons.json 的 oral/balanced 差值为截图舍入值相减，不是额外服务器精度结果。

INSPECT_FM.py / INSPECT_FM.sh 是下一步只读诊断。diagnostic_synthetic_check.txt 明确仅为合成测试输出，diagnostic_validation.json 记录只读文件哈希检查通过；不得将其消息或系数当真实结果。未下载服务器数据、预测或拟合权重，未重拟合或修改服务器模型。

后续更新：FM 诊断已收到，口腔 3/10、联合 9/10 为迭代上限；具体权重见 [诊断记录](../2026-11-08_fm_diagnostic/record.md)。
