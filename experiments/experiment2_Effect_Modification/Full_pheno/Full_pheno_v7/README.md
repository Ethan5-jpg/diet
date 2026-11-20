# Full Pheno v7 饮食与菌群交互复核

在同一发现组 18 个组合和家庭测试折上，比较不同饮食主效应收缩下的肠道 PC5 修正。固定候选还测试与 PC2 的低维混合、口腔 Shannon 补充项、在训练折用同一饮食和表型域的其他结局估计系数先验，以及对大训练残差降权。所有流程保留原纤维→平均血糖锚点。主比较的第一列复核 v5 半幅 PC5 参照。

受限选模仅在外层训练家庭的四个内折进行。重新训练微生物预处理和 UPF 暴露校准。最多替换原预测的 50%。更新同时要求完整预测和交互增量各改善至少 0.02 个百分点，并在至少三个内折方向一致。分别报告三臂的预测增量、相对参照的完整预测差和仅交互增量差。当其余 17 组至少 9 个为正、至少 2 个达到 +0.10 个百分点、没有组低于或等于 −0.10 个百分点时，达到开发门槛。

主筛查结束后记录排序靠前的两个候选。两套冻结分折仅对这些候选和参照计分，并在每一折按训练家庭重新拟合口腔与肠道预处理和 UPF 校准。它们使用相同的发现组家庭。按家庭同步重抽 9,999 次的区间在各自比较集合内校正。预留确认家庭的结局不参与。

```bash
OPENBLAS_NUM_THREADS=2 "$HPP_FULL_PHENO_PYTHON" -B -m unittest test_interaction_rules test_replay
"$HPP_FULL_PHENO_PYTHON" -B -u run_interaction.py --run-id interaction-1 --commit "$CODE_COMMIT"
"$HPP_FULL_PHENO_PYTHON" -B -u run_interaction.py --run-id partition-1 --commit "$CODE_COMMIT" --frozen-from interaction-1 --partition-seed 20261137
"$HPP_FULL_PHENO_PYTHON" -B -u run_interaction.py --run-id partition-2 --commit "$CODE_COMMIT" --frozen-from interaction-1 --partition-seed 20261183
```

依赖已部署的 Full Pheno v1–v6 和 v6 暴露复核代码。真实结果、运行记录及输入均留在 TRE。阶段结果位于 `runs/TRE_ONLY/<run-id>/summary.json`，可以用同一运行名加 `--report` 阅读汇总。
