# Full Pheno v6 · 饮食主效应复核

用 W20 的发现家庭五折和原有 18 个饮食–表型组合，复核从无目标饮食到关交互、再到开交互的预测差。在原模型之外，比较线性剂量、线性剂量岭回归、对折线项加强收缩，以及对剂量 × 宿主项进一步收缩。各改动用同一批人和共同宿主与菌群背景。岭回归惩罚只作用于饮食项，菌群修正在相应饮食预测的训练残差上估计。保留原三菌＋Shannon 作为血糖锚点及跨表型对照。固定配方之间的比较属于发现组探索。

程序对照 v5 汇总中的各组配对人数、家庭数和三个绝对 R²。对整套流程和 18 个组合的配对差按家庭同步重抽样，报告同时 95% 区间。另报告正向折数，以及按家庭绝对配对误差变化去除最大 1% 家庭后的描述性敏感性结果。该删减结果不用于选模。

在已有 `Full_pheno` 下新增本目录。使用此前运行 v5 的 Python 3.11 环境并记录这次部署的实际 Git 提交号。

```bash
"$HPP_FULL_PHENO_PYTHON" -B -m unittest test_diet_audit
"$HPP_FULL_PHENO_PYTHON" -B -u run_diet_audit.py --commit "$CODE_COMMIT" --run-id diet-main-1
"$HPP_FULL_PHENO_PYTHON" -B run_diet_audit.py --report --run-id diet-main-1
```

注册信息和汇总写入版本目录下的 `runs/TRE_ONLY/<run-id>/`。数据读取和真实运行均在 TRE 内完成。预留家庭结局不参与本轮。
