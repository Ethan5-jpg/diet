# Full Pheno v7 饮食记录精度复核

沿用同一批留出家庭、v5 参照和 v6 的剂量及宿主收缩模型。用现有日级记录估计日间摄入波动，复核每个参与者的日数与登记剂量。用训练家庭估计人间变异和个人均值的不确定度，将记录偏少或日间波动大的剂量向训练均值收缩。另比较训练折分位点截尾及肠道 PC2 混合。替代剂量在训练折匹配原剂量尺度，变化只进入菌群交互乘积。UPF 日间波动使用与登记访视均值一致的逐人未知能量系数近似。

同批比较采用家庭同步重抽 9,999 次，并保留原纤维→平均血糖锚点。百分点是折外 R² 差。该阶段仅使用发现组结局。

```bash
OPENBLAS_NUM_THREADS=2 "$HPP_FULL_PHENO_PYTHON" -B -m unittest test_precision
"$HPP_FULL_PHENO_PYTHON" -B -u run_precision.py --run-id precision-1 --commit "$CODE_COMMIT"
```

此目录依赖已部署的 Full Pheno v1–v7。登记和汇总位于 TRE 的版本运行目录。
