# 饮食记录精度交互的幅度与分折复核

用已验收的 `precision-1` 冻结两种交互方向。在原分折比较 1.00、1.15、1.30、1.45 倍修正幅度，并保留半幅 PC5 比较基准。关交互与无饮食臂沿用同一拟合。把排序靠前的两套模型固定后，在两套不同的家庭划分下重新拟合并计分。重分折使用同一批发现组家庭，口腔及肠道预处理和 UPF 暴露校准在各自训练折重建。

归档分数同时记录一个回顾性诊断。利用固定折外修正下误差对统一幅度的二次关系，检查在至少 9 组正向且零明显下滑的约束下，最多几组能过 +0.10 个百分点。该诊断使用留出的结局来描述上限，实验流程的选模仍使用预先指定的有限候选。

```bash
"$HPP_FULL_PHENO_PYTHON" -B -u scale_precision.py --run-id scale-1 --commit "$CODE_COMMIT"
"$HPP_FULL_PHENO_PYTHON" -B -u scale_precision.py --run-id scale-partition-1 --commit "$CODE_COMMIT" --frozen-from scale-1 --partition-seed 20261137
"$HPP_FULL_PHENO_PYTHON" -B -u scale_precision.py --run-id scale-partition-2 --commit "$CODE_COMMIT" --frozen-from scale-1 --partition-seed 20261183
```
