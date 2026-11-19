# Full Pheno v6 · 饮食表征与内层校准

`features` 使用已登记的发现组饮食日期，比较累计能量加权纤维、明确全谷物分类与豆类及果蔬的纤维密度、宏量营养素与钠密度、已确认 NOVA 4 的零食饮料和其他食物，以及未分类能量比例。食物类别来自源记录和固定映射。未分类能量单列。每日分类后重加总并与原每日总量对照，每个候选至少需三天且覆盖原有效天数的 80%。结果包括前后段可靠性、评分覆盖率和缺少支持的组合。候选与原模型均在相同覆盖子样本和家庭折上重新拟合，血糖锚点使用三菌＋Shannon 对照。

`nested` 在原 18 组上比较半幅 PC5，固定饮食的交互收缩，饮食形状与交互一起选择，同域部分汇聚，以及在每个内层训练家庭重新拟合 FoodID 概率校准后的两种敏感性流程。训练内层四折分别拟合 CLR/PCA、结局清洗和模型。饮食分支和交互各要求内层至少三个折改善，并有至少 0.02 个 R² 百分点的合并增益。通过门槛的交互可以部分借用同域校准幅度，内层评价不足时可回到零。外折仍用同一登记的饮食缓存，保留纤维→血糖的原锚点。两类流程分别报告无饮食、关交互和开交互分数，以及按家庭配对重抽样的同时区间。预留家庭结局不参与本轮。

使用既有 v5 Python 环境和已部署的 `Full_pheno_v6`。运行记录保存实际 Git 提交号和输入、源码指纹。

```bash
OPENBLAS_NUM_THREADS=2 "$HPP_FULL_PHENO_PYTHON" -B -m unittest test_exposure
"$HPP_FULL_PHENO_PYTHON" -B -u run_exposure.py --phase features --run-id foods-1 --commit "$CODE_COMMIT"
"$HPP_FULL_PHENO_PYTHON" -B -u run_exposure.py --phase nested --run-id nested-1 --commit "$CODE_COMMIT"
```

结果写入本目录的 `runs/TRE_ONLY/<run-id>/`。`--report` 在 TRE 内可复读完成的汇总，并用 `--phase` 核对运行类型。
