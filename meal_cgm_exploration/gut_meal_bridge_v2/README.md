# 上传包 gut_meal_bridge_v2：旧重点菌与餐前餐后血糖

当前版本 `2026.09.29.2`：修正无效餐连接编号为空导致整批停止的问题。本次上传新文件夹gut_meal_bridge_v2，与服务器旧gut_meal_bridge并列；无需覆盖旧文件夹。包含MANIFEST.json。运行日志会显示版本号。

**只上传整个 `gut_meal_bridge_v2/` 文件夹**，放到服务器已有的：

`/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration/gut_meal_bridge_v2/`

它与hppmeal、seven_diet_meals、nutrition_bridge、outputs并列。保留其他目录和旧结果。沿用pheno环境（Python>=3.8及已有numpy/pandas/scipy），不安装新软件。本包仅新增独立输出目录，不改写旧分析。

## 运行一条命令

在服务器 `meal_cgm_exploration` 目录内：

```bash
bash gut_meal_bridge_v2/run_gut_meal_bridge_v2.sh --data-root ..
```

默认原餐次运行为 `outputs/meal_20260924T143630_752200Z`，CGM来源核对用 `outputs/nutrition_bridge_20260926T071736_080048Z`。如果原结果搬家，可用`--meal-run`、`--nutrition-run`明确指定；不自动选别的旧运行。

新输出为 `outputs/gut_meal_bridge_v2_时间戳/`。默认显示`COMPLETE_WITH_LIMITATIONS`，表示分析完成但仍是同队列候选关联，且菌日期可能不完整。若某模型不可估计，保留NE及原因；检查完成数量，不能把“完成”误读成全部模型显著。`--stage check`只检查输入和共同样本，不产生效应。

## 只回传三页

```bash
python3 gut_meal_bridge_v2/run_analysis.py --show latest --brief
```

1. 状态、样本筛选、采样时间。
2. 主样本4菌的净增量、餐前水平、峰值结果。
3. strict敏感性和同人整体CGM参照。

一次看不方便时，将`--brief`换成`--page 1`、`--page 2`或`--page 3`。不需要发送全部明细，不上传private目录。latest始终指向最新尝试；若最新失败/仅check，不会回退显示旧成功结果。错误时只发错误信息及RUN_DIR。

## 输入与输出

- 原餐次、微生物CLR-Z、协变量、三份饮食评分及CGM表按历史运行SHA256核对；代码依赖也核对。缺文件/哈希不一致时停止，不跳过。
- hPDI使用更正来源；NOVA4/日能量沿用seven_diet_meals配置。候选菌须在原物种CLR-Z表中精确唯一匹配。
- 菌日期默认优先原配置，或标准 `Data/Transfer/gut_microbiome/gut_microbiome.csv`。没有该文件仍可运行，会注明UNKNOWN。有多次样本且无法匹配时不会猜日期。
- 若必须搬动源文件，在包外复制sources.json再指定`--sources-config`，评分来源可用`--score-sources`。改路径不绕过历史数据指纹。
- `tables/models.csv`保存48规格、点区间、p/q、单位、样本哈希、协变量和不可估计原因；其他表保存样本、日期、描述及完整系数。
- `tables/01_meal_connection_audit.csv`分别计数可分析餐与无效餐的连接编号缺失。无效餐不进入分析；可分析餐缺编号或同人含多个连接时仍停止。
- `private/person_inputs.csv`、`meal_membership.csv`是个体数据，只保留服务器。三页只含聚合结果。
- `analysis_spec.json`和`manifest.json`保存设计、来源、版本、状态和结果校验码。

这轮回答旧菌群信号是否也表现于餐次特征；不重做中介、不估计恢复率、不筛所有菌、不将原NOVA4交互阴性当作菌主关联阴性。方法见PLAN.md；本地检查见REVIEW.md。本地测试均标SYNTHETIC，不是新的研究结果。
