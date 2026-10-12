# 共同口径的两菌定位与同样本营养比较

版本：**2026-10-11.cross.1**。这是基于已完成中介、整体交互和日级结果的新探索性延伸，尚无真实服务器结果。只复用已经核验的酒精修订日级面板，不重建CGM。全部运行依赖代码包含在本文件夹，不需要合并或改动其他模块代码。

## 研究什么

1. 统一NOVA4定义、参与者及有效日期，比较T068/T096与**个人间窗口平均饮食关联**、**个人内当日饮食偏离关联**的关系。
2. 在完全相同人日内比较单菌和双菌共同模型，直接估计个人内减个人间的系数差及区间。
3. 将营养比较拆成F、N0、N1：F是共同主样本，N0是营养资料完整的子样本、沿用基础模型，N1在N0完全相同人日上增加营养调整。F→N0反映样本/日期组成变化，N0→N1反映调整条件变化，不叫营养中介比例。
4. 补充总能量/已分类能量分母的同样本比较，以及给定Mean和Mean平方时的SD条件关联。

NOVA4主定义为超加工食物能量/总能量，单位10个百分点；两个菌保留原CLR-Z。主要结局是日SD，Mean为解释性结局。B为个人间，W为个人内。每人等权；有效日均值不叫长期习惯。共有26个模型、44条菌交互结果及40个直接差值。完整设计和校正家族见[METHODS.md](METHODS.md)。

## 上传哪个文件夹

仅上传整个本地 `cross_scale_nutrition` 文件夹，放在服务器：

```text
/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/cross_scale_nutrition/
```

它与 `daily_within_person`、`extension_sensitivity` 同级。保留服务器原 `daily_within_person/alcohol_revision/outputs/` 中的结果。新文件夹本地不含真实数据，不生成ZIP。

## 完整服务器运行指令

上传后整段复制：

```bash
conda activate pheno &&
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/cross_scale_nutrition &&
python -B -c 'from crossscale import VERSION; print("CODE_VERSION=" + VERSION); assert VERSION == "2026-10-11.cross.1", "Please upload cross_scale_nutrition version 2026-10-11.cross.1"' &&
PYTHON=python bash run_analysis.sh \
  --reference-run ../daily_within_person/alcohol_revision/outputs/revised_20261011T103556_258035Z_9583d5
```

末尾只需回传 **COMMON 1/3、2/3、3/3三屏及RESULT_DIR** 截图。中间 `COMMON MODELS` 是进度；不需要下载CSV或个人记录。如果出现 `COMMON STOPPED`，回传报错截图；不要跳过指纹或历史复算检查。

服务器路径的前缀可以与本地不同。脚本默认参考目录相对于这个新文件夹，命令也明确指向同级原结果；没有把本地 `/Users/...` 写进运行路径，也不使用原manifest里的本地/服务器Data根去重新读取原始数据。不自动挑选“最新”运行。

依赖沿用numpy、pandas、scipy。仅在明确提示缺包时执行：

```bash
python -m pip install -r requirements.txt
```

## 三屏怎么看

- 第一屏：真实/合成标签、版本、旧结果8/8复算、F/N人数与日数、拟合完成数、支持提示。支持提示包括单人信息过多、尾部样本不足、双菌联合取值支持不足；不把提示为零当成所有统计假设均成立。
- 第二屏：F共同样本的单菌/双菌系数，8条双菌结果的区间与q，以及4条W−B直接差值。不能以一个显著、另一个不显著代替差值检验。
- 第三屏：8组F/N0/N1交互系数、N1−N0区间与q，以及4条Mean条件下的SD结果。Mean仍为解释性终点。

第一屏如显示 `COMPLETE_WITH_LIMITATIONS`，计算已结束，但存在支持提示或不可估计行，须结合结果解释；`NO_ESTIMABLE_PRIMARY_MODELS` 表示四条主要SD交互均未估计。失败行保留在预定校正分母中，不能通过删除它们提高显著率。运行标签为 `SYNTHETIC` 时只属于代码验证。

`qSD4`：F共同双菌模型的2菌×2尺度SD检验；`qMain8`：同一主模型再包括Mean，共8项。支持性模型各自的 `qFamily` 与全部44项 `qAll44` 同时保存。差值另有自己的固定家族和 `qAll40`。这是看过旧结果后的新探索，不能称为预注册或独立验证。

## 结果保存和可追溯性

每次新建 `outputs/common_.../`，拒绝覆盖已有目录。`manifest.json`记录输入、代码、配置、版本和结果指纹。完整文件包括：

| 文件 | 内容 |
|---|---|
| tables/00_reference_reproduction.csv | 旧主分析和营养模型8条系数、区间、p、样本指纹复算 |
| tables/01_sample_flow.csv、02_sample_summary.csv | 共同样本流程、人数、天数、菌间相关 |
| tables/03_model_sample_audit.csv | 全部26个模型的人日一致性指纹 |
| tables/10_interactions.csv | 全部44条交互、单位、区间、q和支持提示 |
| tables/11_comparisons.csv | 全部40条差值，保留共享参与者协方差 |
| tables/12_model_diagnostics.csv | 残差自由度、条件数、杠杆值、失败原因 |
| tables/13_all_coefficients.csv | 所有模型项的系数和区间 |
| tables/14_conditional_slopes.csv | Q25/Q50/Q75处饮食斜率及Q75−Q25对比 |
| tables/15_microbe_anchors.csv | F人群的菌均值与分位锚点 |
| private/sample_membership.csv | 个人、日期和F/N归属，只留服务器 |

所有旧结果保留。新B模型使用同一批合格日的平均日SD和平均日Mean；不会冒称精确复现此前全部CGM日期的整体模型。新W主模型按人等权，也不冒称与此前按天等权的主模型参数完全相同。旧日级模型已作为独立的8项检查点先复算。

数据都按同一研究时间窗解释，不按粪便采样前后增加筛选或重新突出日期顺序。其观察性解释与相对组成单位保持不变。

## 可选查看方式

只做输入、历史结果及共同样本检查（不保存新文件、不拟合新研究模型）：

```bash
PYTHON=python bash run_analysis.sh \
  --reference-run ../daily_within_person/alcohol_revision/outputs/revised_20261011T103556_258035Z_9583d5 \
  --check-only
```

已经完成后如需重看某一屏，将路径替换为实际RESULT_DIR：

```bash
python -B show_results.py --run-dir '/这里替换为实际RESULT_DIR' --screen 2
```

展示前核对保存结果指纹，不重新拟合。本地数值和运行检查见[VALIDATION.md](VALIDATION.md)；后续真实截图保存在本文件夹的[research_records](research_records/README.md)。
