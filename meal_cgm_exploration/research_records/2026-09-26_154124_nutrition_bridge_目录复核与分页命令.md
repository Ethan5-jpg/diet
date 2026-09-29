# nutrition_bridge：第1页复核及下一批查看命令

来源：用户提供截图「截屏2026-09-26 15.41.24.png」。原图已无修改复制至 `evidence/nutrition_bridge_2026-09-26_154124.png`。

- 原图来源路径：`/var/folders/f9/rv3yqgwd61z97g_9l8zf_krc0000gn/T/TemporaryItems/NSIRD_screencaptureui_cPxDTZ/截屏2026-09-26 15.41.24.png`。
- 页面：1/54，运行与目录。
- RUN：`nutrition_bridge_20260926T071736_080048Z`。
- DATA：REAL；STATUS：COMPLETE。
- 旧模型复现：84/84；扩展规格估计：302/302。
- 本图与15:25:33完成记录一致，只重复确认运行状态，没有新增效应、区间、q或模型样本量。nutrition_bridge具体效应已接收数仍为0。
- 图示家族：每套样本营养调整70项、正增量21项、衔接60项；区间为逐项95%区间。

## 下一批：第2、3页

用户终端已在 pheno 环境、meal_cgm_exploration 项目目录。逐条执行，每条输出一页；固定本次运行路径，避免latest随后续运行改变。核对本地入口代码，--show分支读取已有报告后返回，不调用run_models。

```bash
python3 nutrition_bridge/run_analysis.py --show outputs/nutrition_bridge_20260926T071736_080048Z --page 2
```

```bash
python3 nutrition_bridge/run_analysis.py --show outputs/nutrition_bridge_20260926T071736_080048Z --page 3
```

两页分别为原样本与joint的七项饮食平均净增量五步营养调整。每张保留RUN、页码、表头及全部七行。q_full仅属于完整调整模型，各步区间/q另见详情页。

本批没有运行分析或修改冻结代码。服务器全精度结果仍保留在服务器。
