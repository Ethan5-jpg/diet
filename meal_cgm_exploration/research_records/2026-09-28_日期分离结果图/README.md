# 日期分离CGM结果图

依据2026-09-28 23:55–23:56用户提供的3份真实服务器摘要，2026-09-29整理。全部24项主关联及24项配对差值同时展示；没有重新拟合、计算p或重新做FDR。

[PNG预览](Figure4_date_split.png) · [PDF](Figure4_date_split.pdf) · [SVG](Figure4_date_split.svg) · [真实结果解读](../2026-09-28_date_split_bridge_真实结果解读.md)。

左图为前窗个人餐反应关联后窗CGM；右图为同人同尺度“后窗−全期”系数差。蓝圆点为原样本，橙方点为joint，两者重叠。实心代表对应24项家族q<0.05；空心代表q>=0.05。区间为逐项95%；主分析与配对差值是两个不同检验家族。三项Mean的joint差值均为空心，正增量—Mean虽逐项CI不跨0，q=0.0539仍未通过。

单位为新全期参考Y的SD/前窗X参考SD，不能解读为mg/dL或相关系数。全期包含后窗，右图不是共享读数的纯因果贡献。MODD15为预定新操作定义，不断言精确复现旧iglu。

数据文件保留截图精度及q显示零的标记；图表不能提供服务器未展示的完整精度。artifact_provenance.json保存源JSON、绘图脚本与输出哈希。可在本目录使用已装有matplotlib的Python复现：

```bash
python3 plot_date_split.py --input ../2026-09-28_date_split_bridge_三份摘要_截图精度.json --out .
```

这张图追加到原6张结果图之后，不覆盖旧图。
