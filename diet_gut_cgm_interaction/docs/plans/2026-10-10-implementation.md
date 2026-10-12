# Whole-cohort diet–gut–CGM interaction Implementation Plan

**Goal:** 在独立目录构建可迁移的个人层面饮食×菌群交互分析，复用原主分析数据，不依赖餐次结果。

**Architecture:** 独立Python包读取既有CSV，先验证输入与人群，再执行冻结的有限模型，输出完整汇总表、图和最多三屏摘要。用户已授权本地直接实现；当前目录不是Git仓库，按用户指定在Data下开发，不另建worktree。

**Tech Stack:** Python 3.8+、NumPy、pandas、SciPy、Matplotlib；bash启动器。OLS/HC3使用直接矩阵计算，statsmodels仅用于本地独立数值对照，不增加服务器运行依赖。

## 设计选择

比较三种方案：全379菌扫描、群落降维、既有四菌的有限交互。第一版选择第三种，确保衔接已有发现、避免立刻扩大检验空间。群落模块和全菌扫描不属于本次冻结分析。

重点：hPDI/NOVA4×4菌×Mean/CV/SDw=24；扩展：其余5饮食×4菌×3结局=60。完整84项同时提供整体BH；24项重点另提供BH，但两者不能混作同一检验家族。

## Task 1 — 输入与配置

创建config/analysis.json、interaction/common.py、interaction/inputs.py。默认路径相对Data；支持--data-root和逐文件配置。读取更正hPDI、验证NOVA4总能量口径与候选master一致、精确菌种匹配、CGM clean/Z一致性、ID/访视/设备/排除标记。每次运行记录实际文件和SHA256。测试错访视、重复ID、错误评分、缺失协变量行、设备冲突和整体目录迁移。

## Task 2 — 统计模型

创建interaction/models.py。OLS包含饮食、菌、乘积和原Model3协变量；HC3协方差与残差自由度t区间。保留原Z尺度。输出原单位交互、菌Q25/Q50/Q75的饮食条件斜率、饮食与菌双IQR差中之差。检查秩、杠杆值、样本量、共同支持。用独立矩阵公式验证HC3和对比协方差。

## Task 3 — 冻结模型矩阵

创建interaction/pipeline.py。84项主样本拟合；重点24项执行strict、protocol相同人未调整/已调整、二次主效应、年龄/性别/BMI交互控制、两饮食共同样本人群敏感性。所有预定失败项仍计入校正家族（以p=1参与，结果q留空），不能只保留可拟合或阳性模型。敏感性是重叠人群证据，不是独立验证。

## Task 4 — 交付与复现

创建run_analysis.py、run_analysis.sh、interaction/reporting.py、requirements.txt、README.md、METHODS.md。独立时间戳输出目录；失败写manifest；完整表与最多三屏摘要；重点24项完整森林图和条件斜率图；仅汇总文件打包。主结果为探索性关联，四菌既有选择来源显式保留。

## Task 5 — 合成验证

创建interaction/synthetic.py和tests。合成样本注入已知交互，同时验证零交互、非线性主效应伪交互、缺失/别名/排除规则、FDR失败分母、路径迁移和CLI端到端。合成输出明确标为SYNTHETIC，不能当作HPP结果。

## 完成标准

本地数值与接口测试通过；合成端到端包含所有预定模型、图和三屏摘要；真实本地预检准确报告缺失服务器文件；文档提供上传后一步运行命令。真实HPP效应与样本规模留待服务器运行确认。

## 执行状态

2026-10-10：Task 1–5均完成。17项自动测试通过，完整228项合成流程通过，图件和汇总打包已核对。详细证据见`docs/本地验证记录_2026-10-10.md`。
