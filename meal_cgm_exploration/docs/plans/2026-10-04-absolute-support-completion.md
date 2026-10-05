# Absolute glucose support completion Implementation Plan

**Goal:** 补齐三个已固定门槛下共同调整的Pre/Net/Abs，复用已有结果，生成三页终端摘要、完整结果表及森林图。

**Architecture:** 新建独立gut_absolute_completion；只读原gut_meal_bridge_v2个人汇总及已完成gut_absolute_robustness运行。校验内容哈希、共同成员、旧系数和FDR后，复用40行共同调整参照，只拟合较高两个门槛的Net/Abs共32个模型。旧B72行独立呈现，保留原主结论。

**Tech Stack:** Python3.8兼容语法；numpy/pandas/scipy；Matplotlib（PNG/PDF/SVG）。现有HC3、编码、CLR-Z、资格和门槛复用，不修改旧模块。当前不是Git仓库；用户已确认继续，无需另行确认或工作树。

## 固定统计设计

- 四菌、primary_common/strict_common，5餐/3日、10餐/5日、20餐/7日；仅筛人，保留其全部原合格餐均值；同访视，不筛菌日期先后、不加两周5%门槛。
- 共同调整结果72行：旧A的24行+旧B较高门槛Pre的16行+新Net/Abs的32行。Pre/Net/Abs每组同X；不含餐前水平/斜率。每样本×门槛固定12项BH（q_panel12），NE仍占既定分母。不会跨全部门槛或样本宣称整体FDR控制。
- 旧B72行保留原Net4、Pre+Peak8、q_meal12和原调整集，不被新Net替换。合并展示144行包含重复Pre参照，不能称144个独立发现。
- 24项同X系数恒等式仅作代数核对；不作中介、机制、因果排序或独立复制。新增检验在看过原结果后确定，标为探索/敏感性。

## Tasks

1. inputs.py + CONTRACT.json：校验完整新包与运行依赖；原五个来源文件；上轮结果及私有成员哈希；来源哈希链允许目录搬迁但拒绝不同内容；旧A24、B72规格、计数、调整集、状态/数值、FDR及原参照一致。
2. analysis.py：40行复用、32行新增；每组12项FDR、24项恒等式；计算复用核对清单，生成统一字段并明确估计来源。所有计划行和NE保留。
3. reporting.py：三页（来源/人数/完成情况；主样本12行×三门槛；严格样本12行×三门槛）。q近0.05保留边界信息；原主模型另行汇总；英文Results/Methods和所有CI/p/q在服务器保留。
4. figures.py：两张6面板森林图分别呈现共同调整和原模型；三门槛同列共享坐标范围；完整四菌、逐项95%CI、NE、人数、数据标签、阈值图例及导出CSV。PNG/PDF/SVG；使用服务器全精度结果绘图。
5. run_analysis.py：一条命令；两个来源只读、新时间戳目录、禁止写进来源/既有运行；manifest、spec、成员审计、NE原因和结果哈希；--show latest --page 1/2/3。模型/绘图状态分开，缺Matplotlib保留分析并明确缺图，不暗示全交付完成。
6. tests/test_completion.py：来源复用一致、FDR/NE、同X恒等式、独立矩阵HC3、来源混用/损坏拒绝、输出保护、三个摘要、只32新拟合、独立上传/pheno测试、图件生成及目视复核。README/上传说明/REVIEW/MANIFEST记录真实测试边界。

## Acceptance

不在本地伪造真实新结果。先用合成来源完成全部测试并验证旧文件未变；再交付独立已展开文件夹。终端默认只三页，完整表图留在服务器，用户无需下载数据或结果。
