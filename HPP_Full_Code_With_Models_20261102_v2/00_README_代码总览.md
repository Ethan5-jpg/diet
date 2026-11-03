# HPP 全部本地代码包（包含模型）

整理日期：2026-11-02。包含当前工作区可检索到的全部源码及配置/方法说明选集。代码原样保存，另提供常用入口副本；并非重构后的单一可安装软件包。本次只打包和核验，没有重跑科学分析。

## 目录

- 01_CURRENT_CORE：常用源码入口，按预处理、W19历史基线、v2 ridge、v3残差/PC候选与日级分析、数据交接分组。
- 02_ALL_LOCAL_SOURCE：全部本地源码，保留原相对路径，包括历史版本、失败版本、核验、报表、传输辅助程序和旧审阅包内副本。按05_MANIFEST/files.json可查来源和哈希。
- 03_CONFIG：本地配置与规则；部分是历史/草案，不能据此替代服务器当前执行登记。
- 04_DOCUMENTATION：预处理规则、版本报告、现行实验登记和接续说明。
- 05_MANIFEST：源文件SHA-256、语法检查及原有来源哈希。

这是按最新要求另外生成的本地“含模型”包。服务器此前交给合作者的collaborator_handoff_20261102_v1目录仍是仅数据/预处理包；此处没有改动它。

## 优先阅读与模型入口

|部分|入口（均在01_CURRENT_CORE相应子目录）|
|---|---|
|菌群CLR/表型处理|preprocessing/preprocessing_v1_20261030.py|
|食物QC、真实转换/合表|preprocessing/07_food_qc_amendment.py、26_apply_microbiome_preprocessing.py、27b_apply_phenotype_preprocessing.py、29b_build_analysis_ready_tables.py|
|共享转换/宿主编码/表型接口|models_w19_v1/w19_baseline_core.py|
|原W19/W20学习模块|models_w19_v1/w20_score_core.py、34_run_W19_baselines.py、38_run_W20_scores.py（历史50/50版本）|
|70/30五折v2 block ridge|models_v2/v2_ridge_core.py、41_register_v2_inputs.py、43_run_v2.py|
|残差变系数、有限PC对照|models_v3_daily/targeted_core.py、49b_models.py|
|日级人内CGM分析|models_v3_daily/50b_daily.py|
|新版本独立核验/封存|models_v3_daily/51b_verify.py、52b_seal.py|
|合作者数据包构建|data_handoff/build_handoff.py、add_processed_views.py、final_verify.py|

v2涵盖两饮食×九结局；v3新候选只完成纤维×D_Glu，不能理解为其余17组合已跑完。v3日级使用固定v2家庭出折评分。正式W22/W23尚未运行。

## 环境与路径

真实分析在TRE pheno：/home/ec2-user/miniconda/envs/pheno/bin/python；已核验Python3.11.5、NumPy1.26.4、pandas2.1.0、SciPy1.11.4；还需pyarrow。requirements_core_reference.txt只列核心依赖，不是完整锁文件，报表/浏览器辅助代码可能另需JS工具或本机库。

服务器项目：/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021。

人级输入：data_qc/w20_simplified_20261101_v2；v2源码部署：scripts/w20_simplified_20261101_v2；共享依赖原部署：scripts/preprocessing_20261030_v1；v3源码：scripts/w20_targeted_20261101_v3。

01_CURRENT_CORE副本可用于阅读/复用，但许多流水线硬编码TRE路径、依赖旧阶段退出码和独立输出目录。不要直接批量执行所有脚本或覆盖既有结果。修改/复用时应在独立实验目录显式配置数据、依赖和输出。模型不应在本机寻找个体数据；个体数据、拟合权重和预测产物未打入ZIP。

需要导入核心模块时，可在Python中把01_CURRENT_CORE/preprocessing、models_w19_v1、models_v2、models_v3_daily加入sys.path；这只是解决源码导入，不代替服务器数据/角色/运行前置条件。

## 版本与核验

历史49_models.py因挂载seek写入失败，正式成功的是49b_models.py；50_daily.py仅传输未运行，成功的是50b_daily.py；51/52历史读取和append问题保留，成功的是51b/52b。旧W19/v1与旧预处理角色不等于当前70/30角色。

既有开发成绩参与过选择，不能称独立确认；目标是观测关联修饰，不直接声称个体因果获益。详细方法与结果见04_DOCUMENTATION。本代码包不会更改服务器研究方案或新增实验。

解压后运行python verify_archive.py可校验所有清单文件。Python语法检查不执行源码、不等于科学分析重跑。历史问题原样留存并列于05_MANIFEST/python_syntax.json；当前关键实现已单独检查。
