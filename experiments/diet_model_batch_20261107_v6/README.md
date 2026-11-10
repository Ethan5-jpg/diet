# V6：饮食–口肠菌群模型批量实验

本批接续组员已完成的 v5，统一编号为 v6。六组方法都属于本轮 v6；上传整个文件夹即可。数据接口继续使用服务器上已有的 v1/v4 代码，早期两模型草稿已归档，不另计实验版本。

本批固定 `fiber_density → D_Glu`，沿用两套外层家庭五折及内层家庭三折。所有方法使用相同的平均血糖结局、准入人群和折内预处理。0.2–0.3 是探索目标，真实 R² 需服务器运行后判断。

语言与篇幅约定：服务器上的代码注释、日志和终端标签使用英文，截图输出保持精简；给用户的解释和本地 summary 使用中文，详细保留数值、阴性结果、比较口径与局限。本轮已收到完成截图，中文详细记录见 `research_records/SUMMARY.md`。

## 六组方法

| 组 | 内容 |
|---|---|
| CatBoost | 比较全部菌群、训练折内筛选64/256种菌；口腔及口肠联合输入 |
| XGBoost | 同样的输入与筛菌选项，检验另一种提升树实现 |
| HGB＋筛菌 | 保留既往 HGB 网格，新增训练折内监督筛选，作为处理方式的比较 |
| 分块核回归 | 完整标准化 CLR 上的线性/Gaussian核，口腔、肠道分别调权重；匹配比较线性/样条饮食交互 |
| 监督交互降维 | 用“饮食×肠道菌群”对训练主模型残差学习PLS方向；不是既往直接用PLS预测血糖的模型 |
| 模型融合 | 用外层训练家庭内的OOF预测学习非负、和为1的融合权重 |

共11条新预测管线，默认87个内层候选设置，外加融合权重估计。参数集中在 `config.json`，不按外层分数扩大网格。`top_k=0` 表示保留全部菌群。筛选范围及模型参数一起在内层选择，最终报告的是整条管线的表现。

## 上传与启动

2026-11-07 的服务器截图定位到原后台日志写入等待。本次新增 `launch_console.py`，进度直接显示到终端，使用独立运行编号 `console_20261107`。原 `launch.py` 保留追溯。已经上传旧版时，只需补传这个新文件，原源码、数据和依赖沿用。

本文件夹在服务器上的位置：

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_batch_20261107_v6
```

然后复制执行：

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_batch_20261107_v6
/home/ec2-user/miniconda/envs/pheno/bin/python -m pip install -r requirements.txt
/home/ec2-user/miniconda/envs/pheno/bin/python launch_console.py --run-id console_20261107 --threads 4
```

依赖已安装时，可省略安装命令。新入口先运行现有合成测试，通过后替换为同 PID 的训练进程；保持当前终端打开，进度直接显示，不能重定向到文件或管道。同一外折内依次运行各方法，共用预处理与缓存，限制CPU占用，无需逐个模型启动。

这是独立的重新运行，产物保存在 `runs/console_20261107/`。旧 `main_20261107` 的日志、锁与缓存保留；新运行不访问或复用旧检查点，也不替旧任务删除锁。两个运行的输出目录隔离，科学配置保持不变。若 FUSE 的其他文件操作也等待，仍需平台维护方处理，新入口不能修复挂载服务本身。

完整运行结束后，自动生成约一屏的 `runs/console_20261107/SUMMARY.md`：展示前三条新管线的两轮 R²、相对 v4 口腔 HGB 基线的提升，以及三种显式交互相对匹配主模型的 ΔR²。按两轮中较低的 R² 排序；同一管线两轮都达到阈值才标记达标，不混用不同模型的成绩。所有管线的完整结果保存在同目录 `summary.json`。

原冻结程序可重新查看紧凑汇总：

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python run.py --run-id console_20261107 --summary
```

为避免原终端的中文方框，另有 `show_results_en.py`：只读已完成运行的 aggregate `summary.json`，显示全部管线两轮分数及匹配交互增量，不拟合模型、不读取逐人数组、不改冻结程序。该文件需先补到服务器，本地新增不代表服务器已存在；补入后执行：

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python show_results_en.py --run-id console_20261107
```

只需反馈汇总截图。真实输入、逐人预测、拟合对象和计算缓存全部留在TRE；不下载结果表、模型或ZIP。

## 代码与检查

- `data.py`：唯一真实数据入口；只读发现输入，调用已有训练角色预处理接口。
- `numerics.py`、`learners.py`：共享核计算、筛菌、树模型、监督投影和融合。
- `experiment.py`：所有方法共用的嵌套验证循环。
- `storage.py`：校验和、候选级缓存与断点续跑。
- `run.py`：训练与结果汇总；`launch_console.py` 为本次修订后的终端启动入口；`reporting.py` 生成短版摘要。
- `tests/`：数值解、所有默认候选、数据泄漏、完整流程、保存和续跑测试。

代码通过Ruff格式与静态检查；完整测试状态和环境见本地 `VALIDATION.json`。合成验证不是真实HPP成绩。

## 关键口径

1. 每个内/外训练角色重新拟合筛菌、伪计数、CLR、尺度、饮食基函数、结局处理及宿主编码。新增筛菌、PLS和融合只使用外层训练家庭的信息。树模型不向验证集传 `eval_set`，不做外层 early stopping。
2. 先在内层为核主效应选参数，再固定该背景比较交互；交互允许权重为0。`kernel_supervised` 的PLS方向仅用相应训练角色的饮食交互列和主模型残差学习。
3. 融合输入是内层选参后的OOF预测。这些OOF结果继续用于选择融合权重，不作为最终无偏成绩；最终只在未参与本次拟合/选择的外层家庭评分。
4. 每轮分别计算 pooled OOF R²，保留负值和阴性结果。总预测提升相对 v4 HGB 比较；显式交互增量只相对同一个 `kernel_main` 比较。不同主效应背景下的 ΔR² 不能直接当作机制证据。
5. 两轮共享参与者，且该发现集已有多轮开发，本批仍是探索性分析。确认集结局和冻结PC10规则不读取、不修改；旧模型不重新运行。
6. 不完整运行不报告 pooled R²。每个完成候选带内容校验和；续跑前核对源代码、配置、输入和环境，完整收尾再检查一次输入/源码。

正常退出或中断后可用 `python launch_console.py --run-id console_20261107 --resume --threads 4` 继续，复用该新运行自身的已完成计算。新配置应使用新的 `--run-id`。强制结束留下运行锁时，先核实文件内PID对应进程已停止，再处理该锁；代码不会自动结束其他进程或删除其锁。

本批研究的是访视内关联及预测。固定纤维10→15对比的交互部分SD不是实际饮食干预效应。

## 方法文档

[XGBoost 2.0参数](https://xgboost.readthedocs.io/en/release_2.0.0/parameter.html) · [CatBoost参数](https://catboost.ai/docs/en/references/training-parameters/common) · [PLSRegression](https://scikit-learn.org/1.4/modules/generated/sklearn.cross_decomposition.PLSRegression.html) · [菌群核学习背景](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011240)。本包使用自定义分块核与显式交互，不是完整KernelBiome模型库的复现。
