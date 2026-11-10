# V7：饮食—口腔/肠道菌群模型扩展

这批代码可一次上传、一次启动，共 **30 条新管线、67 组内层候选设置**。继续固定 `fiber_density → D_Glu`，使用与 v4/v6 相同的外层家庭划分和样本。服务器代码及输出为英文，本地说明与研究记录为中文。

**当前仅完成代码与本地合成验证，尚未在服务器真实数据上运行。**检查证据见 `research_records/VALIDATION.md`，不能把合成验证当作 R² 提升结果。

## 上传和启动

将本文件夹 `diet_model_expansion_20261107_v7` 整体上传到服务器 handoff 的 `experiments/`，保留已有 v4、v6、输入与预处理。新代码只读引用旧文件，所有新结果留在本目录 `runs/`。默认评价参照为 v6 已完成的 `console_20261107`；其 fusion 预测只在最后比较，不参与训练、选型或选权。

服务器依次执行：

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_expansion_20261107_v7
bash setup.sh
```

安装脚本用原 pheno Python 创建本模块 `.venv`，继承原数值环境，仅安装 LightGBM 3.3.5、Interpret Core 0.6.5，不升级原 pheno 环境。数值包版本不符时会停止。

若提示没有 tmux，执行一次：

```bash
sudo yum install -y tmux
```

启动整批，先自动运行合成检查，通过后才读真实数据：

```bash
.venv/bin/python launch.py --run-id main_v7_20261107 --threads 4
tmux attach -t hpp_v7_main_v7_20261107
```

按 `Ctrl+b`，松开，再按 `d` 离开会话。任务保留在服务器 tmux 中，关闭网页后可以重新连接。服务器关机、重启或平台终止整个用户会话仍可能中断任务。

不要把输出重定向到长期开启的日志文件。v6 曾在 FUSE 日志写入上阻塞，本版直接使用真实终端；检查点仍需服务器文件系统正常读写。无法安装 tmux 时可用 `launch.py --console` 前台运行，但不建议在可能断开的网页入口使用。

## 查看进度与截图

另开终端，进入本目录后：

```bash
.venv/bin/python report.py --run-id main_v7_20261107 --progress
```

只显示收据文件数、外层结果收据数及锁状态；锁存在本身不证明正在正常计算。首次加载及哈希未完成时可能显示 `NOT_REGISTERED`。详细活动看 tmux 中的 `START`、`SELECT`、`OUTER_COMPLETE`。

结束后截图下面这条命令的结果即可，默认约一至两屏：

```bash
.venv/bin/python report.py --run-id main_v7_20261107
```

需要完整模型表时才用：

```bash
.venv/bin/python report.py --run-id main_v7_20261107 --all
```

默认报告包含 v4/v6 基准、前五条新管线、三种选型/融合、两个交互基线及三条交互的 ΔR²、折间获胜次数和 modifier SD。完整数值与警告保存在服务器 `summary.json`，模型及逐人信息保存在 `TRE_ONLY/`。

**只需给终端截图，服务器数据、逐人预测和权重都留在服务器。**后续原图、可见数值转录和详细中文 summary 存在本模块 `research_records/`。本地 `.venv`、`runs`、缓存不需要上传；本交付没有真实参与者数据。

## 方法安排

| 路线 | 管线数 | 检验的问题 |
|---|---:|---|
| LightGBM：host/oral/gut/balanced | 4 | 新的梯度树学习方式及不同站点视图 |
| EBM：oral/balanced | 2 | 可逐特征解释的非线性加性主效应 |
| Hellinger-HGB：oral/balanced | 2 | 参考集合内闭合相对丰度的开方表示 |
| Presence-HGB：oral/balanced | 2 | 只看出现与否是否减少丰度噪声 |
| 单菌样条 GAM：oral/balanced | 2 | 在筛选菌种上做加性非线性，区别于旧 PC 样条 |
| 随机树嵌入 + ridge：oral/balanced | 2 | 无监督树分区形成稀疏表示 |
| 宿主残差监督 PLS → SVR：oral/balanced | 2 | 监督低维方向之后的非线性预测 |
| 二阶低秩 FM：oral/balanced | 2 | 用共享低秩结构压缩大量乘积项 |
| Host residual ExtraTrees：oral/balanced | 2 | 宿主/饮食线性部分之外的菌群信息 |
| Hellinger Nyström → ridge：oral/balanced | 2 | 组成距离的固定规模核近似 |
| Gut mixture experts | 1 | 肠道软分群下的局部预测规则 |
| 分块 additive boosting | 1 | 宿主/饮食、口腔、肠道分块回拟的无交互基线 |
| EBM/PC/gate 显式交互 | 3 | 饮食×肠道菌群的额外样本外信息 |
| Inner selector/convex blend/ridge stack | 3 | 只在内层选择或组合模型是否更稳 |
| **总计** | **30** | **另加 3 条只读历史评价基准** |

`oral`/`gut` 都包含宿主与饮食；`balanced` 加两站点，每站点分别分配相同筛选上限。筛选只用训练角色的宿主/饮食拟合残差与单菌表示的相关程度。`count=0` 表示该表示下全部训练支持的菌种。

开方本身是单调变换，树通常不受它影响；Hellinger-HGB 检验的是相对丰度表示与 CLR 的差别。随机树嵌入不同于普通 RF，监督 PLS→SVR 不同于旧的无监督 PC→SVR。

## 评价与解释

- 固定结局、原始单位及两轮家庭五折，外层训练集内再做家庭三折。每个角色重新拟合目标处理、CLR、筛选、PCA/PLS、样条、聚类、核带宽和模型。家庭不跨训练/验证。
- 内层用 pooled MSE，按样本数加权；外层合并 OOF 预测算 R²，不取折 R² 的简单平均。保留负 R² 与负增量。
- EBM 关闭内部随机验证与早停；正式外层验证不参与调参。融合成员预先固定，权重及 inner selector 仅用内层 OOF。经调参的内层 OOF 不当作无偏性能报告；它们的评价来自外层 OOF。
- 显式交互保留零交互候选：`ebm_diet_gut` 对 `ebm_balanced`；两条 additive 交互对同一已选择 `additive_boost`。EBM 全模型可重估主效应；残差交互固定基线预测。
- Additive 基线已包含训练肠道 CLR PC10，因此 PC/gate 交互不是只凭额外引入肠道 PC 获得增量。EBM 只开放饮食线性项×最多 8 个训练筛选肠道菌的成对项。
- 主表 `dV6` 是总体预测相对 v6 fusion 的 R² 差，不能叫饮食交互贡献。交互表另列相对匹配 null 的 ΔR²。
- `SD` 是纤维密度从 10 到 15 时交互预测变化的个体间 SD，单位 mg/dL；不是因果效应。`active` 要求选中非零交互且验证 modifier 有非零变异。
- `W` 表示某个已选择外层拟合存在警告或 FM 达到迭代上限；保留结果与完整警告，不暗中剔除。
- v1–v6 已用于选择开发方向，新的外层结果仍是开发结果。两轮共享参与者，不是独立人群重复；多模型事后最高值也不是确认性结果。不读取封存确认集。

## 恢复与计算量

每个内层候选及外层管线完成后单独写入并校验检查点。正常退出释放锁，意外终止留下的锁不会自动删除。任务仍在 tmux 中运行时不要重复启动。

任务退出后 tmux 保留 shell。确认旧任务已结束且锁正常释放后，在该会话里恢复：

```bash
.venv/bin/python run.py --run-id main_v7_20261107 --threads 4 --resume
```

也可在没有旧同名 tmux 会话时用 `launch.py --resume`。如旧会话还在，应 attach 后恢复；不要删除旧运行目录。残留锁的进程状态不清楚时，先截图锁与进程状态，不盲目删除。

代码、配置、输入、参考文件、Python/包版本或线程数改变时，拒绝混用旧检查点，另起 run-id。

默认约 **2010 次内层候选评价**（67×3×10），另加选定模型重拟合和三种融合。各方法顺序运行，单方法最多使用指定线程数，不同时启动 30 个大进程。真实耗时需要看首个外折日志，不能从合成数据耗时外推。

## 文件

`features.py` 为共享表示；`estimators.py`/`models.py` 为模型与交互；`engine.py`/`registry.py`/`config.json` 为统一验证与冻结网格；`bridge.py` 只读旧适配器和 v6 参照；`runtime.py`/`run.py` 管版本、哈希及锁；`setup.sh`/`launch.py`/`check.py` 负责环境、启动与合成检查；`report.py` 输出聚合结果；`tests/` 不读取真实参与者表。

方法来源与设计见 [计划](docs/plans/2026-11-07_v7.md)，已有证据与后续判断标准见 [中文记录](research_records/PLAN_AND_STATUS.md)。不保证达到 R²=0.2–0.3；若仍提升很小，应继续审视时间对齐、测量误差及“人均代谢水平”与“同样饮食的个体响应”是否一致。
