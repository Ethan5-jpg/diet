# Experiment 2 Effect Modification v1 本地复查

正式目录为 experiment2_Effect_Modification/v1，运行名为 experiment2_effect_modification_20261110_v1，共 65 条路线。较早的 56 条路线包移至本地代码归档 02_ALL_LOCAL_SOURCE/archive_experiment2_56_routes，归档包的 13 项运行文件通过原清单校验。

改名时逐项核对了模型、特征、训练、评分和存储模块的 SHA256。这些文件与 65 条路线开发包一致。本次变更集中在正式运行名、服务器启动路径、报告标签和模拟目录。正式包预检校验了 13 个运行文件、历史适配器和固定环境。

改名后重新跑完 65 条路线的十折模拟和六屏报告。两轮输出的 65 条路线分数、施加修正折数和对比幅度与改名前的相同模拟数据一致。另用服务器目录布局从外部工作目录启动，完成首折、暂停及第二折续跑，核对了两折的文件和收据均写入 v1 目录。

正式包上传清单 SHA256 为 6db1e9b5160475301bd99c0e57c15f5362e5adcf79f29b36a0f1dd5afad4e897。

[65 条路线的开发验证记录](development/DRAFT_65_ROUTE_VERIFICATION.md)保留开发时的临时编号。真实 HPP 输入与结果由服务器运行核验。
