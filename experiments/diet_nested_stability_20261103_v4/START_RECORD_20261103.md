# V4 启动记录

启动时间：2026-11-03 20:37:21.998451（北京时间，UTC+08:00）。
检查时间：2026-11-03 20:38:39（北京时间）。

| 项目 | 实际状态 |
| --- | --- |
| 运行代码提交 | `58d42d9e7ed263d3aec032aff2c3ce2eda243cee` |
| 启动器 PID | 1314 |
| 本地合成测试 | `SYNTHETIC_NESTED_PASS` |
| HPP 合成测试 | `SYNTHETIC_NESTED_PASS` |
| GitHub / 本地 / HPP 四份源码哈希 | 一致 |
| 原 fold0 预处理重现 | `PREPROCESS_REPLAY_PASS` |
| 检查时运行状态 | RUNNING，尚无最终退出码 |
| 已确认完成 | `INNER_DONE r0_f0 0 fits 36`，首个内层折 36 次拟合 |
| 计划总量 | 1080 次内层拟合＋90 次外层拟合，共 1170 次 |
| 完成外层折 | 0 / 10，检查时仍在第一个外层折内 |
| 启动器错误输出 | 检查时为空，实时日志未见异常堆栈 |

实际实验已在 HPP 后台启动，不依赖当前终端持续打开。
新文件均在 `experiments/diet_nested_stability_20261103_v4/`；没有改动学长代码、旧输入或 v1/v2/v3。

这是启动快照，不是最终结果或最终审计通过证明。
后续检查 `start.json`、`logs/execution.json` 指向的实时日志、
`results/r*_f*_progress.json`、`logs/actual_exit_code.txt`、`results/audit.json`。
完成后重新核对全部 1170 项误差、家庭隔离、源文件哈希及警告。

通过最终审计后，启动器将按白名单准备聚合结果并写入正式 Egress Store。
平台导出申请/批准及 GitHub 结果上传是后续步骤，不把暂存视为已经发布。
此前 v3 已完成；其成绩与分析文件尚未取得可下载的正式导出版本。
