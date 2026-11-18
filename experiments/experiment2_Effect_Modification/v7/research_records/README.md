# V7 服务器结果

[2026-11-18 03:34 的服务器汇总](../../research_records/2026-11-18_ratio/server_results.md#v7) 对应 `effect_modification_v7_ratio_pc10_20261118_expanded`。发现组两次家庭五折均为 3,343 人、3,059 个家庭。固定历史 PC10 在 r0 返回 `predictions_match`。下表抄录终端显示的交互预测增益占比，单位为 %。

| 流程 | r0 | r1 |
|---|---:|---:|
| 固定历史 PC10 | 6.457 | 1.704 |
| 全部 52 组内层选型 | 7.062 | 0.880 |
| 原 35 组内层选型 | 6.802 | −1.913 |
| 分母保护选型 | 6.161 | −1.913 |
| `gut_pc10_ecology_both` | 7.439 | 4.908 |
| `gut_pc10_marker_gate_fermentation` | 7.184 | 4.687 |
| `gut_pc10_fermentation` | 6.372 | 4.029 |

r0 单次最大为 `gut_pc10_marker_gate_both` 的 7.744%，关闭与开启交互的 R² 分别为 0.07638、0.08279，ΔR² 为 +0.641 个百分点。r1 单次最大为 `gut_pc10_ecology_both` 的 4.908%，对应 R² 为 0.07828、0.08232，ΔR² 为 +0.404 个百分点。两次的 `SIMULTANEOUS_THRESHOLD` 都显示 `supported=none`。同一方案按较低占比排序，`gut_pc10_ecology_both` 位列第一，也是 V8 的起点。原始终端截图保存在本地实验记录中。
