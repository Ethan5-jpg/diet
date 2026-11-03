# 肝脏六表型：用户已确定类别和均值口径

2026-10-27。只保留六类，每类一个phenotype。用户明确选择均值，不再等待mean／median选择。

| 表型 | 明细输入字段 | 同访视运算 |
|---|---|---|
| 弹性 | `elasticity_qbox_mean` | 合格观测算术均值 |
| 剪切波速度 | `velocity_qbox_mean` | 合格观测算术均值 |
| 黏度 | `viscosity_qbox_mean` | 合格观测算术均值 |
| 频散 | `dispersion_qbox_mean` | 合格观测算术均值 |
| 衰减系数 | `attenuation_coefficient_qbox` | 合格观测算术均值 |
| 超声声速 | `speed_of_sound_qbox` | 合格观测算术均值 |

按participant_id＋cohort＋research_stage汇总，使用通过必要QC的有效测量，缺失不补0；全部缺失时该表型保持缺失。每个测量行等权，保留有效测量数用于QC，不作为第七个结局。前四类采用Q-box mean列，再对访视内合格行求均值；衰减和声速对其原值求均值。

现有汇总表前四类的mean_mean列是可复用候选，需核对纳入测量及QC和上述口径一致，不能把mean_median或median_median混作本次均值。没有运行新的个体表型构建，W10尚未完成；六类独立分析不强制全部六项完整。

本地配置已更新；服务器规则快照仍为这两次明确选择之前的版本，正式构建前必须同步当前配置。
