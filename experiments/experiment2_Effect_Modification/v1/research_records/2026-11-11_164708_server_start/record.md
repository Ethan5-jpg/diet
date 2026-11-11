# 101 条路线服务器启动

2026-11-11 16:47:08 的终端截图显示，运行 `experiment2_effect_modification_20261111_v1_101` 通过预检并进入真实发现集训练。启动命令复用了 V12 保存的 Python 解释器。

```text
EXP2_UPLOAD_CHECK_OK files=15
EXP2_ENV_OK numpy=1.26.4 scipy=1.11.4 pandas=2.1.0 pyarrow=15.0.2 scikit-learn=1.4.1.post1 threadpoolctl=3.3.0
EXP2_CONFIG experiment2_effect_modification_20261111_v1_101 routes=101
EXP2_PREFLIGHT_DONE | synthetic checks only; no native rows read or writes
EXP2_SOURCE_READY discovery=4427 people=3343 households=3059
OUTER_COMPLETE r0 f0 people=661 seconds=72.3
OUTER_COMPLETE r0 f1 people=677 seconds=67.1
SEARCH r0 f2 101 routes; common penalty=1
```

截图时已完成十折中的两折，第 0 轮第 3 折正在内折选型。图中没有最终 R² 或 ΔR²。

（终端原图存于本地归档） · [此前的 PyArrow 预检记录](../2026-11-11_163909_pyarrow_preflight/record.md)

[16:59 十折完成及主结果](../2026-11-11_165942_primary_results/record.md)
