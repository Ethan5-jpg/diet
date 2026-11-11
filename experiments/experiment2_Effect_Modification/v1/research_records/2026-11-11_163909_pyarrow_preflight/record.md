# 101 条路线首次服务器预检

2026-11-11 16:39 的终端截图显示，上传清单的 15 个文件通过校验。预检在检查 Python 环境时退出，随后回到终端提示符。该次启动没有训练输出。

```text
EXP2_UPLOAD_CHECK_OK files=15
EXP2_STOP RuntimeError: Use the existing pinned HPP environment; version mismatch: pyarrow=15.0.0 (need 15.0.2)
```

当前启动脚本优先使用 `pheno`。V12 的运行记录显示其解释器通过了 PyArrow 15.0.2 版本检查。V12 的 `setup.sh` 将当时使用的解释器写入 `.python_path`。重试时读取该文件，显式将同一解释器传给 Experiment 2。服务器仍需执行预检确认。

（终端原图存于本地归档）

[16:47 重试通过并进入训练](../2026-11-11_164708_server_start/record.md)
