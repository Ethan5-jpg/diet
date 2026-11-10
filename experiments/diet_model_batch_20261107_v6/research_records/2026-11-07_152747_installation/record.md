# V6 依赖安装记录

来源：用户提供的服务器终端截图；截图标注时间为 2026-11-07 15:27:47（未独立核验服务器时钟）。模块为 `diet_model_batch_20261107_v6`；未见运行编号。

原始截图已复制为 `installation.png`；SHA256：`1b58feceefd6a79e34c2592cfb7fb4ee1782932f7e148fce5eeae1c62f58134e`。

截图显示在正式 v6 目录运行 `python -m pip install -r requirements.txt`，终端最后一条安装结果为：

```text
Successfully installed catboost-1.2.8 graphviz-0.21 narwhals-2.28.0 plotly-7.1.0 xgboost-2.0.3
```

已有环境版本可见：NumPy 1.26.4、pandas 2.1.0、SciPy 1.11.4、Matplotlib 3.8.3。安装后已返回命令提示符，截图未见安装错误。

结论：依赖安装成功。尚未见启动、21 项测试、训练进度或真实数据 R²；本记录不代表模型已运行。

下一步：运行 `python launch.py --threads 4`，通过测试后检查日志中的后台训练进度。
