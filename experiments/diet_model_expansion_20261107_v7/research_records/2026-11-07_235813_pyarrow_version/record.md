# V7 原生依赖安装成功，预检被 PyArrow 版本阻止

来源：用户两张 2026-11-07 23:58:13、23:58:23（Asia/Shanghai）截图。原图为 install_and_checks.png、pyarrow_error.png；可见数值与状态在 visible_results.json。

截图显示服务器本机 V7 环境创建成功，LightGBM 3.3.5、Interpret Core 0.6.5 安装成功。26 项检查耗时 54.898 秒，25 项通过、1 项报错。宽菌群正式候选网格、各模型拟合/重载、PCA 一致性等原生检查均为 ok；唯一错误为 runtime 集成检查执行 environment(config) 时，配置期望 pyarrow 15.0.2，而实际为 15.0.0。check.py 返回 1，launch.py 因此停止，尚未出现正式数据训练证据。

原 requirements.txt 仅列两个新模型包，错误地假设 pheno 基础包均已与配置对齐。已修正为完整列出 config.json 中七个固定依赖，并新增“安装清单与配置一一对应”的测试，保留严格版本检查。模型、目标、网格与家庭划分不改动。

官方发布文件提供 CPython 3.11、Linux x86_64 的 manylinux_2_17 / manylinux2014 pyarrow 15.0.2 wheel，可由 pip 根据服务器平台选择；不用编译源码。来源：[PyArrow 15.0.2 官方发布文件](https://pypi.org/project/pyarrow/15.0.2/#files)。这属于兼容性核对，实际服务器安装仍待用户执行。

ALIGN_AND_RUN.sh 保存本次终端命令：用 V7 独立 Python 对齐全部固定依赖，保留 pheno 原环境，再以前台 --console 启动。用户无需重新上传代码；已上传的 requirements.txt 仍是原先两项版本，临时通过明确 pip 参数补齐，避免继续引用旧清单。

本地新增回归检查后共 27 项，完整日志在本目录 validation_tests.txt；服务器当前已上传测试仍为 26 项，重启会运行该 26 项，并非测试丢失。上传目录模拟与源文件哈希另见本目录 validation.json。没有远程操作或下载服务器数据，没有真实 v7 R²。

后续更新：2026-11-08 00:15 截图确认 PyArrow 15.0.2 安装成功、26 项预检全部通过并进入正式训练，见 [启动记录](../2026-11-08_001549_running/record.md)。
