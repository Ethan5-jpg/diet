# V7 环境创建失败与前台运行修正

用户截图（2026-11-07 23:31:18，Asia/Shanghai）显示 `UPLOAD_CHECK_OK`，随后执行 `bash setup.sh`，报 `Errno 38 Function not implemented: lib -> .../.venv/lib64`。原图为 setup_error.png，可见内容见 visible_results.json。

错误发生于创建虚拟环境的符号链接步骤；没有依赖安装完成、训练启动或性能结果证据。原 setup.sh 将环境建在研究项目的挂载目录内，未兼容该目录的符号链接限制。已修正本地安装脚本，默认使用服务器本机 `/home/ec2-user/.venvs/hpp_v7_20261107`，保留模型、输入与结果的原位置。没有移动任何参与者数据至用户电脑。

用户随后明确选择前台终端运行，并说明关闭当前虚拟机网页不影响其会话。本轮使用 `launch.py --console`，不要求 tmux，不改变模型网格、目标或验证流程。

服务器已上传的 setup.sh 仍是旧版；为避免重复上传，本次给出 RUN_FOREGROUND.sh 中的直接终端命令，绕过旧安装位置。保留挂载目录内未建完的 .venv，不要求删除或复用。

本地 shell 语法检查与路径/参数模拟安装检查通过。模拟覆盖含空格的环境路径及工作目录切换，不等于 Linux 原生依赖安装成功。完整合成回归测试输出见本目录 validation_tests.txt。下一项远端证据仍需用户执行命令并提供截图；本地没有操作服务器。
