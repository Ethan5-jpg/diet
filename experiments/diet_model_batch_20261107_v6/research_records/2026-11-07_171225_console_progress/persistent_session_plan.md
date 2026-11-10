# 网页断开与持久终端方案

用户说明：服务器位于美国，像虚拟机，通过网页连接。推测为网页远程桌面，但未确认平台品牌或断开/注销策略。不能把关闭浏览器等同于关闭服务器，也不能保证页面关闭不会导致终端会话结束。

当前已确认进展：console_20261107，PID 29534（17:12 截图），已完成部分内层候选，尚无外层分数。当前入口使用前台终端，现有进程尚未进入 tmux。仅安装 tmux 或新建一个会话不会自动迁移已有进程。

第一阶段在另一个服务器终端准备 tmux，原任务保持运行：

```bash
sudo yum install -y tmux
tmux -V
```

安装未成功前不停止原任务。

待工具确认可用后，回到原训练终端按一次 Ctrl+C，等待 Python 的 finally 清理完成并返回命令提示符。再检查 PID 29534 已退出，且 runs/console_20261107/RUNNING.lock 已移除；实际 PID 如已变化，应以当前已验证的运行 PID 为准。若锁仍在或进程未退出，不自动删除锁，不并行启动同编号任务。

确认正常停止后创建持久会话：

```bash
tmux new -s hpp_v6
```

进入 tmux 后，用相同运行编号、4 线程和原科学配置续跑：

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_batch_20261107_v6/launch_console.py --run-id console_20261107 --resume --threads 4
```

启动器检查现有锁、运行登记、源码、输入和环境；Memo 只复用校验通过的已完成候选，正在计算且未保存的候选可能重算。tmux 提供伪终端，符合 launch_console.py 的 TTY 检查，进度留在终端会话中，不重新引入原 FUSE 文件日志。

在 tmux 中观察到启动与推进后，按 Ctrl+B、松开，再按小写 d 脱离；会话保留在服务器。之后可关闭网页连接。重连后查看：

```bash
tmux attach -t hpp_v6
```

这针对连接断开，不承诺在虚拟机重启、会话被平台整体销毁或用户主动结束 tmux 时继续。当前只提供方案，尚未观察安装、停止、迁移或脱离动作。

依据：[tmux 官方手册](https://man.openbsd.org/tmux.1)、[Amazon Linux 2 软件安装](https://docs.aws.amazon.com/linux/al2/ug/find-install-software.html)。
