# V6：小文件退出码 0，旧进程仍阻塞

来源：用户服务器截图，标注时间 2026-11-07 16:31:12（未独立核验服务器时钟）。

原图已保存为 `still_blocked.png`；SHA256：`99e0c4ccf0181a1e017376b91650bc3403aa8d2b104cd69426c002bda96e878d`。

可见输出：

```text
probe_exit=0
PID    ELAPSED   TIME      STAT
24812  01:01:13  00:05:48  Dsl
```

进程命令行为 `/home/ec2-user/miniconda/envs/pheno/bin/python -u run.py --run-id main_20261107 --threads 4`。`/proc/24812/syscall` 仍显示调用号 1、首参 0x1、计数参数 0x1d，对应此前确认的向 fd=1 日志写入 29 字节。地址参数与前一张 syscall 截图一致；完整内容保留在原图。

`command -v tmux` 无路径输出，并返回命令提示符，说明当前 PATH 未发现 tmux；不能据此排除其他位置存在安装。

判断：同目录新文件读写完成且退出码为 0，旧任务却仍在原日志写入处等待，累计 CPU 时间仍为 5分48秒。此时继续按大样本计算慢来估时没有依据。原日志持续写入/读取与 FUSE 实现的具体问题仍未定根因，尚无真实数据 R² 或已完成外折数。

下一步在服务器终端向本批 PID 24812 发送 TERM 退出请求，等待 3 秒后核实状态：

```bash
kill -TERM 24812
sleep 3
ps -p 24812 -o pid,etime,time,stat,args
```

TERM 并不保证不可中断等待立即结束。若 PID 仍存在，保留输出，暂不启动第二个同运行编号任务，也不处理旧锁。若 PID 已退出，下一步才核对登记/锁和已完成缓存，选择终端输出续跑；保留原日志及全部缓存，尚未完成的计算允许重算，不改变模型参数、源码和评估口径。

此记录仅给出由用户执行的下一步命令，本地未向服务器发送信号，也未观察到任务退出。

参考：[kill 信号语义](https://man7.org/linux/man-pages/man1/kill.1.html)、[FUSE 中断行为](https://docs.kernel.org/filesystems/fuse/fuse.html)。
