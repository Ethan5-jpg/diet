# V6：KILL 后进程仍存在，下一步查内核等待

来源：用户服务器截图，标注时间 2026-11-07 16:36:26（未独立核验服务器时钟）。

原图已保存为 `kill_pending.png`；SHA256：`1d8b2b1ea9ddeee89851c6e8d0d525e9918380f4f107fc00c9f023ba176c0e51`。

用户已执行：

```bash
kill -KILL 24812
sleep 3
ps -p 24812 -o pid,etime,time,stat,args
```

可见输出：

```text
PID    ELAPSED   TIME      STAT
24812  01:06:33  00:05:48  Ds
```

命令行仍为本批任务：`/home/ec2-user/miniconda/envs/pheno/bin/python -u run.py --run-id main_20261107 --threads 4`。未见 kill 报错，ps 已返回提示符。截图没有显示待处理信号位，不能据此确认 SIGKILL 正在排队、已被内核处理或卡在退出清理中的哪一步；可以确认 PID 尚未消失。

此前已定位 write(fd=1) 对原日志的等待；项目目录为 FUSE 挂载。同目录一次性小文件读写已成功、退出码 0，因此不能简单断言整个挂载不可用。

下一步只读系统诊断：

```bash
awk '/^(Name|State|Pid|PPid|Threads|SigPnd|ShdPnd):/' /proc/24812/status
timeout -k 2s 5s sudo -n cat /proc/24812/stack
```

读取本批 PID 的状态/信号及内核调用栈，不打开参与者数据。sudo -n 不交互请求密码；若权限不足或内核未提供 stack 文件，保留报错即可。不反复发送终止信号，不绕过旧任务锁，不启动同运行目录的第二个任务；平台挂载维护需评估其他作业影响，不自行卸载或中断整个 FUSE 连接。

已准备 [平台维护说明](../PLATFORM_IO_HANDOFF.md)。旧日志与检查点保留，未观察到旧进程退出或训练恢复；尚无真实数据 R²。

参考：[内核调用栈](https://man7.org/linux/man-pages/man5/proc_pid_stack.5.html)、[状态及信号字段](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)、[FUSE 官方说明](https://docs.kernel.org/filesystems/fuse/fuse.html)。
