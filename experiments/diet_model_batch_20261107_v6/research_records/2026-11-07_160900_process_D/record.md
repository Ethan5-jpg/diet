# V6：进程存在，当前主线程处于 D 状态

来源：用户提供的服务器截图，标注时间 2026-11-07 16:09:00（未独立核验服务器时钟）。

原图已保存为 `process.png`；SHA256：`bb754ed15cbd6daff9cd3e573bdd124a4317884d22b1c608cad54b87d1902dec`。

原始显示值：

| PID | ELAPSED | TIME | %CPU | %MEM | STAT |
|---|---|---|---:|---:|---|
| 24812 | 39:03 | 00:05:48 | 14.8 | 9.4 | Dsl |

命令行：`/home/ec2-user/miniconda/envs/pheno/bin/python -u run.py --run-id main_20261107 --threads 4`。新终端的检查命令已返回提示符。

判断：进程存在，截图瞬间主线程处于不可中断等待（D，通常为 I/O 等待）；s 表示会话首进程，l 表示多线程。TIME 为累计 CPU 时间；%CPU 是运行期间平均值，不能证明截图时正在计算。9.4% 的进程内存占比不足以判断整机内存是否充足。单次 D 状态也不足以证明永久阻塞，其他线程仍可能工作。

结合此前 tail 无返回，存储/文件系统读写是需要优先核查的候选原因；尚未定位具体文件、设备或服务器故障。未见训练阶段、外折完成数或真实 R²。此前半天至一天的估时不是测量值，在此状态下不能用于剩余时间预测。

下一步仅进行只读检查：

```bash
ps -p 24812 -o pid,etime,time,pcpu,stat,wchan:36
vmstat 1 5
cat /proc/24812/io
```

核对等待点、累计 CPU 时间是否增长、系统 I/O 等待/换页以及进程 I/O 计数；当前不重启训练、不修改模型源代码、不删除锁或缓存。

参考：[Linux 进程状态](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)、[ps 字段语义](https://man7.org/linux/man-pages/man1/ps.1.html)、[等待点](https://man7.org/linux/man-pages/man5/proc_pid_wchan.5.html)、[进程 I/O 计数](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html)。以上用于解释系统字段，截图值为用户提供的当前观测。
