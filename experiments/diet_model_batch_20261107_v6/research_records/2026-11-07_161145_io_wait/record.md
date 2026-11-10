# V6：累计 CPU 时间不变，系统存在 I/O 等待

来源：用户服务器截图，标注时间 2026-11-07 16:11:45（未独立核验服务器时钟）。

原图已保存为 `io_wait.png`；SHA256：`987fe421ee8b5924c3e1305ca9ae29da0f01eac7e64dfd5fdd0267a7f01b3d61`。

进程状态原始显示值：

```text
PID    ELAPSED  TIME      %CPU  STAT  WCHAN
24812  41:39    00:05:48  13.9  Dsl   -
```

相比 16:09 截图（ELAPSED 39:03，TIME 00:05:48），本次已运行时间增加 2 分 36 秒，但累计 CPU 时间在秒级显示精度上未增加。两次采样主线程均为 D；不能据两张瞬时截图断言整个区间始终处于同一内核等待。

`vmstat 1 5` 原始显示值：

```text
r b swpd free    buff cache   si so bi bo  in   cs   us sy id wa st
2 1 0    9633168 2088 3899060 0  0  1  4   11   31   0  0 99  0  0
0 1 0    9619788 2088 3910580 0  0  0  148 1283 1972 3  1 71 23  0
0 1 0    9624856 2088 3910580 0  0  0  0   1050 1581 3  1 73 24  1
0 1 0    9622960 2088 3910580 0  0  0  0   1044 1553 3  1 72 24  1
0 1 0    9622976 2088 3910580 0  0  0  0   1112 1540 3  1 72 24  1
```

注意：第一行 CPU/IO 等速率是启动以来平均值；其后四行才是每秒采样。后四行均 b=1，CPU 空闲 71%–73%，I/O 等待 23%–24%，si=so=0，空闲内存约 9.2 GiB（按该命令默认 KiB 口径解读）。暂未见内存耗尽或活跃换页证据，不能据此证明全系统无其他资源问题。

进程 I/O 计数：

```text
rchar: 60463525
wchar: 2135743
syscr: 4411
syscw: 154
read_bytes: 0
write_bytes: 0
cancelled_write_bytes: 0
```

这些值仅有一次采样。rchar/wchar 为成功读写调用的累计字符量，read_bytes/write_bytes 为存储层计数；后者为 0 不能推出从未读取数据，缓存和文件系统类型会影响解释，也不能据此确定具体阻塞文件。

当前判断（推断）：持续读写等待是主要排查方向，依据为跨截图 CPU 时间无变化、D 状态、系统 I/O 等待，以及此前 tail 无返回。尚未确定磁盘、网络挂载、文件系统或具体代码位置，也未证明实验逻辑有错。训练阶段、外折完成数和真实 R² 仍未观察到。

下一步：仅查看系统调用、已打开文件的符号链接以及 /proc 中的挂载信息，不打开参与者数据文件。命令在当前服务器终端任意目录执行：

```bash
timeout -k 2s 5s cat /proc/24812/syscall

timeout -k 2s 5s ls -l --color=never /proc/24812/fd

awk '$2 == "/" || $2 ~ /^\/home(\/|$)/ {print $2, $3}' /proc/24812/mounts
```

如有权限不足，保留该错误输出，不自行升级权限。保留现有训练进程，不重复启动、不修改模型源代码、不删除锁或缓存。

参考：[vmstat](https://man7.org/linux/man-pages/man8/vmstat.8.html)、[进程 I/O 计数](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html)、[进程打开文件](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html)。
