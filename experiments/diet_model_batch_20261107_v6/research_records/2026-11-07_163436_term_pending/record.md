# V6：TERM 后进程尚未退出

来源：用户服务器截图，标注时间 2026-11-07 16:34:36（未独立核验服务器时钟）。

原图已保存为 `term_pending.png`；SHA256：`4168d9b8a6bd77e3501a6251b360f9d24644da5df1d5280a67bf6107a16cae96`。

用户已执行：

```bash
kill -TERM 24812
sleep 3
ps -p 24812 -o pid,etime,time,stat,args
```

截图显示：

```text
PID    ELAPSED   TIME      STAT
24812  01:04:43  00:05:48  Ds
```

命令行仍为本批原任务：`/home/ec2-user/miniconda/envs/pheno/bin/python -u run.py --run-id main_20261107 --threads 4`。终端未显示 kill 错误，检查命令已返回提示符，但这不能证明退出已完成。状态从 Dsl 变为 Ds，多线程标志不再显示，主进程仍存在且 CPU 累计时间没有变化；不据此推断精确的内核清理进度。

下一步仅对已核实的本批 PID 尝试 KILL，并核对退出状态：

```bash
kill -KILL 24812
sleep 3
ps -p 24812 -o pid,etime,time,stat,args
```

强制结束不执行 Python finally 清理，可能留下运行锁或不完整候选文件。现有 Memo 使用校验收据复用完整候选，没有完成收据的候选可重算；已有缓存和旧日志不删除。尚未核验缓存完整性，不能保证所有未完成计算均可复用。只有确认旧进程退出后才检查本批残留锁并准备续跑。

若 KILL 后仍为 D，不能断言信号无效或已退出，也不重复启动并发任务。需进一步由平台维护方定位/恢复 FUSE 挂载等待；不自行卸载挂载、不杀其他服务、不删除旧锁来绕过存活进程。

此时只有退出尝试，未观察到 KILL 的执行结果、旧任务退出或新任务启动；未见真实数据 R²。模型源码与配置未改。

参考：[kill 手册](https://man7.org/linux/man-pages/man1/kill.1.html)、[FUSE 中断行为](https://docs.kernel.org/filesystems/fuse/fuse.html)。
