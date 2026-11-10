# V6：阻塞位置为日志写入，项目挂载类型为 FUSE

来源：用户提供的服务器截图，标注时间 2026-11-07 16:16:19（未独立核验服务器时钟）。模块 `diet_model_batch_20261107_v6`，后台任务 PID 24812。

原图已保存为 `log_write_fuse.png`；SHA256：`1fc4eebd709f8a6319056219f4f1a3371cdc690ef471d0fc117cbaf31fa08001`。

`/proc/24812/syscall` 可见开头为 `1 0x1`，第三个参数为 `0x1d`（29 字节）；其余参数含内存地址，诊断摘要省略，完整值保留在原图中。此前下载的包名显示 x86_64 平台，该架构 Linux 系统调用号 1 为 write；第一个参数 0x1 为文件描述符 1。

打开文件映射：

| fd | 可见目标 |
|---:|---|
| 0 | `/dev/null` |
| 1 | 正式 v6 目录下 `main_20261107.log` |
| 2 | 同一 `main_20261107.log` |
| 3 | `/home/ec2-user/miniconda/envs/pheno/lib/libffi.so.8.1.2` |

截图中的挂载摘要：

```text
/ xfs
/home/ec2-user/studies/diet_gut_cgm_v2 fuse
/home/ec2-user/studies/diet_gut_cgm fuse
/home/ec2-user/studies/hpp_dataset fuse
/home/ec2-user/studies/egress-store-ccca276b-c9a1-4c19-88dd-471d674de477 fuse
/home/ec2-user/thinclient_drives fuse.xrdp-chansrv
```

当前定位：截图瞬间进程处于对标准输出 fd=1 的 write 调用，目标为日志文件。结合此前 D 状态、CPU 时间停滞及 tail 无返回，日志写入阻塞的证据较明确。实验目录属于 FUSE 挂载。尚不能确定 FUSE 守护进程、底层网络/服务、单个文件状态或其他具体原因，也不能据此断言整个挂载均不可用。

尚无新模型真实 R² 或外折完成数；本地算法测试和服务器 21 项合成测试的通过状态不变。此时不修改模型源代码，不删除已有日志、运行锁或缓存，不启动第二个训练实例；不进行卸载或中断 FUSE 连接的操作。

后续只进行同目录新文件的最小读写测试，脚本为 [probe_io.sh](probe_io.sh)。文件名由 mktemp 唯一创建，内容仅为固定的公开测试标记 HPP_V6_IO_PROBE，不打开参与者输入/预测/模型文件。先写出创建成功、写入关闭成功、读取完成各阶段标记；timeout 发出 TERM/KILL 以限制等待，但不可中断 I/O 的底层恢复仍取决于文件系统。测试小文件保留作为诊断证据。

验证：脚本通过 bash 语法检查，内部正常路径在本机临时目录读写成功；这不是对服务器 FUSE 的测试结果。

恢复分支（待截图后选择）：若新文件可正常读写，进一步检查原任务状态及已有检查点，再选择新日志或终端输出续跑；若新文件也阻塞，优先请平台维护方修复该挂载。终止旧任务和续跑必须先核对进程状态，防止两个任务同时写同一运行目录。只处理本批任务，保留既有产物。

参考：[Linux x86_64 调用表](https://raw.githubusercontent.com/torvalds/linux/master/arch/x86/entry/syscalls/syscall_64.tbl)、[/proc/PID/syscall 字段](https://man7.org/linux/man-pages/man5/proc_pid_syscall.5.html)、[FUSE 官方说明](https://docs.kernel.org/filesystems/fuse/fuse.html)。
