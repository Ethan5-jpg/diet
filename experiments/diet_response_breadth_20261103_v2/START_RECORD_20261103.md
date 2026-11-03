# HPP 第二轮：广度模型比较启动记录

记录时间：2026-11-03，北京时间约17:19。状态：RUNNING，不是完成报告。

## 当前进度

- 本地与HPP环境的合成验证均PASS。
- 首折已完成全部54个模型家族、126个候选设置，用时433秒。
- 第二折已进入计算。最后检查时拟合文件163份，其中包括变系数特征映射文件，不能把文件数当作已完成拟合数。
- 已确认训练子进程18934、启动器18853；检查时RSS约446124 KiB。
- 该轮尚无最终五折成绩，不能声称某新算法获胜。

## 代码与版本

- 基于远程main：0123506777d748c8eced4e1be2995282ffe40fe8。
- 本地源码提交：8a8181efdf1d4a69e6b412d9d526dcb920fc4989，尚未推送。
- 部署payload SHA256：d9aa1a4a103b7c21fbdc19741479c8775a65b2c3674bb4a3314d754bb217473e。
- 各源码哈希见source_sha256.json。运行登记使用UNPUBLISHED_SOURCE_SHA256加上述payload哈希。
- 本轮GitHub create_file被自动审批拒绝，原因为未识别本轮向公开仓库默认分支发布的授权；没有改用浏览器或其他通道绕过。本次拒绝发生在GitHub API执行之前，不表示重新验证过应用403问题。
- postrun_audit.py是运行后复核工具，编译检查通过；在本记录时间尚未部署或执行，不属于首轮部署payload。

## 服务器路径

数据包根：

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1
```

相对数据包根的新实验位置，与GitHub工作副本中的相对位置相同：

```text
experiments/diet_response_breadth_20261103_v2/
```

该目录保存源码、source_sha256.json、deployment.json、launch_pid.txt、logs/、results/。
这次直接放置源码，不使用code/子目录。新目录创建，没有覆盖旧代码或结果，因此没有旧版备份对象。
v1依赖位于数据包下experiments/diet_response_heterogeneity_20261103_v1_attempt02/code。
core.py及run_pilot.py在部署时与本地Git版本逐字节哈希匹配。

## 运行与检查

启动器：pheno环境Python，launch.py，显式--package、--pilot-code、--code-commit参数。
完整实际命令与实时日志路径保存在logs/execution.json；实时日志在服务器临时目录。
完成后启动器会把日志保存为logs/run.log，并写logs/actual_exit_code.txt。
日志每完成一个模型家族写MODEL_DONE，每折结束写FOLD_DONE；最终写BREADTH_COMPLETE。
每折结束独立保存checkpoint_foldN_TRE_ONLY.npz。原始数据与个体产物均留在TRE。

完成后需确认退出码0，再部署并运行postrun_audit.py，重算全部候选损失、选择结果、汇总R2及匹配无交互增量。
MLP及其他方法的优化警告保存在results/diagnostics.json，不应忽略收敛警告后直接解释排名。
结果表：results/model_comparison.csv；完整汇总：results/summary.json。
本轮为广泛开发筛选，沿用家庭五折；后续再选少量方向深入。
