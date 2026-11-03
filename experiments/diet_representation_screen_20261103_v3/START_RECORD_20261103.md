# V3 启动记录：已启动，尚未完成

2026-11-03，用户确认继续进行模型和特征表示对照，并授权网页提交 GitHub。

- 运行代码提交：`fbac248693ec58acf8d5841786636595acdec6bf`。
- 四份源码/说明文件通过 GitHub 网页提交，远端内容与本地逐字节一致。
- 本地和 HPP `pheno` 环境均通过 `verify_screen.py`。
- 在新的独立目录部署，部署时核对全部源码 SHA256。
- 已启动脱离交互终端的后台任务，launcher PID 为 `5158`。
- 计划 52 个模型族、182 组设置、五折共 910 次候选拟合。
- 本记录写入时，日志已推进至首折的肠道 PCA50 模型；仅宿主、口腔全部输入版本及肠道 PCA10/30/50 已完成。
- 截至该位置，已执行的仅宿主和口腔 CLR 树模型预测与 v2 核对通过。
- 尚无完整五折结果和最终审计结论；本文件不是完成报告。

## 服务器位置

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_representation_screen_20261103_v3
```

`start.json` 记录启动时间、PID 和 launcher 日志路径。
`logs/execution.json` 记录实际命令、源码提交和实时日志路径。
完成后查看 `logs/actual_exit_code.txt`、`results/audit.json`、`results/ANALYSIS.md`。
全部模型需完成 60 组 v2 预测重现核对，最终审计还会重算全部 910 个验证折损失。

通过最终审计后，launcher 会生成聚合结果包、CSV、分析报告和 SHA256 清单，
并复制到该工作区正式 Egress Store 的 `diet_representation_screen_20261103_v3_aggregate/`。
结果进入导出区不等于获准下载或已经提交 GitHub，仍需平台允许的导出流程。

## GitHub 与 HPP 导出的区别

GitHub 网页提交已成功，代码提交不受 HPP 文件下载问题影响。
HPP 内部生成的结果则需要先经过平台导出，才能取得可上传的文件。
上一轮 v2 的聚合 JSON/CSV 已进入正式导出区并发起导出请求，尚未取得下载文件。
不能把“代码已提交”或“导出区已出现文件”写成“全部结果已上传”。

本轮仅新增当前实验目录，未修改学长原代码及 v1/v2 实验内容。
