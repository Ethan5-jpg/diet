# 2026-11-03 广度实验：运行完成，结果包待正式导出

## 已完成

- 源码、启动记录、验证程序、运行后审计程序及聚合结果导出程序已提交到本目录。
- HPP 服务器上的五折实验已完成：54 个模型族、126 组设置、630 次候选拟合。
- 运行退出码为 0；五折累计耗时依次为 433、810、1154、1510、1855 秒。
- 运行后审计显示 `BREADTH AUDIT PASS`、`SOURCE HASHES PASS`、`ALL PREDICTION SCORES RECOMPUTED`。
- 原始部署源码标识仍保留在启动记录中。后续 GitHub 提交没有改写历史运行的源码登记。

## 文件交付状态

| 内容 | 状态 |
| --- | --- |
| 本轮源码、方法说明、启动记录、验证与审计程序 | 已提交 GitHub |
| 完整模型比较表、候选参数成绩、诊断、编码说明、运行日志与审计收据 | 已在 HPP 生成，尚未提交 GitHub |
| 聚合结果包与全部产物清单 | 已在 HPP 的 `public_export/` 生成，待平台允许的结果导出流程 |
| 个体预测、标签、家庭标识及拟合对象 | 未公开；不包含在聚合结果包内 |

服务器实验根目录：

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_response_breadth_20261103_v2
```

重点文件：`postrun_audit.json`、`results/model_comparison.csv`、
`public_export/aggregate_results.json`、`public_export/model_comparison.csv`。

本次会话中，远程桌面的剪贴板未能传出聚合结果，My Files 未显示下载入口；
Pheno 管理页面显示会话过期。未绕过平台导出限制，也未把未传回的结果包记为已发布。
后续应恢复 Pheno 登录，使用平台允许的输出流程取得聚合包，核对 SHA256 后再补交。

## 代码隔离与后续解释

本轮所有 GitHub 新增内容限定在 `experiments/diet_response_breadth_20261103_v2/`。
没有修改学长原来的代码或上一轮实验目录，也没有删除服务器文件。

本轮属于探索性筛选：参数选择和成绩报告使用同一组五折。后续应在未参与选择的
数据上确认方向。通用预测模型的整体提升不能直接当作饮食效应异质性的证据；
显式交互模型应与具有相同非线性主效应的无交互模型比较。所有候选与优化警告均保留。
