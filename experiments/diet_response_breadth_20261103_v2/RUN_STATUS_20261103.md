# 2026-11-03 广度实验：运行完成，聚合包已进入 Egress Store

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
| 聚合结果包与全部产物清单 | 已生成并复制到正式 Egress Store；页面已显示文件，尚未取得下载文件 |
| 个体预测、标签、家庭标识及拟合对象 | 未公开；不包含在聚合结果包内 |

服务器实验根目录：

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_response_breadth_20261103_v2
```

重点文件：`postrun_audit.json`、`results/model_comparison.csv`、
`public_export/aggregate_results.json`、`public_export/model_comparison.csv`。

先前会话中，远程桌面的剪贴板未能传出聚合结果，My Files 未显示下载入口，
Pheno 管理页面登录过期。2026-11-03 晚间用户重新登录后，已继续执行正式导出流程：

1. 确认工作区 `dietgutcgmv3` 的 Egress Store 挂载路径。
2. 将聚合 JSON 与模型比较 CSV 复制到独立子目录，未复制个体文件或拟合对象。
3. 核对文件字节数和 SHA256；Pheno 页面随后显示约 433 KB、13 KB 两个对象及目录占位对象。
4. 点击 `Submit Egress Request`，页面显示 `Egress Request submitted`。这表示页面已接受请求操作，不代表已获导出批准。
5. 尚未看到批准状态或下载入口；因此两个结果文件目前仍未提交 GitHub。

导出子目录：

```text
/home/ec2-user/studies/egress-store-ccca276b-c9a1-4c19-88dd-471d674de477/diet_response_breadth_20261103_v2_aggregate
```

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| aggregate_results.json | 443153 | e8c3f69e1b76a00c27d288de87b97607199d328d4f45387271b0e00569da3366 |
| model_comparison.csv | 13301 | 5e17fcef57226315ad434d3ebd38bfcdd3a41db52de2b26fc655613c17faafa8 |

操作历史：最初在文件复制前点击过一次导出请求，页面日志提示没有新增对象；
随后完成上述文件复制，再提交请求。未绕过平台导出限制，也未把结果包记为已发布。
后续使用平台允许的输出流程取得上述两个文件，核对 SHA256 后补交本实验目录。

## 代码隔离与后续解释

本轮所有 GitHub 新增内容限定在 `experiments/diet_response_breadth_20261103_v2/`。
没有修改学长原来的代码或上一轮实验目录，也没有删除服务器文件。

本轮属于探索性筛选：参数选择和成绩报告使用同一组五折。后续应在未参与选择的
数据上确认方向。通用预测模型的整体提升不能直接当作饮食效应异质性的证据；
显式交互模型应与具有相同非线性主效应的无交互模型比较。所有候选与优化警告均保留。
