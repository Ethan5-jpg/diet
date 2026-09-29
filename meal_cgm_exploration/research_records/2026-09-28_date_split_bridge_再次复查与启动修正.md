# 上传前再次复查：date_split_bridge2026.09.28.2

用户要求再次自查后再上传。复查发现并修复启动早期失败后可能显示旧结果的问题；当前可上传整个date_split_bridge文件夹，放在服务器meal_cgm_exploration/内，与response_followup同级。

## 复现与修复

v1在检查前置目录、读取来源JSON后才建立运行目录。复现：缺少前置目录时运行返回2，但随后show latest返回0并显示已有旧成功摘要。该问题不会改变已有模型，但可能让用户误认旧结果为本轮结果。

v2先保护输出位置，再用标准库写入UNVERIFIED/RUNNING记录；缺少文件、来源JSON损坏等早期失败都会留下FAILED及ERROR.txt，latest拒绝展示旧成功结果。输出位置不可写或本身无效时不能登记，以当前错误为准，只在完整分析成功后查看摘要。另修复创建输出目录失败后错误处理再次写入不存在目录引发第二个异常的问题。

统计方法、日期切分、人数门槛、指标公式、24项主分析及两组次要24项完全未改。旧response_followup等冻结包未改。

## 验证

- 原19项完整测试再次通过；修复后含8项新增边界检查共27项全部通过，无跳过。
- SYNTHETIC完整端到端运行72/72均可估计，14项旧可行性人数一致。
- 新旧模拟结果的models、scales、flow、legacy_concordance、selection逐项精确相同；科学合同仅版本元数据变化。
- 新增检查覆盖四种早期来源失败、历史输出保护、无效输出路径、零合格样本、带空格上传目录和不同工作目录下启动。
- 最终73个包与依赖文件哈希全部匹配；归档更新前268项长度/SHA256全部核对，v1包和旧入口文档备份在history。
- Python3.8语法通过；实际测试解释器仍为本地Python3.12.14，真实服务器运行待用户上传后确认。

## 上传与查看

[上传文件夹](../date_split_bridge/) · [README](../date_split_bridge/README.md) · [具体测试记录](../date_split_bridge/REVIEW.md)。只需上传这一个文件夹；已上传过v1则用当前整个目录更新。版本2026.09.28.2。

在服务器原pheno环境、meal_cgm_exploration目录运行：

```bash
bash date_split_bridge/run_date_split_bridge.sh --followup-run outputs/response_followup_20260927T155611_920880Z --data-root ..
```

成功后显示3份摘要：

```bash
python3 date_split_bridge/run_analysis.py --show latest --brief
```

只回传3份关键摘要；报错只发末尾。不需要逐页回传明细。本地没有新增真实模型结果、截图或研究图；模拟数据不加入真实结论。
