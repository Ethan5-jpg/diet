# 2026-10-11 13:30 CGM读取失败与修复

来源：用户提供的服务器终端截图，文件名时间为2026-10-11 13:30:52（Asia/Shanghai）。原图已原样复制并校验一致；完整可见文本保存在[逐行转录](screen_transcription.txt)。最上方被裁切的一行未补造。

![服务器报错](截屏2026-10-11%2013.30.52.png)

## 截图确认的事实

- 最后可见进度：`CGM files 2220/9497; cached 2,154,633 rows`。
- 报错：`DAILY STOPPED: Cannot cast DatetimeArray to dtype float64`，随后返回pheno终端提示符，运行已经停止。
- 失败目录：`/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person/outputs/daily_20261011T052609_243544Z_944c1f`。
- 截图没有展示完整traceback、manifest或版本行。与前次2026-10-11.2运行指令关联，但不将版本视为截图直接核实的信息。
- 2220是最后打印进度，不是已知的具体故障文件编号。9497是匹配到的CGM文件数，缓存行是尚未完成日级质控的原始记录数，不是模型样本量。
- 本次尚处CGM原始读取阶段，未生成日级模型结果；没有新的beta、CI、p/q或研究结论。

## 本地复现和修复

在pandas 2.1.0及2.3.2均复现：空的时区datetime Series经`map(timestamp)`仍保留datetime类型，随后`.astype(float)`抛出与截图完全一致的错误。旧CGM读取先按参与者筛选，再按最终唯一connection_id筛选；同一参与者其他连接的整文件/数据块会在第二次筛选后清空，但未立即跳过。

版本2026-10-11.3同时修复两层：

1. `daily/common.py::epoch`明确创建float64数组，空输入返回空浮点Series；缺失时间仍为NaN，保留原索引与列名。
2. `daily/build.py::ingest_cgm`在最终连接筛选后检查空表；已按规则排除的其他连接行保留在计数中，不参与时间解析，也不因其他连接的坏时间影响所选连接。

没有放宽质量要求、合并传感器连接或修改统计方法。新增5个用例在修复前为3失败/2通过；修复后完整36项测试通过。本地pheno合成全流程84/84模型完成，全部正常输入估计、区间、p/q与前版在1e−12精度下相符。完整环境及验证见[LOCAL_VALIDATION](../../docs/LOCAL_VALIDATION.md)。这只能确认已复现路径的修复，真实服务器重跑成功仍待后续截图。

## 服务器重跑指令

将本地整个`daily_within_person/`新版代码重新上传，覆盖同名代码文件；保留服务器已有`outputs/`。失败缓存不是完整构建，不使用`--stage analyze`续算，不删除失败历史。

```bash
conda activate pheno &&
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person &&
python -c 'from daily import VERSION; print("CODE_VERSION=" + VERSION); assert VERSION == "2026-10-11.3", "Please upload version 2026-10-11.3"' &&
PYTHON=python bash run_analysis.sh \
  --data-root /home/ec2-user/Desktop/HPP/Data \
  --stage all
```

本命令新建输出目录；完成后回传DAILY 1/3、2/3、3/3及RESULT_DIR。若仍报错，记录新错误，不把旧缓存或合成输出作为真实结果。

## 后续清理决定

用户随后询问能否删除之前的输出并要求清理本地旧文件。该次失败运行没有模型结果，其服务器输出目录可删除；原图、转录和本页保留。服务器删除动作由用户执行，本地没有该失败目录的副本。旧版本地合成运行已按[清理记录](../../docs/CLEANUP_2026-10-11.md)删除，最新代码和最新合成验证保留。
