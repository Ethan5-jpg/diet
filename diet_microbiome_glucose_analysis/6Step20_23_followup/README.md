# HPP Step20–23 后续分析运行包

上传整个 `6Step20_23_followup` 文件夹，放在服务器 `5Step19b_core_taxa_direction` 旁边。原有文件夹不需要覆盖。

## 一键运行

在此前成功运行Step19b的`pheno`环境中执行：

```bash
cd /home/ec2-user/Desktop/HPP/Data/diet_microbiome_glucose_analysis/6Step20_23_followup
bash run_followup.sh
```

运行不需要联网，不重跑回归或bootstrap。每次建立新的`outputs/followup_时间戳`，结束后显示结果目录。4470条样本核查期间会分批打印进度。

**完成后先把终端最后的 `HPP STEP20-23 FOLLOW-UP` 摘要截图发回。** 再按需要显示各部分：

```bash
python3 scripts/show_results.py --section 20
python3 scripts/show_results.py --section 21
python3 scripts/show_results.py --section 22
python3 scripts/show_results.py --section 23
```

每次运行还生成`aggregate_results.zip`，只包含汇总表、图、核查记录和报告，不打包参与者数据矩阵。可以下载此文件供完整核查；截图能看概览，压缩包用于精确复核。

## 本次准备了什么

| 阶段 | 实际执行内容 | 主要文件 |
|---|---|---|
| Step20 | HPP定义字典、25字段来源和Z核查、20/25项相关性及pairwise N、领域和结局分母/保留率、Mean/GMI冗余 | `20_*` CSV、相关图、领域图 |
| Step21 | 原三模型逐路径N及样本哈希4470次核查、协变量检查、已有全局FDR对照、raw/clean清洗记录、比例解释提醒 | `21_*` CSV、完整1490×3模型补充表 |
| Step22 | 全部109菌的路径背景、机制证据矩阵、已核对的4篇文献起点、功能/代谢组文件盘点 | `22_*` CSV及机制说明 |
| Step23 | 核心菌方向图、三模型原CI森林图、E. rectale NOVA4/hPDI对应结局展示、Methods/Results草稿 | `23_*` PDF/PNG及报告 |

Step22是整理好的证据核查材料，**不是已经完成全部菌的系统文献综述或功能验证**。论文结论仍需结合服务器真实输出收尾；本地测试数据从未混入正式outputs。

## 自动定位输入

程序从所在目录推断HPP的`Data`路径；默认读取已确认的Step19b运行：

```text
5Step19b_core_taxa_direction/outputs/step19b_20260920T030809_033720Z
```

先核对其manifest、输入和输出SHA256，再读取其中锁定的三套原始模型及1490条候选。若该目录不存在，仅当恰好有一个正式PASS运行时使用它；不会按最新修改时间猜选。多个候选时可显式指定：

```bash
bash run_followup.sh --step19b-run /完整路径/step19b_具体时间戳
```

参与者来源与此前18b/18c/18e2代码一致，详见`docs/输入路径与输出索引.md`。CGM表从三套模型的manifest获取，并要求文件内容一致。原始字段来源默认`Data/Transfer/cgm/iglu.csv`。仅路径不同时可指定：

```bash
bash run_followup.sh --data-root /完整路径/HPP/Data --iglu-csv /完整路径/iglu.csv
```

`--cgm-csv`只能指定与模型所引用文件内容一致的副本，不能用它替换成另一次处理的数据。原模型指向的CGM文件缺失时会停下该依赖部分并报告。

## 如何读运行状态

- `PASS`：本包的计算和输入核查通过；不表示因果机制或历史IGLU版本已被证实。
- `REVIEW_REQUIRED`：至少一个阶段失败、字段未核实或主动跳过。先截图摘要，不修改规则凑数。
- 字段raw匹配或clean→Z核查失败时，相关性阶段标记`SKIPPED_DEPENDENCY`；独立的冻结结果汇总可继续输出。
- `--no-plots`仅供诊断，记录为跳过图形，不当作全部完成。
- 出错的详情在`logs`，输入指纹和各阶段状态在`manifest.json`。

SDw已按官方定义更正为日内SD；COGI较高是评分设计上的较佳方向；GRADE eugly是GRADE贡献比例，不是TIR。历史HPP计算版本/阈值以及SDw新增清洗过程若缺记录，仍需补齐；这些边界已写入报告。

## 环境与验证

只需NumPy、pandas和matplotlib。此前服务器版本为1.26.4、2.1.0和3.8.3，通常可直接运行。若确实缺依赖，再在当前Python环境运行：

```bash
python3 -m pip install -r requirements.txt
```

代码按Python 3.8语法编写。开发测试及图形检查见`docs/本地验证记录.md`；本地没有服务器参与者数据，因此没有声称真实结果已经跑完。

自测（可选，使用临时合成数据；不会读取或改动研究数据）：

```bash
python3 -m unittest discover -s tests -v
```
