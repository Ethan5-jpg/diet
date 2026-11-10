# HUMAnN 3.6与HPP说明核对，2026-11-10

[HPP口腔字典](https://knowledgebase.pheno.ai/datasets/070-oral_microbiome.html)明确聚合表来自HUMAnN 3.6；页面未给出这些表的空值编码或单位。

[3.6版官方手册](https://raw.githubusercontent.com/biobakery/humann/3.6/readme.md)说明：零丰度通路不输出；coverage对非零丰度通路提供检测置信信息，不是“多少百分比反应存在”或功能活性。丰度与覆盖度共享报告条目，后者可以记录为0。该规则支持“部分Arrow空格可能来自未报告通路”的假设，尚未直接验证HPP的聚合转换。

[3.6版计算源码](https://raw.githubusercontent.com/biobakery/humann/3.6/humann/quantify/modules.py)写出两个特殊总量，并将其coverage设为1。丰度特殊量由缩放常数计算，该常数在无可用通路总量时可能为0。因此六人若有coverage特殊量1值，只能支持存在相容的输出标记，不能称作生物通路已检出或QC通过。

[PhenoLoader官方页面](https://phenoai.github.io/pheno-utils/pheno_loader.html)的定向fillna文本查找未命中；仅表明该页未检索到此说明，不证明平台从未执行填补。未采用第三方解释、旧版论坛讨论或current master规则替代3.6。后续仍需明确本地单位/归一化及缺测处理假设。
