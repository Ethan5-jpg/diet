# HPP官网核查：功能数据、日期定义与读取入口

日期：2026-11-09。按用户要求，以HPP/Pheno.AI公开数据字典和平台文档为数据查找边界。未访问用户服务器，未下载参与者数据，没有把未发表结果、人数或用户S3地址提交到公开搜索。本记录更新数据准备判断，不改写旧截图和V12结果。

## 已核实的结论

**口腔功能数据在官网明确列出，后续优先查聚合Arrow主要表。** 官网同时列出逐样本TSV/Parquet支持文件；前面服务器访问测试针对后者。聚合字段的官方位置仍是口腔主索引，实际表格式、访问和原人群覆盖没有由本次网页核查验证。[官方口腔字典](https://knowledgebase.pheno.ai/datasets/070-oral_microbiome.html)

**肠道公开可用性清单未列可直接调用的HUMAnN功能表。** 官网相关研究确实使用过肠道功能谱，因此不能写成HPP完全没有肠道功能数据；论文使用过与当前目录已提供现成表是不同证据。按本轮用户范围，暂不将未定位到直接入口的肠道功能表作为可用输入，也不另行重处理原始测序来补造当前数据条件。[肠道字典](https://knowledgebase.pheno.ai/datasets/013-gut_microbiome.html)、[HPP官方论文说明](https://knowledgebase.pheno.ai/publications/12fc42c7-a9e4-807c-8cd4-ffe82a9c6d5d.html)

**日期的字典层含义已经查到。** 口腔和肠道collection_date表示交样日期，肠道collection_timestamp为交样时刻；人类遗传模块同名日期却指下载Gencove结果的日期，不能跨数据集套用。网页定义不自动验证每条历史记录，也不建立采样先于CGM或因果方向。[口腔字典](https://knowledgebase.pheno.ai/datasets/070-oral_microbiome.html)、[肠道字典](https://knowledgebase.pheno.ai/datasets/013-gut_microbiome.html)、[遗传字典](https://knowledgebase.pheno.ai/datasets/014-human_genetics.html)

官方软件文档提供PhenoLoader的字典/bulk读取接口和TRE既有配置说明。它是后续核对路径的依据，不是已在用户当前环境验证的修复方案；不能承诺换SDK就能解决超时。[读取文档](https://phenoai.github.io/pheno-utils/pheno_loader.html)、[配置入口](https://phenoai.github.io/pheno-utils/)

## 后续工作的具体入口

优先的三类非菌种分层聚合字段，以及另外三类分层字段，精确名称和格式已整理在 [official_function_manifest.json](official_function_manifest.json)。保留来源familites拼写；官网标注的格式为Arrow，不按Parquet页脚读取，也不在未核实前假定一定是某一种Arrow IPC/Feather编码。具体单位在公开字典未给出，不能自行补填。

首版泛文件名搜索的后缀白名单遗漏了.arrow，原来的“未定位”也未覆盖这一格式；后续逐样本Parquet抽查同样没有测试这些聚合入口。旧脚本和截图不覆盖重写，此处明确收紧它们的解释范围。选择聚合表改善查找目标与数据粒度，不证明它必然有本地映射或能绕过当前S3读取问题。

接下来只围绕这些官网已列字段核对当前TRE的既有映射和读取方法，先确认一个聚合表的结构，再在TRE内匹配原锁定样本和固定3343人。不重复前面的逐样本S3查询，不随机扫目录，不改写平台配置或升级冻结运行环境。无法匹配或质量不足时先报告不可用，不临时换样本或声称修饰阳性。

官网关联论文用于判断资源来源、处理层级和方法，不将可能使用同一HPP人群、已按代谢结局挑选的关联特征冒充独立先验或验证。新的功能修饰分析仍属发现期探索，候选需基于明确注释和研究问题另行固定。

## 检索边界

已核对官方主站、口/肠/遗传数据页、相关官方论文说明和平台软件文档。主站链接的Google Drive目录PDF只读到标题/查看器，未成功获取正文；未拿它证明任何缺失。部分软件源码链接失效；当前服务器安装版本、权限、配置和聚合文件实际位置均未由公开网页验证。

完整来源与限制见 [source_ledger.json](source_ledger.json)。本次仅存字段事实、链接和自主判断，没有复制完整网页、论文或数据字典。
