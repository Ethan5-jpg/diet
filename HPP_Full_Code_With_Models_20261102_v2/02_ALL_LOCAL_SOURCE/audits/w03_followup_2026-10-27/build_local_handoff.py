"""Local aggregate-only record of observed TRE liver schema inventory and user rules."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';Q=P+'/data_qc/w03_liver_schema_20261027'
families=[('弹性','elasticity',5,10,'elasticity_qbox_median','kPa'),('剪切波速度','velocity',5,10,'velocity_qbox_median','m/s'),('黏度','viscosity',3,6,'viscosity_qbox_median','Pa·s'),('频散','dispersion',3,6,'dispersion_qbox_median','(m/s)/kHz'),('衰减系数','attenuation',1,0,'attenuation_coefficient_qbox','dB/cm/MHz'),('超声声速','speed_of_sound',1,0,'speed_of_sound_qbox','m/s')]
observed={'date':'2026-10-27','source':'Manual transcription of current metadata-only TRE outputs, not source-file download','scope':'/home/ec2-user/studies/hpp_datasets/liver_ultrasound and subdirectories only','recursive_traversal_complete':True,'entries_seen':4,'parquet_files':2,'other_file':'metadata/liver_ultrasound_data_dictionary.csv','source_rows_read':0,'actual_exits':{'01_schema_probe':0,'03_recursive_inventory':0,'02_independent_readback_and_policy':0},'files':[{'name':'liver_ultrasound.parquet','rows_metadata':26881,'physical_columns':28,'pandas_index_columns':['participant_id','cohort','research_stage','array_index'],'measurement_columns':18,'measurement_families':6},{'name':'liver_ultrasound_aggregated.parquet','rows_metadata':8937,'physical_columns':36,'pandas_index_columns':['participant_id','cohort','research_stage','array_index'],'measurement_columns':32,'measurement_families':4}],'families':[{'label':label,'prefix':prefix,'raw_field_count':raw,'aggregate_field_count':agg,'raw_representative_field':field,'unit':unit} for label,prefix,raw,agg,field,unit in families],'artifact_hashes_rechecked':9,'no_final_six_field_selection':True,'no_individual_phenotype_values_or_models':True,'sample_selection_executed':False,'policy_recorded':True,'evidence':'liver_inventory_verified.png','probe_code_sha256':hashlib.sha256((D/'01_check_liver_schema.py').read_bytes()).hexdigest(),'inventory_code_sha256':hashlib.sha256((D/'03_inventory_liver_parquets.py').read_bytes()).hexdigest(),'local_policy_sha256':hashlib.sha256((R/'config/w03_followup_20261027/visit_and_sampling_policy.json').read_bytes()).hexdigest()}
(D/'observed_metadata_summary.json').write_text(json.dumps(observed,ensure_ascii=False,indent=2)+'\n')
lines=['# W03补核：Liver ultrasound全部Parquet文件与访视规则','', '2026-10-27。通过HPP浏览器工作区在当前服务器执行。只读目录、Parquet元信息和数据字典，未读取个体临床数值。','',
'## 1. 目录确实全部遍历了','',
'范围：`/home/ec2-user/studies/hpp_datasets/liver_ultrasound`及其子目录。递归遍历完成，共4个目录项：2张Parquet、metadata目录及其1份CSV字典。没有发现第三张Parquet或只有6列的表。未把这一结果扩张成“整个HPP或其他研究目录也不存在自建六列表”。','',
'| 文件 | 元信息行数 | 物理总列数（包含索引） | 肝测量列数 | 测量类别数 |','|---|---:|---:|---:|---:|',
'| `liver_ultrasound.parquet` | 26,881 | **28** | 18 | **6** |',
'| `liver_ultrasound_aggregated.parquet` | 8,937 | **36** | 32 | **4** |','',
'两个文件均包含4个存储索引列：participant_id、cohort、research_stage、array_index。若软件将这些恢复为DataFrame索引，显示的数据列数会分别为24与32，不能与物理总列数混用。行数不是独立人数。','',
'你之前截图选中的右侧aggregated确实是汇总表，但**列更少的是左侧明细表**。汇总文件体积较小也不能证明列更少，因为它的行数更少。','',
'## 2. 六类测量分别在哪里','',
'下表的“代表字段”用于说明来源位置，不代表本轮已经选定最终六个研究结局。','',
'| 类别 | 明细列数 | 汇总列数 | 明细中的代表字段 | 当前字典单位 |','|---|---:|---:|---|---|']
for label,prefix,raw,agg,field,unit in families:lines.append(f'| {label} | {raw} | {agg} | `{field}` | {unit} |')
lines+=['',
'**当前汇总表没有衰减系数和超声声速列。** 补核不是“没找到汇总表”，而是实际列清单与“六项都在汇总表”的记忆不符。两项均在明细表，因此还没有理由据此宣布肝脂相关数据不可用。','',
'汇总表的32个测量列由弹性10列、剪切波速度10列、黏度6列、频散6列构成。多个列来自mean／median及区域内统计量的不同组合，例如 `elasticity_mean_median_of_qboxes` 与 `elasticity_median_median_of_qboxes`；它们不是同一个定义。不能按列名顺序随手挑六列，也不能将剪切波速度当作超声声速。','',
'如研究确实要六类各一个指标，可核对既有六列清单；若没有既定清单，则需要明确四类汇总值的取法，并为衰减／声速规定从明细按访视生成研究汇总的方法。那是后续表型构建，本轮没有偷偷执行，也没有指定mean或median。','',
'## 3. 本轮已记录的用户规则','',
'| 项目 | 当前规则 | 执行边界 |','|---|---|---|',
'| Body、肝脏、CGM访视汇总表型 | 同participant_id＋cohort＋research_stage连接；不以具体timestamp、固定日期距离或唯一测量日作为准入要求 | 仍核身份、单位和表型质量；日期可保留作描述，不用于因此排人 |',
'| 饮食—CGM日级／逐餐模型 | 日级仍按真实日期与点级覆盖；逐餐仍需进食时间依据 | same visit不能替代逐日对齐；不把现有共同日均值与新的访视总均值混作同一指标 |',
'| 同访视多个菌群样本 | 正确连接样本身份、完成必要QC后随机选一个；种子20261027，固定一次，各域复用 | 记录了规则，尚未建立或执行样本选择名单；599个未匹配肠道键不能由随机选择解释 |',
'| 饮食local_timestamp | 不作为准入必需字段；当前主日期采用与带Asia/Jerusalem时间戳一致的collection_date | 1066个日期冲突保留诊断，未假称已解释 |',
'| CGM“14天” | 用户根据经验回忆，登记为待核，而不是逐条实测事实 | 没有证明那5组连接各为两个完整14天窗口 |','',
'**饮食日索引的数量修正：**是14713个事件日期键没有日索引，不是114731，也不是“没有日期或没有食物”。与149912个匹配键相比，全源事件日期键共164625个，未匹配比例为8.937%。这是全源日期键口径，尚未复算基线人群／饮食—CGM实际分析中受影响比例。用户提出若很少可剔除受影响日；目前没有冻结主分析删除规则，因为这些日存在带日期食物事件。只保留索引匹配日可以另外明确为候选或敏感性视图，不把未匹配当零摄入。','',
'上述规则保存在[本地配置](../config/w03_followup_20261027/visit_and_sampling_policy.json)。这是用户澄清的实施补充，优先于早期相冲突的日期准入解释；旧V3文件、旧运行结果和历史日期统计不回写。尚未重建分析表、改变正式检验预算或生成新的分析N。','',
'## 4. 跑了什么、代码和Result在哪里','',
'| 代码 | 功能 | 实际状态 |','|---|---|---|',
'| `01_check_liver_schema.py` | 核两表schema／footer，读取当前模块字典，建立来源证据 | 实际exit0 |',
'| `03_inventory_liver_parquets.py` | 遍历整个liver目录及子目录；读取每个Parquet全部列名、类型和Pandas索引信息 | 实际exit0 |',
'| `02_verify_liver_and_save_policy.py` | 独立读回退出记录、来源指纹、源码hash；核6类代表字段单位、两表清单；保存用户规则与MEMO | 实际exit0；9产物hash |','',
f'服务器代码：`{P}/scripts/w03_liver_schema_20261027/`。','',
f'服务器Result：`{Q}/`，包括 `all_liver_parquet_inventory_TRE_ONLY.json`、`source_metadata_TRE_ONLY.json`、`dictionary_evidence_TRE_ONLY.json`、`verification_and_family_inventory.json`、`artifact_manifest.json`、`MEMO.md`。','',
f'服务器规则：`{P}/config/w03_user_clarification_20261027/visit_and_sampling_policy.json`。','',
f'服务器MEMO：`{P}/memos/w03_liver_schema_20261027/MEMO.md`。运行退出记录在服务器本地 `L/w03_liver_schema_20261027`（L=`/home/ec2-user/multisystem_runtime_20261021`）。','',
'本地自编代码／聚合转录／截图／MEMO：[核查目录](../audits/w03_followup_2026-10-27/MEMO.md)。完整来源schema和字典文件仍留TRE，本地并非它们的直接导出。','',
'## 5. 验收画面','',
'![肝脏全部Parquet遍历与字典单位读回](../audits/w03_followup_2026-10-27/liver_inventory_verified.png)','']
report='\n'.join(lines);(R/'reports/HPP_W03_Liver_Table_Inventory_2026-10-27.md').write_text(report)
memo=f'''# W03后续代码备忘录\n\n2026-10-27，在当前HPP服务器通过Chrome/WorkSpaces/pheno执行。\n\n01_check_liver_schema.py：两张肝表schema与当前字典；03_inventory_liver_parquets.py：模块目录及子目录全部Parquet遍历；02_verify_liver_and_save_policy.py：独立读回、单位、hash和用户规则保存。三次实际exit均0。\n\nResult：{Q}\nCode：{P}/scripts/w03_liver_schema_20261027\nMEMO：{P}/memos/w03_liver_schema_20261027/MEMO.md\nConfig：{P}/config/w03_user_clarification_20261027/visit_and_sampling_policy.json\n\n结论：模块仅2张Parquet，raw26881×28、aggregated8937×36（物理列含4索引）。raw包含6类18测量列；aggregated仅4类32测量列，没有衰减和声速。六个最终结局尚未指定，未读个体数值、未构造汇总或运行模型。\n\n已记录Body/liver/CGM访视汇总same person/cohort/stage准入和菌群QC后随机选一个规则；未执行新合表或抽样。日级CGM日期对齐保留；缺饮食日索引比例8.937%全源，尚未冻结剔除；旧V3正文与历史运行不改，当前实施补充见本地config/w03_followup_20261027/visit_and_sampling_policy.json。\n\n本地observed_metadata_summary.json为屏幕聚合结果人工转录；完整schema与字典留TRE。\n''';(D/'MEMO.md').write_text(memo)
print('Local liver inventory report, metadata summary and MEMO written.')
