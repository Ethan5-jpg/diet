"""Finalize audit documentation in the user-approved server-only directory."""
from pathlib import Path
import json, hashlib, sys, datetime

T=Path('/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z')
R=Path('/home/ec2-user/studies/hpp_datasets')
assert '/envs/pheno/' in sys.executable
for name in ['03_exit_code','04_exit_code','05_exit_code']:
    assert (T/name).read_text().strip()=='0', name
s=json.loads((T/'aggregate_summary.json').read_text())
j=json.loads((T/'joint_window_feasibility.json').read_text())
m=json.loads((T/'source_manifest.json').read_text())
assert hashlib.sha256((T/'03_time_alignment.py').read_bytes()).hexdigest()==m['source_sha256']
assert all(x['rows_scanned']==x['rows_metadata'] for x in m['sources'])
assert j['n']==s['four_source']['unique_microbiome_dates']
signatures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(T.glob('0[1-6]_*.py'))}
context=dict(status='COMPLETE_SERVER_DIRECTORY_ACCEPTED',active_directory=str(T),raw_root=str(R),
             python=sys.executable,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             latest_user_instruction='Temporarily keep everything in the independent server audit directory.',
             cloud_writes_enabled=False,cloud_write_issue='Historical attempts failed; no further cloud copy is required for the current task.',
             successful_exit_codes={'03':0,'04':0,'05':0},retained_failed_version={'02':1},source_sha256=signatures,
             rerun_instruction='Historical scripts retain their original CLOUD/PROJECT constants. Before a new run, explicitly set the output to the active server directory or a new server-only run subdirectory; do not retry cloud writes.',
             limits=['Observed-day audit, not full raw-point CGM QC','No diet completeness or meal timestamp truth certification',
                     'No abundance matrix QC','Sample dates follow current source dictionaries','No causal or prospective-baseline claim'])
(T/'RUN_CONTEXT.json').write_text(json.dumps(context,indent=2,ensure_ascii=False))
notice='''# 当前保存约定（2026-10-18用户确认）\n\n本核查暂时全部保存在服务器独立目录，不再尝试云盘写入。当前目录：
`/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z`。
云盘错误属于历史操作记录，不是当前任务仍需完成的复制步骤。
旧脚本保留实际运行时的路径常量与哈希；后续重跑前显式把输出设置到本目录或新建的服务器运行子目录，停止使用历史云盘复制逻辑。
最新机器可读上下文见RUN_CONTEXT.json，代码功能和修改入口见下方备忘录。
\n---\n\n'''
memo=(T/'MEMO.md').read_text()
if not memo.startswith('# 当前保存约定'):
    memo=notice+memo
memo+='\n## 06_finalize.py：收尾登记\n\n核验03/04/05完成状态、扫描行数与源码哈希；保存RUN_CONTEXT.json、README.md和不含个体标识的final_review.txt。不读取新的参与者数据，不再尝试云盘。\n'
(T/'MEMO.md').write_text(memo)
readme='''# HPP饮食、CGM与双部位菌群：时间核查\n\n状态：核查完成，按用户最新约定保存在当前服务器独立目录。\n\n## 先读\n\n- MEMO.md：代码功能、修改入口、方法、完整聚合结果、失败记录与局限。\n- RUN_CONTEXT.json：当前目录、环境、源码哈希、保存约定。\n- aggregate_summary.json：日期交集及采样间隔。\n- joint_window_feasibility.json：双部位同时接近记录期的交集。\n- resolved_source_locations.json：原始数据文件指针；含可能带个体编号的CGM文件名，仅留TRE。\n\n## 代码\n\n01_discover.py：有界盘点与schema。\n02_time_alignment.py：首次严格扫描，失败版本保留，不作为最终结果。\n03_time_alignment.py：五源完整投影日期扫描、连接选择、实际共同日期与双部位采样间隔。\n04_verify_and_locate.py：完成状态、输入行数、哈希和18个产物验收。\n05_joint_windows.py：共同时间窗交集、字典编码文件指针与3份CGM分片元数据。\n06_finalize.py：当前约定和验收结果收尾登记。\n\n## 原始数据\n\n根目录：/home/ec2-user/studies/hpp_datasets\n\n- diet_logging/diet_logging_events.parquet：饮食事件。\n- cgm/iglu_daily.parquet：实际日指标记录日期。\n- cgm/cgm.parquet：连接汇总与时间序列指针。\n- cgm/timeseries/*.parquet：原始时间序列，文件名可能含个体编号，依connection_id核对。\n- oral_microbiome/oral_microbiome.parquet：口腔样本日期和结果指针。\n- oral_microbiome/oral_microbiome_2025_04_UUID.parquet：口腔另一版本。\n- oral_microbiome/aggregated/metaphlan_abundances_aggregated/species.parquet：口腔物种矩阵。\n- gut_microbiome/gut_microbiome.parquet：肠道样本日期和结果指针。\n- gut_microbiome/abundance/metaphlan4/aggregated/abundance_species.parquet：肠道物种矩阵。\n\n原始数据只读；参与者级日期审计CSV仅留TRE。\n'''
(T/'README.md').write_text(readme)
near30=next(x for x in j['rows'] if x['max_nearest_day_gap']==30)
review=[
    'STATUS: COMPLETE_SERVER_DIRECTORY_ACCEPTED',
    'DIRECTORY: '+str(T),
    'CLOUD WRITES: DISABLED BY LATEST USER INSTRUCTION',
    'AUDIT EXIT CODES: 03=0, 04=0, 05=0; failed version 02 preserved',
    'PROJECTED SOURCE SCANS: 5/5 complete; source SHA256 verified',
    'DIET + CGM same visit: '+str(s['diet_cgm_same_visit']),
    'ACTUAL COMMON DAYS >=1: '+str(s['actual_common_days_at_least']['1'])+'; >=7: '+str(s['actual_common_days_at_least']['7']),
    'FOUR SOURCES, UNIQUE SAMPLE DATES: '+str(j['n']),
    'MEDIAN NEAREST DAY GAPS, ORAL diet/CGM: 67/69 days; GUT: 9/10 days',
    'BOTH SITES <=30 days from BOTH streams, common days>=7: '+str(near30['plus_common_days_7']),
    'BOTH SITES on/before first CGM day, common days>=7: '+str(j['both_sites_on_or_before_first_cgm_day_and_common_days_7']),
    'ENTRY DOCUMENTS: README.md, MEMO.md, RUN_CONTEXT.json',
    'LIMIT: observed-day alignment; not full raw-point/meal/abundance QC',
]
(T/'final_review.txt').write_text('\n'.join(review)+'\n')
(T/'06_exit_code').write_text('0\n')
assert (T/'README.md').stat().st_size>0 and (T/'RUN_CONTEXT.json').stat().st_size>0
print('\n'.join(review),flush=True)
print('FINALIZATION_EXIT_CODE_0',flush=True)
