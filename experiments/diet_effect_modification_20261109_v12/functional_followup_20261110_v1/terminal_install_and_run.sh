#!/usr/bin/env bash
bash <<'HPP_FUNCTION_INSTALL_AND_RUN'
set -euo pipefail
hpp_function_parent="${HPP_FUNCTIONAL_PARENT:-/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12}"
cd "$hpp_function_parent"
hpp_function_target="$PWD/functional_followup_20261110_v1"
if [ -L "$hpp_function_target" ]; then
    echo 'STOP: functional directory is a symlink.'
    exit 1
fi
mkdir -p "$hpp_function_target"
hpp_function_python=python
if [ -s .python_path ]; then
    IFS= read -r hpp_function_python < .python_path
fi
hpp_function_write() {
    local hpp_name="$1"
    local hpp_expected="$2"
    local hpp_temporary
    hpp_temporary="$(mktemp "$hpp_function_target/.incoming.XXXXXXXX")"
    cat > "$hpp_temporary"
    "$hpp_function_python" -B - "$hpp_function_target" "$hpp_name" "$hpp_temporary" "$hpp_expected" <<'HPP_VERIFY_INCOMING'
import hashlib,os,sys
from pathlib import Path
folder,name,incoming,expected=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),sys.argv[4]
try:
    if Path(name).name != name or name in ('.','..'):
        raise SystemExit('STOP: invalid source filename.')
    if hashlib.sha256(incoming.read_bytes()).hexdigest() != expected:
        raise SystemExit('STOP: incoming source checksum mismatch.')
    target=folder/name
    if target.is_symlink():
        raise SystemExit('STOP: existing source is a symlink.')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit('STOP: differing existing functional source; nothing overwritten: '+name)
    else:
        os.replace(str(incoming),str(target))
finally:
    if incoming.exists():
        incoming.unlink()
HPP_VERIFY_INCOMING
}
hpp_function_write 'IMPLEMENTATION_PLAN.md' 'ba8fb78c33145fe08eac0ce7895519b59a6463e6cd87c64a6ccbb24bd7ca58fb' <<'HPP_FUNCTION_SOURCE_000_END'
# Functional follow-up implementation plan

Goal: deliver one bounded, reproducible TRE pilot for the already authorized diet–function question, with a fixed primary route and two explicitly secondary encodings.

Architecture: reuse the verified source-record audit and frozen V12 background/ridge implementations through read-only imports; implement only functional representations, technical covariates, nested household orchestration and aggregate reporting in this isolated subdirectory. No edits to the original 51 files. Local synthetic tests only; the user runs native code.

Tech stack: the existing pinned numpy/scipy/pandas/pyarrow/scikit-learn/threadpoolctl environment.

1. Preserve the 00:53 screenshot and displayed values in the V12 records (completed before implementation).
2. Add `audit_driver.py`, extracted unchanged from the verified null-contract command; `native.py` invokes it with explicit existing project roots, captures aggregate audit output inside the TRE, and requires 3337 predefined usable records. Read both numeric matrices and technical metadata by verified source row. Keep six statuses and source masks.
3. Add `functional.py` with training-fitted representations, technical encoder and an adapter to the frozen V12 design and matched ridge functions. Tests cover scale invariance, held-out mutation, missing values, unseen/tied ranks, null vs zero, technical interactions, heredity and exact no-interaction equivalence.
4. Add `pilot.py` with inner/outer family isolation, frozen selection order, fixed population checks, checkpoint identities, aggregate paired household uncertainty and complete reporting. Tests include known synthetic interaction/null scenarios, no held-out outcome selection, tampered checkpoints/source identity, missing/small groups, constant functions and negative score preservation.
5. Add a native CLI and a separate clearly labeled synthetic CLI path; no native imports execute local HPP reads. Save all actual arrays under `runs/<id>/TRE_ONLY/`, report no individual rows. Registration includes all copied/frozen adapter, code and source inputs.
6. Generate a complete reviewable terminal installer that writes only this child folder, skips identical existing sources, refuses differences and runs checked code. Copy to clipboard after successful verification. Preserve the original V12 upload manifest (51 files).

No Git repository is present in this workspace, so there is no worktree/commit step. User authorization already covers preparation and server execution of the follow-up; routine implementation choices do not introduce a new permission gate.
HPP_FUNCTION_SOURCE_000_END
hpp_function_write 'PROTOCOL.md' 'bb845b41d395cf6748a401e7077b1c86f52432e2f16c4bfc815326c1091ab123' <<'HPP_FUNCTION_SOURCE_001_END'
# 口腔功能与饮食关联异质性：探索性预测试验 v1

方案时间：2026-11-10，00:53数据准备截图之后、任何本模块功能—结局拟合之前。研究者已经看过同一发现集V4–V12结局结果；本方案不是独立确认或前瞻注册。旧V12的51份冻结文件和结果不修改。

## 问题、目标人群与已知限制

在原发现集的个人平均纤维密度—D_Glu关联中，口腔HUMAnN功能表能否提供超出相同主效应背景的、家庭隔离折外可复现的交互预测信息？结果只涉及关联/预测，不是同一次餐食反应、个体因果响应、菌群功能活性或肠道发酵能力。

原固定资格3343人/3059家庭的来源记录均通过核对。六人在去除UNMAPPED/UNINTEGRATED后485个候选功能项全为空；虽然有软件输出标记及正读段计数，本试验不把六人制造成生物全零谱。主分析为其余3337人的功能可用子群，完全由00:53的测量可用性定义，不看新关联选择人群。保留六人的状态与原3343人历史结果，不比较旧3343人与新3337人的R²；每条路线的主效应参照必须在同一3337人和同一折内重新拟合。

## 测量与三条固定路线

候选字典固定为提供表中除两个特殊总量外的485项；不根据本次全人群“观察值内常量”计数预删19/92项，也不全局筛掉十一项。训练角色内按非零/已报告信号比例>=10%筛选，再训练PCA，最多10个主效应PC，前三个作为交互候选。没有可估计变异时保留零维表示，明确报告，不选其他编码替代。

1. **主要路线 abundance_relative**：部分通路空格按“未报告信号=0”的工作假设编码；对提供的非负丰度信号按行闭合，再开平方，训练角色内中心化PCA。这里定义的是提供信号的相对谱，既不声称已验证RPK/CPM，也不把它当代谢通量。该表示对逐样本正的共同倍数不变。
2. **敏感性 coverage_confidence**：部分空格按未报告0编码，提供的0–1覆盖度分数开平方后训练PCA。不把它称反应完成比例或酶活性；该路线与丰度路线共享零遗漏假设，不能单独验证此假设。
3. **空值假设敏感性 abundance_unknown_null**：保留未报告状态；每项仅用训练中有限观察值拟合经验中位秩映射，未知值填训练观察秩中位数，另加入报告/缺失指示。训练内标准化及PCA。不把未知当生物0。该路线检验表示/编码依赖，不可事后改为主要路线。

依据3.6版官方手册与源码，零丰度条目省略、两表报告条目相同；00:53二表Arrow掩码一致，但HPP聚合编码未独立认证。上述编码是明确假设，两条敏感性完整报告，不择优。

## 匹配背景与验证

读取已完成V12的受校验源代码；沿用V6/V4发现集读取和训练内预处理，以及V12的显式主效应背景（宿主、饮食非线性、口/肠分类组成与生态主效应、已有非菌群背景及其饮食项）。每条路线另加本路线功能PC主效应和前三个PC的训练分位数样条主项，以减少把非线性功能主效应误判为饮食交互的风险。raw_read_count、metaphlan4_aligned_read_count及其比值采用训练内缺失处理/标准化，作为主效应和饮食技术交互同时进入该路线的两个比较模型，以减少把测序深度相关差异误称功能修饰的风险。这不是排除全部技术混杂。

保留原两轮5折家庭分配；只在功能可用子群的训练角色拟合全部预处理。每外折内再按家庭3折选择主效应岭惩罚（0.01、0.1、1）。随后在固定该主效应惩罚下，从无交互和交互岭惩罚0.03、0.3、3中选择。新增项仅为标准化线性纤维项×前三个功能PC，主效应保留这些PC，交互先以训练主效应设计投影残差化。两比较模型主效应结构/惩罚相同，不用旧V11融合模型或旧分数作参照。

外层验证结果绝不参与特征、阈值、惩罚或路线选择。原资格已经由历史结果固定，因此本试验复用原最短95%核心区间的训练稳健限值，但对所有原入选成员只执行训练5SD截尾，不再按新训练限值实施8SD剔除；避免验证结局改变已固定名单。原始非有限结局不填补，记录原规则的极端值标记供审计。实现仅在本进程的数据适配器替换目标变换，不修改原源码或V12结果。原预处理若使预定训练/验证成员发生变化则停止，不默默删人。各路线有各自匹配主效应参照，不能跨路线相减或挑最大分数。

## 输出与停止规则

主要读数为第0轮abundance_relative的折外ΔR²，另报相同参照的MSE差；第1轮为同人分折敏感性，不是独立重复。三条路线的两轮主效应R²、交互R²、Δ、所有负值、未选择交互的折数均完整显示。以家庭成对自助法1999次给出固定折外预测条件下的95%区间，明确不涵盖重训/选型不确定性；不把这些区间或预测改善当正式修饰效应P值或因果证据。

补充输出同样预测对原始Y的敏感性评分、10→15 g/1000 kcal模型生成关联对比的SD、同人两轮对比相关及训练邻域局部范围支持。这些均为诊断，不是观察到的饮食响应、临床获益、个体分群或入选条件。不输出个人预测或个人功能值到截图；详细数组留TRE。

若主要路线未在两轮均有正增益，本轮按缺乏稳定预测支持报告并停止该组合的继续算法扩展。即使均为正，仍报告幅度、条件区间和敏感性是否一致；不得把较好的敏感性路线升级为确认性主要发现。需要独立人群/后续设计才能形成确认或机制证据。

## 候选解释与可区分结果

功能关联异质性、分类组成代理、测序/报告过程、空值处理、时间错位与随机选型均可产生表面差异。三个功能表示、共同技术背景及家庭隔离用于辨别部分替代解释，不保证排除所有混杂。只有模型生成对比稳定但无折外增益时，不称存在有效预测修饰。没有逐通路P值筛选或事后改结局、改阈值、改分折。

## 数据处理与交付

所有HPP数据/预测/检查点留在TRE；本地仅生成数据检查。新运行文件限定本子目录，旧模型目录保持原样。注册保存代码/输入/功能文件SHA、假设、排除状态、配置、环境与线程。每外折原子检查点和完成SHA供同源恢复，失败保留锁与原因。最终只需不超过三屏汇总，不下载参与者文件。

公开测量说明见父目录research_records/2026-11-10_pathway_null_contract/official_review.md；原始数据证据见00:53截图归档。研究团队负责核对论文主张与数据使用范围。
HPP_FUNCTION_SOURCE_001_END
hpp_function_write 'README.md' 'e9b1879e6af5c3999c33e4090b3b44c93b506f1b78ce5aaa0496ef34c48e3158' <<'HPP_FUNCTION_SOURCE_002_END'
# 口腔功能探索 v1

此模块是V12完成后的独立子目录，不修改父目录51份冻结交付文件或旧结果。方案见[PROTOCOL.md](PROTOCOL.md)。主问题仍为纤维密度与个人平均D_Glu的关联异质性；只做固定范围的探索性折外预测比较，不生成逐通路显著性排行榜。

原3343人来源已匹配；六个候选功能项全空者单独留标记，功能可用3337人为本次评价人群。各路线均重新拟合相同人群/折的主效应参照，不直接与旧3343人的R²比较。主要路线为相对丰度信号，覆盖度和未知空值编码仅作敏感性；二轮均是同人家庭分折，不能称独立重复。

部分空格按未报告0处理是有依据的工作假设，不是对HPP聚合格式的认证。相对信号表示不声称已知RPK/CPM，覆盖度不称反应完成率。六人不造全零功能谱；观察值内常量列不全局删除。训练内筛选/PCA和选型均不看外层验证Y。固定人群内只按训练稳健阈值截尾结局，不按新阈值再次剔除成员；原8SD规则的标记保留作审计。技术读段主效应及其饮食项进入两个比较模型。

安装指令会只创建本子目录，已有相同文件跳过，不同内容拒绝覆盖。默认运行命令为：

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/functional_followup_20261110_v1
bash server_run.sh --threads 2
```

启动依次核对新模块文件、生成数据检查、原51份源码、既有环境与来源资格，再进入两轮五折的家庭隔离探索。原生数据从不发送到本地；最终数组和资格标记保存在`runs/function_20261110_v1_iofix1/TRE_ONLY/`，只展示不超过三屏的聚合报告。耗时取决于TRE计算资源，不预先承诺完成时间。

2026-11-10存储修复：首次原生运行在模型拟合前保存NPZ时报seek不支持。修复版在内存中组装ZIP，随后顺序写入并核对SHA；统计方案、配置、特征变换、人口、分折和惩罚网格保持原样。默认新运行标识为`function_20261110_v1_iofix1`，旧失败目录`runs/function_20261110_v1/`及其锁/部分文件保留，不把不同代码的旧注册强行当作同源恢复。修复指令对已知旧源作备份，拒绝未知改动或活动进程。

十个外折有原子检查点。失败后保留锁和已完成折；仅当同一主机旧PID不存在、代码/配置/输入/环境完全相同时可恢复：

```bash
bash server_run.sh --threads 2 --resume
```

查看完成汇总不重新读取参与者数据：

```bash
bash server_run.sh --threads 2 --report
```

报告保留所有路线的两轮负值、无交互选择、匹配主效应R²、增量及条件家庭自助区间。区间不包含重训/选型不确定性，也不是正式交互P值。10→15对比为模型生成的关联差，不是实际饮食干预获益。主要路线缺乏两轮一致增益时按预定停止规则报告，不再改网格寻找好结果。

`check_synthetic.py`和`demo_*.py`只生成测试数据，不触碰原生HPP输入；验证输出存本模块`research_records/local_validation/`并醒目标为生成数据。真实服务器运行目前尚未执行，任何本地模拟分数都不是HPP结果。
HPP_FUNCTION_SOURCE_002_END
hpp_function_write 'audit_driver.py' '0c2186ecfc4b42456437f5145847addf807d4cff9979088e30f57bcccfbcf92c' <<'HPP_FUNCTION_SOURCE_003_END'
import hashlib, json, math, os, re, signal, sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def expired(signum, frame):
    print('FEASIBILITY_TIME_LIMIT; incomplete, no retry.', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGALRM, expired)
signal.alarm(60)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
P = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
expected = int(sys.argv[3]) if len(sys.argv) > 3 else 3343
input_count = int(sys.argv[4]) if len(sys.argv) > 4 else 4427
expected_households = int(sys.argv[5]) if len(sys.argv) > 5 else 3059
B = P/'collaborator_handoff_20261102_v1'
O = R/'oral_microbiome'
K = ['participant_id', 'cohort', 'research_stage']
META = K + ['array_index', 'collection_date']
F = ['humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']
BAD = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
print('FIXED_PATHWAY_FEASIBILITY: TRE-only linkage and descriptive QC; old outcome availability only for fixed membership; no outcome summaries, models or writes.', flush=True)

def require(condition, label):
    if not condition:
        raise ValueError(label)

def digest(path):
    require(path.stat().st_size <= 128*1024**2, 'CHECKSUM_FILE_BUDGET')
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'CHECKSUM_FILE_CHANGED')
    return h.hexdigest()

def json_file(path):
    with path.open('rb') as stream:
        value = stream.read(2*1024**2 + 1)
    require(len(value) <= 2*1024**2, 'JSON_BUDGET')
    result = json.loads(value)
    require(isinstance(result, dict), 'JSON_NOT_MAPPING')
    return result

def valid(value):
    if value is None or isinstance(value, str) and value.strip().lower() in BAD:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def key(row):
    values = tuple(row.get(n) for n in K)
    return values if all(valid(v) for v in values) else None

def day(value):
    if not valid(value):
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError, OverflowError):
        return None

def same(a, b):
    return valid(a) and valid(b) and a == b

def matches(a, b):
    stage_compatible = not valid(b.get('research_stage')) or same(a.get('research_stage'), b.get('research_stage'))
    return stage_compatible and same(a.get('cohort'), b.get('cohort')) and same(a.get('array_index'), b.get('array_index')) and day(a.get('collection_date')) is not None and day(a.get('collection_date')) == day(b.get('collection_date'))

def signature(path):
    state = path.stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns

def unique_pid(rows):
    counts = Counter(r.get('participant_id') for r in rows if valid(r.get('participant_id')))
    return {r['participant_id']:i for i,r in enumerate(rows) if valid(r.get('participant_id')) and counts[r['participant_id']] == 1}

def candidates(raw):
    u = urlsplit(raw)
    if u.scheme not in ('', 'file') or u.netloc not in ('', 'localhost') or u.query or u.fragment:
        return []
    p = Path(unquote(u.path))
    if '..' in p.parts:
        return []
    paths = [p] if p.is_absolute() else [O/p, R/p]
    for marker in ('oral_microbiome', 'aggregated'):
        if marker in p.parts:
            paths.append((R if marker == 'oral_microbiome' else O).joinpath(*p.parts[p.parts.index(marker):]))
    allowed, result = Path(os.path.realpath(R.parent)), []
    for p in paths:
        q = Path(os.path.realpath(p))
        if allowed in q.parents and not re.search(r'confirm|holdout|heldout|held_out|w22|w23', str(q), re.I) and q not in result and q.is_file():
            result.append(q)
    return result

def ipc_schema(stream):
    require(10 <= stream.size() <= 64*1024**2, 'IPC_FILE_BUDGET')
    require(stream.read(6) == b'ARROW1', 'NOT_IPC_FILE')
    stream.seek(-10, 2)
    require(int.from_bytes(stream.read(4), 'little') <= 32*1024**2, 'IPC_FOOTER_BUDGET')
    stream.seek(0)
    return ipc.open_file(stream).schema

def ipc_columns(path, columns, expected_signature=None):
    before = signature(path)
    require(expected_signature is None or before == expected_signature, 'FUNCTION_CHANGED_SINCE_METADATA')
    with pa.memory_map(str(path), 'r') as stream:
        schema = ipc_schema(stream)
        require(all(n in schema.names for n in columns) and bool(columns), 'IPC_COLUMNS_MISSING')
        stream.seek(0)
        options = ipc.IpcReadOptions(use_threads=False, included_fields=[schema.get_field_index(n) for n in columns])
        result = ipc.open_file(stream, options=options).read_all()
        require(result.nbytes <= 256*1024**2, 'DECODED_TABLE_BUDGET')
        require(before == signature(path), 'FUNCTION_CHANGED_DURING_READ')
        return result

ARROW_NULL_MASKS = {}

def qc(label, path, rows, columns, expected_signature):
    selected = ipc_columns(path, columns, expected_signature).take(pa.array(rows, type=pa.int64()))
    nulls = sum(selected.column(n).null_count for n in columns)
    ARROW_NULL_MASKS[label] = np.column_stack([selected.column(n).is_null().to_numpy(zero_copy_only=False) for n in columns])
    matrix = np.column_stack([selected.column(n).to_numpy(zero_copy_only=False) for n in columns]).astype(float)
    finite = np.isfinite(matrix)
    observations = finite.sum(axis=0)
    low = np.where(finite, matrix, np.inf).min(axis=0)
    high = np.where(finite, matrix, -np.inf).max(axis=0)
    values = matrix[finite]
    quantiles = np.quantile(values, [0, .5, .99, 1]).tolist() if len(values) else []
    positive = (finite & (matrix > 0)).sum(axis=1)
    print('  QC_CELLS', label, 'cells=%d arrow_null=%d nonnull_nan=%d inf=%d negative=%d gt1=%d min_med_p99_max=%s units_unverified=True' % (
        matrix.size, nulls, int(np.isnan(matrix).sum())-nulls, int(np.isinf(matrix).sum()),
        int((finite & (matrix < 0)).sum()), int((finite & (matrix > 1)).sum()), json.dumps(quantiles)), flush=True)
    print('  QC_PROFILES', label, 'fixed_people=%d no_finite=%d no_positive_observed=%d all_observed_zero=%d constant_observed_features=%d all_missing_features=%d positive_fields_q10_50_90=%s' % (
        len(rows), int((finite.sum(axis=1) == 0).sum()), int((positive == 0).sum()),
        int((finite.all(axis=1) & (matrix == 0).all(axis=1)).sum()), int(((observations > 0) & (low == high)).sum()),
        int((observations == 0).sum()), json.dumps(np.quantile(positive, [.1, .5, .9]).tolist())), flush=True)
    return matrix

"""Included verbatim in the self-contained terminal command; no native execution locally."""

def scale_metadata(label, schema):
    accepted = {'rpk', 'rpkm', 'cpm', 'tpm', 'counts', 'copies_per_million', 'relative_abundance', 'proportion', 'probability', 'percent', 'fraction', 'raw', 'none', 'log', 'log1p'}
    allowed_keys = {'unit', 'units', 'normalization', 'scale', 'transform'}
    recognized, unknown, entries = Counter(), 0, 0
    for metadata in [schema.metadata or {}] + [f.metadata or {} for f in schema]:
        for k, v in metadata.items():
            name = k.decode('utf8', errors='replace').lower() if isinstance(k, bytes) else str(k).lower()
            if name not in allowed_keys:
                continue
            entries += 1
            value = v.decode('utf8', errors='replace').strip().lower() if isinstance(v, bytes) else str(v).strip().lower()
            if value in accepted:
                recognized[name + ':' + value] += 1
            else:
                unknown += 1
    print('SCALE_METADATA', label, 'explicit_entries=%d recognized=%s unrecognized_entries=%d; provider_scale_not_inferred_from_range.' % (entries, json.dumps(dict(recognized), sort_keys=True), unknown), flush=True)

def null_contract(records, matrices, approved, source_path, expected_source_hash):
    require(records['abundance']['columns'] == records['coverage']['columns'], 'NULL_AUDIT_LABEL_SETS_DIFFER')
    a, b = matrices['abundance'], matrices['coverage']
    am, bm = ARROW_NULL_MASKS['abundance'], ARROW_NULL_MASKS['coverage']
    blank = am.all(axis=1) | bm.all(axis=1)
    paired_finite_profiles = np.isfinite(a).any(axis=1) & np.isfinite(b).any(axis=1)
    populated = ~blank & paired_finite_profiles
    selected_indices = np.flatnonzero(blank)
    print('NULL_STORAGE arrow_null_masks_equal=%s whole_candidate_null_in_either_table=%d with_paired_finite_candidates=%d other_profiles_without_paired_finite=%d coverage_explicit_zero_cells=%d coverage_zero_with_positive_abundance_cells=%d' % (
        np.array_equal(am, bm), int(blank.sum()), int(populated.sum()), int((~blank & ~paired_finite_profiles).sum()), int((np.isfinite(b) & (b == 0)).sum()), int((np.isfinite(a) & np.isfinite(b) & (a > 0) & (b == 0)).sum())), flush=True)
    for label, record in records.items():
        scale_metadata(label, record['schema'])
        special = record['special']
        if not special:
            print('SPECIAL_STATUS', label, 'special_fields_unavailable; no output-state conclusion.', flush=True)
            continue
        table = ipc_columns(record['path'], special, record['signature']).take(pa.array(record['linked'], type=pa.int64()))
        special_matrix = np.column_stack([table.column(n).to_numpy(zero_copy_only=False) for n in special]).astype(float)
        sf = np.isfinite(special_matrix)
        print('SPECIAL_STATUS', label, 'special_columns=%d fixed_rows_all_special_finite=%d populated_rows_all_special_finite=%d blank_rows_all_special_finite=%d blank_rows_all_special_zero=%d blank_rows_all_special_one=%d' % (
            len(special), int(sf.all(axis=1).sum()), int((sf.all(axis=1) & populated).sum()), int((sf.all(axis=1) & blank).sum()),
            int((sf.all(axis=1) & (special_matrix == 0).all(axis=1) & blank).sum()), int((sf.all(axis=1) & (special_matrix == 1).all(axis=1) & blank).sum())), flush=True)
        for name in special:
            value = table.column(name).to_numpy(zero_copy_only=False)[selected_indices]
            finite = np.isfinite(value)
            code = name.split(':', 1)[0].strip().upper()
            print('  BLANK_SPECIAL', label, code, 'group_n=%d finite=%d missing_or_nonfinite=%d positive=%d zero=%d one=%d negative=%d' % (
                len(value), int(finite.sum()), int((~finite).sum()), int((finite & (value > 0)).sum()), int((finite & (value == 0)).sum()), int((finite & (value == 1)).sum()), int((finite & (value < 0)).sum())), flush=True)

    require(digest(source_path) == expected_source_hash, 'SOURCE_CHANGED_BEFORE_NULL_AUDIT')
    technical = ['raw_read_count', 'trimmed_read_count', 'duplicate_reads_count', 'excessive_duplicate_reads', 'metaphlan4_aligned_read_count', 'humann_aligned_read_count', 'humann_unmapped_read_count']
    pointers = ['humann_pathway_abundance_tsv', 'humann_pathway_coverage_tsv', 'humann_pathway_abundance_pathway_level_parquet', 'humann_pathway_coverage_pathway_level_parquet']
    with pq.ParquetFile(source_path, pre_buffer=False) as pf:
        present = [n for n in technical+pointers if n in pf.schema_arrow.names]
        print('BLANK_SOURCE_QC group_n=%d available_technical_fields=%s' % (len(selected_indices), json.dumps([n for n in technical if n in present])), flush=True)
        if present:
            positions = [approved[i] for i in selected_indices]
            rows = pf.read(columns=present, use_threads=False).take(pa.array(positions, type=pa.int64())).to_pydict()
            for name in technical:
                if name not in rows:
                    continue
                values = rows[name]
                if name == 'excessive_duplicate_reads':
                    print('  BLANK_READ_QC', name, 'observed=%d true_flags=%d; absence_not_pass.' % (sum(valid(v) for v in values), sum(v is True for v in values)), flush=True)
                else:
                    numeric = []
                    for v in values:
                        try:
                            if valid(v) and math.isfinite(float(v)):
                                numeric.append(float(v))
                        except (TypeError, ValueError, OverflowError):
                            pass
                    print('  BLANK_READ_QC', name, 'finite_numeric=%d positive=%d zero=%d negative=%d; no thresholds_selected.' % (len(numeric), sum(v > 0 for v in numeric), sum(v == 0 for v in numeric), sum(v < 0 for v in numeric)), flush=True)
            for name in pointers:
                if name in rows:
                    print('  BLANK_POINTER', name, 'nonempty=%d/%d; no file access inferred or attempted.' % (sum(isinstance(v, str) and valid(v) for v in rows[name]), len(selected_indices)), flush=True)
    print('NULL_CONTRACT_LIMIT: software zero-omission rules plus masks are supporting evidence, not HPP aggregation certification; whole-null profiles remain flagged. No filling, filtering, rescaling or fitting.', flush=True)


try:
    cov_path = B/'inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet'
    v4 = B/'experiments/diet_nested_stability_20261103_v4/results'
    receipt = json_file(v4/'registration.json')['source_and_input_sha256']
    require(receipt.get(str(cov_path)) == digest(cov_path), 'DISCOVERY_INPUT_CHECKSUM_MISMATCH')
    with pq.ParquetFile(cov_path, pre_buffer=False) as pf:
        x = pf.read(columns=K+['input_row', 'final_role', 'split_unit_id'], use_threads=False).to_pandas(ignore_metadata=True)
    require(len(x) == input_count and x.final_role.eq('discovery').all() and x.participant_id.is_unique, 'DISCOVERY_INDEX_CHANGED')
    require(np.array_equal(x.input_row.to_numpy(), np.arange(len(x))), 'DISCOVERY_ROW_ORDER_CHANGED')
    with np.load(str(v4/'TRE_ONLY/final_predictions.npz'), allow_pickle=False) as archive:
        eligible = np.isfinite(archive['y'])
    require(eligible.shape == (2, input_count) and np.array_equal(eligible[0], eligible[1]), 'FIXED_MEMBERSHIP_CHANGED')
    target = x.loc[eligible[0]].copy()
    require(len(target) == expected and target.split_unit_id.notna().all() and target.split_unit_id.nunique() == expected_households, 'FIXED_COUNTS_CHANGED')
    people = target[K].to_dict('records')
    require(all(key(r) is not None for r in people), 'FIXED_KEYS_INVALID')
    print('FROZEN_MEMBERSHIP discovery_rows=%d fixed_people=%d households=%d same_two_repeat_mask=True discovery_input_sha_verified=True' % (len(x), len(target), target.split_unit_id.nunique()), flush=True)

    reg_path = P/'analysis/shared/sample_registry_TRE_ONLY.parquet'
    lock = json_file(P/'config/w13_sample_selection_lock.json')
    require(lock.get('sample_registry_sha256') == digest(reg_path), 'LOCKED_REGISTRY_CHECKSUM_MISMATCH')
    reg_fields = K+['site', 'sample_lock_status', 'linked_index_row_ordinal', 'primary_index_source_path', 'primary_index_source_sha256']
    with pq.ParquetFile(reg_path, pre_buffer=False) as pf:
        registry = pf.read(columns=reg_fields, use_threads=False).to_pylist()
    reg_map = defaultdict(list)
    for r in registry:
        if r['site'] == 'oral' and key(r) is not None:
            reg_map[key(r)].append(r)
    src_path = O/'oral_microbiome.parquet'
    src_hash = digest(src_path)
    with pq.ParquetFile(src_path, pre_buffer=False) as pf:
        src_table = pf.read(columns=META+F, use_threads=False)
    source = src_table.select(META).to_pylist()
    src_map = unique_pid(source)
    reasons, approved = Counter(), []
    for person in people:
        candidates_reg = reg_map.get(key(person), [])
        if len(candidates_reg) != 1:
            reasons['registry_missing_or_ambiguous'] += 1
            approved.append(None)
            continue
        r = candidates_reg[0]
        reason = None
        if r['sample_lock_status'] != 'locked_W13_independent_full_source_verification_passed':
            reason = 'registry_not_locked'
        elif not isinstance(r['primary_index_source_path'], str) or os.path.realpath(r['primary_index_source_path']) != os.path.realpath(src_path):
            reason = 'source_path_mismatch'
        elif r['primary_index_source_sha256'] != src_hash:
            reason = 'source_checksum_mismatch'
        else:
            ordinal = r['linked_index_row_ordinal']
            if not valid(ordinal) or not isinstance(ordinal, (int, float)) or not math.isfinite(ordinal) or int(ordinal) != ordinal or not 1 <= ordinal <= len(source):
                reason = 'source_ordinal_invalid'
            else:
                index = int(ordinal)-1
                if key(source[index]) != key(person) or src_map.get(person['participant_id']) != index:
                    reason = 'source_key_or_uniqueness_mismatch'
        reasons[reason or 'source_record_verified'] += 1
        approved.append(index if reason is None else None)
    print('LOCKED_SOURCE_RECORDS registry_sha_verified=True reasons=' + json.dumps(dict(reasons), sort_keys=True), flush=True)

    records = {}
    for label, field in zip(('abundance', 'coverage'), F):
        print('TABLE', label, flush=True)
        try:
            values = list(dict.fromkeys(v.strip() for v in src_table[field].to_pylist() if isinstance(v, str) and valid(v)))
            paths = candidates(values[0]) if len(values) == 1 else []
            require(len(paths) == 1, 'FUNCTION_POINTER_OR_PATH_AMBIGUOUS')
            path = paths[0]
            stamp = signature(path)
            with pa.memory_map(str(path), 'r') as stream:
                schema = ipc_schema(stream)
            numeric = [f.name for f in schema if f.name not in META and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))]
            special = [n for n in numeric if n.split(':', 1)[0].strip().upper() in ('UNMAPPED', 'UNINTEGRATED')]
            columns = sorted(n for n in numeric if n not in special)
            require(bool(columns), 'NO_NUMERIC_CANDIDATE_FEATURES')
            metadata = ipc_columns(path, META, stamp).to_pylist()
            mapping = unique_pid(metadata)
            shared = [(r, metadata[mapping[r['participant_id']]]) for r in source if r['participant_id'] in src_map and r['participant_id'] in mapping]
            print('  SOURCE_METADATA shared_unique_people=%d cohort_equal=%d array_equal=%d calendar_date_equal=%d timestamp_equal=%d source_record_metadata_compatible=%d' % (
                len(shared), sum(same(a['cohort'], b['cohort']) for a,b in shared), sum(same(a['array_index'], b['array_index']) for a,b in shared),
                sum(day(a['collection_date']) is not None and day(a['collection_date']) == day(b['collection_date']) for a,b in shared),
                sum(same(a['collection_date'], b['collection_date']) for a,b in shared), sum(matches(a,b) for a,b in shared)), flush=True)
            linked, fixed_reasons = [], Counter()
            for person, source_index in zip(people, approved):
                if source_index is None:
                    fixed_reasons['locked_source_unresolved'] += 1
                    continue
                position = mapping.get(person['participant_id'])
                if position is None:
                    fixed_reasons['function_person_missing_or_nonunique'] += 1
                    continue
                if not matches(source[source_index], metadata[position]):
                    fixed_reasons['function_metadata_conflict_or_date_missing'] += 1
                    continue
                fixed_reasons['source_record_metadata_agree'] += 1
                linked.append(position)
            print('  FIXED_LINK target=%d linked=%d reasons=%s' % (expected, len(linked), json.dumps(dict(fixed_reasons), sort_keys=True)), flush=True)
            print('  FEATURE_SCHEMA numeric_total=%d special_totals_excluded=%d candidate_columns=%d annotation_and_units_pending=True' % (len(numeric), len(special), len(columns)), flush=True)
            records[label] = dict(path=path, signature=stamp, schema=schema, special=special, metadata=metadata, mapping=mapping, linked=linked, columns=columns, complete=len(linked)==expected)
        except (OSError, ValueError, TypeError, KeyError) as e:
            detail = str(e) if isinstance(e, ValueError) and re.fullmatch(r'[A-Z_]{1,80}', str(e)) else type(e).__name__
            print('  TABLE_UNRESOLVED', detail, flush=True)

    complete = len(records) == 2 and all(v['complete'] for v in records.values())
    if len(records) == 2:
        a, b = records['abundance'], records['coverage']
        print('PAIR_FEATURE_LABEL_SETS_EQUAL=%s' % (a['columns'] == b['columns']), flush=True)
    if complete:
        print('DESCRIPTIVE_QC: all fixed people passed source-record metadata checks; missing cells are retained.', flush=True)
        matrices = {label:qc(label, r['path'], r['linked'], r['columns'], r['signature']) for label,r in records.items()}
        if records['abundance']['columns'] == records['coverage']['columns']:
            a, b = matrices['abundance'], matrices['coverage']
            common = np.isfinite(a) & np.isfinite(b)
            print('PAIR_NUMERIC fixed_people=%d common_finite_cells=%d exact_equal_common_finite_cells=%d' % (expected, int(common.sum()), int((a[common] == b[common]).sum())), flush=True)
    else:
        print('QC_SKIPPED: complete fixed-population source-record linkage not established; no subset replacement.', flush=True)
    if complete:
        null_contract(records, matrices, approved, src_path, src_hash)
    print('Source-record agreement is not independent native-file specimen validation, verified units, causal timing or a new scientific effect.', flush=True)
    print('PATHWAY_NULL_CONTRACT_DONE', flush=True)
except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
    detail = str(e) if isinstance(e, ValueError) and re.fullmatch(r'[A-Z_]{1,80}', str(e)) else type(e).__name__
    print('FEASIBILITY_STOP', detail, '; no model or subset replacement.', flush=True)
signal.alarm(0)
HPP_FUNCTION_SOURCE_003_END
hpp_function_write 'check_synthetic.py' 'a41f7f96ab80dc55204ed14511a456e1725d14b65cca65dad70804d258f31f8b' <<'HPP_FUNCTION_SOURCE_004_END'
"""Deterministic checks with generated inputs only; no native HPP reader calls."""
import copy
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import native
from demo_data import make
from frozen import modules
from functional import Design, FixedMemberTarget, FunctionState, TechnicalState
from pilot import checkpoint, execute, outer, paired_intervals, report, save_arrays, sha, summarize

ROOT=Path(__file__).resolve().parent


def config():
    c=json.loads((ROOT/'config.json').read_text())
    c.update(outer_folds=3,inner_folds=2,prediction_bootstraps=99,functional_main_components=5,
             main_penalties=[.01,.1],interaction_penalties=[.03,.3],support_neighbors=40)
    return c


def expect_error(action, fragment):
    try:
        action()
    except (ValueError,RuntimeError) as error:
        assert fragment in str(error),(fragment,str(error))
    else:
        raise AssertionError('Expected rejection: '+fragment)


def check_sequential_archive(passed):
    original_open=io.open
    writers=[]
    class ForwardOnly:
        def __init__(self,stream,fail_write=False):
            self.stream,self.fail_write=stream,fail_write
            self.seeks=[]
            self.backward_rejected=False
        def __getattr__(self,name):
            return getattr(self.stream,name)
        def __enter__(self):
            return self
        def __exit__(self,*args):
            self.stream.close()
        def seek(self,offset,whence=0):
            self.seeks.append((offset,whence))
            target=offset if whence==0 else self.stream.tell()+offset if whence==1 else os.fstat(self.stream.fileno()).st_size+offset
            if target < self.stream.tell() and not self.backward_rejected:
                # Permit later destructor cleanup after reproducing the first
                # unsupported backwards seek, avoiding ignored-finalizer noise.
                self.backward_rejected=True
                raise OSError(95,'Simulated unsupported backwards write seek')
            return self.stream.seek(offset,whence)
        def write(self,payload):
            if self.fail_write:
                self.stream.write(payload[:5])
                raise OSError(5,'Simulated interrupted sequential write')
            return self.stream.write(payload)
    def hook(fail_write=False):
        def opened(path,mode='r',*args,**kwargs):
            result=original_open(path,mode,*args,**kwargs)
            if isinstance(path,(str,os.PathLike)) and str(path).endswith('.npz') and any(c in mode for c in 'wxa'):
                result=ForwardOnly(result,fail_write)
                writers.append(result)
            return result
        return opened
    with tempfile.TemporaryDirectory(prefix='hpp_npz_forward_only_synthetic_') as tmp:
        folder=Path(tmp)
        arrays=dict(flags=np.array([True,False,True]),values=np.arange(18,dtype=float).reshape(6,3))
        reproduced=False
        with patch('io.open',hook()):
            try:
                np.savez_compressed(str(folder/'old_direct.npz'),**arrays)
            except OSError as error:
                reproduced=error.errno==95
        passed('original_path_zip_writer_reproduces_errno95',reproduced and any(w.backward_rejected for w in writers))
        writers.clear()
        target=folder/'new_sequential.npz'
        with patch('io.open',hook()):
            save_arrays(target,arrays)
        passed('new_archive_writer_never_seeks_target_file',bool(writers) and all(not w.seeks for w in writers))
        with np.load(str(target),allow_pickle=False) as loaded:
            passed('sequential_npz_roundtrip_preserves_flags_values_and_shapes',all(np.array_equal(loaded[name],value) and loaded[name].dtype==value.dtype for name,value in arrays.items()))
        before=target.read_bytes()
        failed=False
        with patch('io.open',hook(True)):
            try:
                save_arrays(target,dict(flags=np.array([False])))
            except OSError as error:
                failed=error.errno==5
        passed('interrupted_write_preserves_completed_file_and_cleans_temporary',failed and target.read_bytes()==before and not list(folder.glob('new_sequential.tmp-*')))


def native_adapter_fixture():
    """Run the copied source audit on generated files, mock only the old model reader."""
    import demo_files as fixture
    with tempfile.TemporaryDirectory(prefix='hpp_function_adapter_synthetic_') as tmp:
        state=fixture.fixture(Path(tmp))
        for which,path in enumerate(state['paths']):
            with pa.memory_map(str(path),'r') as stream:
                table=ipc.open_file(stream).read_all()
            rows=table.to_pylist()
            for row in rows:
                if row['participant_id']==101:
                    for name in ('PWY-1234:demo','GLYCOLYSIS-PWY:demo','VALID-OTHER-FEATURE'):
                        row[name]=None
                if row['participant_id']==102:
                    row['GLYCOLYSIS-PWY:demo']=None
            result=pa.Table.from_pylist(rows,schema=table.schema)
            with pa.OSFile(str(path),'wb') as out:
                with ipc.new_file(out,result.schema) as writer:
                    writer.write_table(result)
        index=pq.ParquetFile(state['cov']).read().to_pandas(ignore_metadata=True)
        with np.load(str(state['reference']),allow_pickle=False) as archive:
            reference=dict(y=archive['y'])
        dataset=SimpleNamespace(index=index,reference=reference,paths=[state['cov']],y=np.arange(5,dtype=float),api=SimpleNamespace(PhenotypeOutliers=lambda:None))
        frozen=modules()
        gate=Path(tmp)/'existing_summary.json'
        gate.write_text('{"synthetic_only":true}')
        fake=SimpleNamespace(root=frozen.root,config=frozen.config,upload_check=frozen.upload_check,
            runtime=SimpleNamespace(environment=lambda c:dict(synthetic_only=True),readiness=lambda *a,**k:dict(v11_summary=str(gate))),
            bridge=SimpleNamespace(load=lambda *a:dataset,adapter_sources=lambda:[]))
        before=native.modules
        old_handler=signal.getsignal(signal.SIGALRM)
        native.modules=lambda:fake
        try:
            cfg=config()
            cfg.update(original_people=3,functional_people=2,candidate_fields=3,source_input_rows=5,original_households=2)
            data=native.load(state['project'],state['raw'],cfg)
        finally:
            native.modules=before
        assert signal.getsignal(signal.SIGALRM)==old_handler
        assert np.flatnonzero(data.available).tolist()==[1,2]
        assert np.isnan(data.signals['abundance'][0]).all()
        assert data.signals['abundance'][1,1] == 2.  # sorted public labels: glycolysis, PWY, other
        assert data.audit['excluded_whole_candidate_null']==1
        assert 'PATHWAY_NULL_CONTRACT_DONE' in data.audit_text
        assert not data.synthetic_only  # native adapter path was exercised, all files were generated


def main():
    checks=[]
    def passed(name, condition=True):
        assert condition,name
        checks.append(name)
    cfg=config()
    check_sequential_archive(passed)
    class OriginalTarget:
        def fit(self,y,participant_ids,*,endpoint,role='training'):
            self.state=dict(kind='synthetic_original',clip_low=-5.,clip_high=5.)
            return self
        def transform(self,y,*,endpoint):
            a=np.asarray(y,float)
            return dict(values=np.where(abs(a)>8,np.nan,np.clip(a,-5,5)),excluded_8sd=abs(a)>8)
    target=FixedMemberTarget(OriginalTarget()).fit(np.array([-1.,0.,1.]),[0,1,2],endpoint='D_Glu')
    transformed=target.transform(np.array([100.,np.nan,-100.,1.]),endpoint='D_Glu')
    passed('fixed_eligible_extreme_outcomes_clipped_not_rescreened',np.array_equal(transformed['values'][[0,2,3]],np.array([5.,-5.,1.])) and not transformed['excluded_8sd'].any())
    passed('nonfinite_outcomes_not_manufactured',np.isnan(transformed['values'][1]) and transformed['original_rule_excluded_8sd'][0] and transformed['winsorized_high'][0] and transformed['winsorized_low'][2])
    data=make(cfg)
    frozen=modules()
    a=data.signals['abundance'][:300]
    state=FunctionState('abundance_relative',cfg).fit(a)
    multiplier=np.linspace(.01,100,len(a))[:,None]
    passed('relative_signal_invariant_to_positive_sample_scale',np.allclose(state.transform(a),state.transform(a*multiplier),atol=1e-10))
    ranks=FunctionState('abundance_unknown_null',cfg).fit(a)
    b=a.copy()
    b[:,0]=np.nan
    passed('unknown_null_encoding_retains_reporting_information',not np.allclose(ranks.raw(a),ranks.raw(b)))
    rank_scaled=FunctionState('abundance_unknown_null',cfg).fit(a*np.arange(1,a.shape[1]+1))
    passed('observed_rank_representation_invariant_to_feature_units',np.allclose(ranks.raw(a),rank_scaled.raw(a*np.arange(1,a.shape[1]+1))))
    empty=np.ones((100,5))
    constant=FunctionState('abundance_relative',cfg).fit(empty)
    passed('constant_relative_spectrum_has_zero_components',constant.transform(empty).shape==(100,0))
    never=np.full((100,4),np.nan)
    never[:,0]=1.
    rare=FunctionState('coverage_confidence',cfg).fit(never)
    passed('training_missing_fields_not_selected',rare.keep.tolist()==[0])
    expect_error(lambda:state.transform(np.full((2,a.shape[1]),np.nan)),'flagged')
    passed('entire_null_profile_not_made_biological_zero')
    expect_error(lambda:FunctionState('coverage_confidence',cfg).fit(np.full((50,4),2.)),'range')
    passed('coverage_range_checked')
    tech=TechnicalState().fit(data.technical[:300])
    old_median=tech.median.copy()
    transformed=tech.transform(np.full((5,3),np.nan))
    passed('technical_missing_values_train_fitted_and_finite',np.array_equal(tech.median,old_median) and np.isfinite(transformed).all())
    train=data.available & (data.dataset.reference['outer'][0] != 0)
    valid=data.available & ~train
    base=data.frozen.engine.Engine(None,None).design(data.dataset,train,valid)
    view=Design(base,'abundance_relative',data.signals['abundance'],data.technical,cfg,frozen.features)
    spec=dict(name='abundance_relative')
    zt,zv,imap=view.interactions(spec)
    passed('functional_interaction_residualized_against_matched_main',np.max(abs(view.xt.T@zt))<1e-6)
    baseline=frozen.models.fit_main(view,.1)
    no_interaction=frozen.models.fit_augmented(view,spec,.1,None,baseline)
    passed('no_interaction_is_exact_same_fitted_baseline',no_interaction is baseline and np.array_equal(no_interaction.predict(),baseline.predict()))
    fit=frozen.models.fit_augmented(view,spec,.1,.03)
    total,component=fit.contrasts()
    passed('counterfactual_model_contrast_matches_direct_prediction_difference',np.allclose(total,fit.predict(diet=15)-fit.predict(diet=10)))
    passed('main_effect_contains_all_interaction_pc_coordinates',view.pcs[0].shape[1]>=zt.shape[1] and np.array_equal(view.xt[:,0],np.ones(len(view.xt))))
    passed('functional_nonlinear_main_terms_in_both_models',view.function_curves is not None and len(view.function_curves.keep)>0 and baseline.design is fit.design)
    unchanged=state.pca.components_.copy()
    state.transform(np.where(np.isnan(a[:5]),np.nan,a[:5]*100000))
    passed('heldout_function_values_do_not_refit_pca',np.array_equal(state.pca.components_,unchanged))

    first,meta=outer(data,cfg,0,0,lambda *a:None)
    altered=make(cfg)
    outer_valid=altered.available & (altered.dataset.reference['outer'][0]==0)
    altered.dataset.y[outer_valid] += 1000000.
    second,meta2=outer(altered,cfg,0,0,lambda *a:None)
    passed('outer_validation_outcome_does_not_change_selection',meta['selections']==meta2['selections'])
    passed('outer_validation_outcome_does_not_change_predictions',np.array_equal(first['main'],second['main']) and np.array_equal(first['interaction'],second['interaction']))
    passed('outer_outcome_mutation_reaches_only_evaluation_target',not np.array_equal(first['y'],second['y']))
    no_variation=make(cfg)
    for values in no_variation.signals.values():
        values[no_variation.available]=1.
    same_prediction,constant_meta=outer(no_variation,cfg,0,0,lambda *a:None)
    passed('zero_dimensional_functions_exactly_revert_to_main_in_nested_selection',np.array_equal(same_prediction['main'],same_prediction['interaction']) and all(v['interaction_penalty'] is None for v in constant_meta['selections'].values()))
    native_adapter_fixture()
    passed('native_adapter_full_generated_file_lineage_and_row_mapping')
    with tempfile.TemporaryDirectory(prefix='hpp_function_pilot_synthetic_') as tmp:
        folder=Path(tmp)
        failed_run=folder/'failed_before_models'
        failure_logs=[]
        startup_failure=False
        with patch('pilot.save_arrays',side_effect=OSError(95,'Simulated old startup failure')):
            try:
                execute(data,cfg,failed_run,threads=1,log=failure_logs.append)
            except OSError as error:
                startup_failure=error.errno==95
        preserved={name:(failed_run/name).read_bytes() for name in ('registration.json','RUNNING.lock','source_audit.txt')}
        passed('cohort_write_failure_occurs_before_any_model_or_complete_fold',startup_failure and not failure_logs and not list((failed_run/'TRE_ONLY').glob('*_COMPLETE.json')) and not (failed_run/'summary.json').exists())
        summary=execute(data,cfg,folder/'run',threads=1,log=lambda *a:None)
        passed('separate_repaired_run_preserves_failed_registration_and_lock',all((failed_run/name).read_bytes()==value for name,value in preserved.items()) and summary['status']=='COMPLETE')
        primary=[r for r in summary['scores'] if r['route']==cfg['primary_route']]
        passed('known_synthetic_interaction_gains_on_both_household_splits',all(r['delta_R2']>0 for r in primary))
        passed('all_routes_and_splits_reported',len(summary['scores'])==6 and all(r['N']==420 for r in summary['scores']))
        passed('synthetic_results_explicitly_labeled',summary['synthetic_only'] is True)
        passed('six_fold_atomic_completion_receipts',len(list((folder/'run/TRE_ONLY').glob('*_COMPLETE.json')))==6)
        again=execute(data,cfg,folder/'run',threads=1,log=lambda *a:None)
        passed('completed_run_replayed_without_refit',again==summary)
        tampered=folder/'run/TRE_ONLY/final_predictions.npz'
        with tampered.open('ab') as stream:
            stream.write(b'changed')
        expect_error(lambda:execute(data,cfg,folder/'run',threads=1,log=lambda *a:None),'checksum')
        passed('tampered_completed_arrays_rejected')
        ck=folder/'chunks'
        ck.mkdir()
        arrays=dict(rows=np.array([1]),y=np.array([2.]))
        checkpoint(ck,'one','same',lambda:(arrays,dict(synthetic_only=True)))
        expect_error(lambda:checkpoint(ck,'one','changed',lambda:None),'identity')
        passed('changed_checkpoint_identity_rejected')
        (ck/'one.json').write_text('{}')
        expect_error(lambda:checkpoint(ck,'one','same',lambda:None),'checksum')
        passed('tampered_fold_metadata_rejected')
        preview=[]
        from contextlib import redirect_stdout
        import io
        output=io.StringIO()
        with redirect_stdout(output):
            report(summary)
        preview=output.getvalue()
        passed('report_contains_scope_and_no_individual_identifiers','SYNTHETIC ONLY' in preview and 'SYNTHETIC_HH_' not in preview and 'formal interaction P value' in preview)
    # Explicitly adverse predictions: every planned negative result must survive aggregation.
    chunks=[]
    y=data.dataset.y
    for repeat in range(2):
        for fold in range(cfg['outer_folds']):
            rows=np.flatnonzero(data.available & (data.dataset.reference['outer'][repeat]==fold))
            main=np.tile(y[rows],(3,1))
            arrays=dict(rows=rows,y=y[rows],raw_y=y[rows],main=main,interaction=main+1.,total_contrast=np.zeros_like(main),delta_contrast=np.zeros_like(main),raw_component=np.zeros_like(main),support=np.ones_like(main,dtype=bool))
            chunks.append((arrays,dict(repeat=repeat,fold=fold,selections={route:dict(interaction_penalty=.03) for route in cfg['routes']})))
    negative,_=summarize(data,cfg,chunks)
    passed('negative_incremental_R2_preserved_for_every_route',all(row['delta_R2']<0 for row in negative['scores']))
    passed('constant_split_contrast_correlation_is_missing_not_zero',all(v is None for v in negative['same_people_added_contrast_split_correlation'].values()))
    error_chunks=chunks+[chunks[0]]
    expect_error(lambda:summarize(data,cfg,error_chunks),'duplicated')
    passed('duplicate_outer_members_rejected')
    groups=data.dataset.index.loc[data.available,'split_unit_id'].astype(str).to_numpy()
    noise=np.tile(y[data.available],(3,1))
    interval=paired_intervals(y[data.available],noise,noise+1.,groups,99,7)
    passed('paired_household_interval_has_correct_adverse_direction',np.all(interval<0))
    for p in ROOT.glob('*.py'):
        compile(p.read_text(),str(p),'exec')
    passed('all_python_sources_compile')
    destination=ROOT/'research_records/local_validation'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'synthetic_preview.txt').write_text('GENERATED DATA ONLY; NOT HPP FINDINGS\n\n'+preview)
    result=dict(status='passed',synthetic_only=True,check_count=len(checks),checks=checks,native_HPP_inputs_read=False,
        native_environment_execution_verified=False,scope='Generated matrices/files; frozen source imports; native reader exercised on generated fixtures only.')
    (destination/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FUNCTIONAL_SYNTHETIC_CHECKS_OK checks=%d; no native HPP data read.' % len(checks))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
HPP_FUNCTION_SOURCE_004_END
hpp_function_write 'config.json' '5f4fe8a03443b1623a9c2a980e3e8aadf3188148ab65290982ddf3cd898e3102' <<'HPP_FUNCTION_SOURCE_005_END'
{
  "version": "functional_followup_20261110_v1",
  "exposure": "fiber_density",
  "outcome": "D_Glu",
  "unit": "mg/dL",
  "endpoint_preprocessing": "fixed_eligibility_training_robust_5sd_clip_without_new_8sd_exclusions",
  "original_people": 3343,
  "source_input_rows": 4427,
  "original_households": 3059,
  "functional_people": 3337,
  "candidate_fields": 485,
  "primary_route": "abundance_relative",
  "routes": ["abundance_relative", "coverage_confidence", "abundance_unknown_null"],
  "primary_repeat": 0,
  "outer_repeats": 2,
  "outer_folds": 5,
  "inner_folds": 3,
  "seed": 20261110,
  "outer_seed": 20261104,
  "contrast": [10, 15],
  "minimum_training_signal_prevalence": 0.1,
  "functional_main_components": 10,
  "functional_modifier_components": 3,
  "functional_curved_components": 3,
  "main_penalties": [0.01, 0.1, 1.0],
  "interaction_penalties": [0.03, 0.3, 3.0],
  "prediction_bootstraps": 1999,
  "support_neighbors": 100,
  "support_quantiles": [0.1, 0.9],
  "partial_null_nonreporting_is_assumption": true,
  "entire_null_profiles_are_biological_zero": false,
  "provider_abundance_unit_verified": false,
  "claim_type": "exploratory_associational_prediction",
  "historical_discovery_search": true,
  "independent_confirmation": false,
  "confirmation_outcomes_read": false
}
HPP_FUNCTION_SOURCE_005_END
hpp_function_write 'demo_data.py' '4a25c6af3289ed250f42307ffa1ccebbe9ddc72abee81f293b0ebedf94a2efab' <<'HPP_FUNCTION_SOURCE_006_END'
"""Generated data only; no native HPP reads. Shared by deterministic checks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from frozen import modules


def folds(groups,k,seed):
    values=np.array(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(values)
    mapping={v:j%k for j,v in enumerate(values)}
    return np.array([mapping[v] for v in groups])


def separated(groups,train,valid):
    if np.any(train & valid) or set(groups[train]) & set(groups[valid]):
        raise ValueError('Synthetic household leak')


class Base:
    def __init__(self,dataset,train,valid):
        self.dataset=dataset
        self.config=dict(contrast=[10,15])
        self.train_rows,self.valid_rows=np.flatnonzero(train),np.flatnonzero(valid)
        self.center=np.median(dataset.index.fiber_density.to_numpy()[train])
        self.scaler=StandardScaler().fit(self.raw(True))
        s=SimpleNamespace(train_rows=self.train_rows,rows=self.valid_rows,yt=dataset.y[train],yv=dataset.y[valid])
        self.role=SimpleNamespace(split=s)
        age=dataset.index.age.to_numpy()
        self.background_t=age[train,None]
        self.background_v=age[valid,None]
        self.pcs={site:(dataset.taxonomy[site][train],dataset.taxonomy[site][valid]) for site in ('oral','gut')}

    def diet(self,training):
        return self.dataset.index.fiber_density.to_numpy()[self.train_rows if training else self.valid_rows]

    def diet_basis(self,training,value=None):
        d=self.diet(training) if value is None else np.full(len(self.diet(training)),value)
        u=(d-self.center)/5.
        return np.column_stack([u,u*u])

    def raw(self,training,value=None):
        rows=self.train_rows if training else self.valid_rows
        age=self.dataset.index.age.to_numpy()[rows]
        diet=self.diet_basis(training,value)
        return np.column_stack([diet,age,age*diet[:,0],self.dataset.index.sex.to_numpy()[rows]])

    def main(self,training=False,value=None):
        raw=self.raw(training,value)
        return np.column_stack([np.ones(len(raw)),self.scaler.transform(raw)])


class Engine:
    def __init__(self,*args):
        pass

    def design(self,dataset,train,valid):
        separated(dataset.index.split_unit_id.astype(str).to_numpy(),train,valid)
        return Base(dataset,train,valid)


def make(config,seed=112,interaction=10.,people=420):
    rng=np.random.default_rng(seed)
    n=people+2
    groups=np.array(['SYNTHETIC_HH_%04d' % (j//2) for j in range(n)])
    diet=rng.uniform(7,24,n)
    latent=rng.normal(size=n)
    age=rng.normal(size=n)
    fields=16
    loadings=np.linspace(-1.3,1.3,fields)
    abundance=np.exp(latent[:,None]*loadings[None,:]+rng.normal(scale=.25,size=(n,fields)))
    present=rng.random((n,fields))<.82
    abundance[~present]=np.nan
    abundance[-2:]=np.nan
    coverage=abundance/(1+abundance)
    y=4*(diet-14)/5.+age+.5*latent+interaction*(diet-14)/5.*latent+rng.normal(scale=.5,size=n)
    index=pd.DataFrame(dict(input_row=np.arange(n),participant_id=np.arange(n)+1,cohort='synthetic',research_stage='synthetic',split_unit_id=groups,fiber_density=diet,age=age,sex=rng.integers(0,2,n)))
    outer=np.stack([folds(groups,config['outer_folds'],config['outer_seed']+j) for j in (0,1)])
    dataset=SimpleNamespace(index=index,y=y,reference=dict(outer=outer,y=np.tile(y,(2,1))),api=SimpleNamespace(folds=folds,separated=separated),taxonomy={site:rng.normal(size=(n,4)) for site in ('oral','gut')})
    counts=np.exp(rng.normal(12,1,n))
    ratio=rng.uniform(.05,.7,n)
    technical=np.column_stack([counts,counts*ratio,ratio])
    available=np.arange(n)<people
    frozen=modules()
    return SimpleNamespace(dataset=dataset,frozen=SimpleNamespace(config=dict(contrast=[10,15]),engine=SimpleNamespace(Engine=Engine),models=frozen.models,features=frozen.features),available=available,
        signals=dict(abundance=abundance,coverage=coverage),technical=technical,labels=['PWY-%d:synthetic' % j for j in range(fields)],
        source_hashes={},versions=dict(synthetic_only=True),audit_text='SYNTHETIC ONLY; no native HPP data.',synthetic_only=True,
        audit=dict(original_people=n,functional_people=people,excluded_whole_candidate_null=2,functional_households=len(set(groups[available])),synthetic_only=True))
HPP_FUNCTION_SOURCE_006_END
hpp_function_write 'demo_files.py' 'ec6f1e5a4f8a972b1473c425827e7a4424a15e4cfaa5d4f11cfa4d9a9c57b965' <<'HPP_FUNCTION_SOURCE_007_END'
"""Standalone generated-file fixtures; never opens native participant data."""
import hashlib, json, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
K = ['participant_id', 'cohort', 'research_stage']
F = ['humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_arrow(path, rows, feature_order=None):
    table = pa.Table.from_pylist(rows)
    if feature_order is not None:
        table = table.select(feature_order)
    with pa.OSFile(str(path), 'wb') as stream:
        with ipc.new_file(stream, table.schema, options=ipc.IpcWriteOptions(compression='zstd')) as writer:
            writer.write_table(table)

def fixture(base):
    raw = base/'hpp_datasets'
    oral = raw/'oral_microbiome'
    project = base/'project'
    package = project/'collaborator_handoff_20261102_v1'
    private = package/'experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY'
    inputs = package/'inputs/discovery_v2'
    for path in (oral/'aggregated', private, inputs, project/'analysis/shared', project/'config'):
        path.mkdir(parents=True, exist_ok=True)
    dates = {pid:datetime(2020, 1, 1)+timedelta(days=pid-101) for pid in range(101, 107)}
    source = [dict(participant_id=pid, cohort='10k', research_stage='00_00_visit' if pid < 104 else 'undefined', array_index=0, collection_date=dates[pid], **{F[0]:'aggregated/abundance.arrow', F[1]:'aggregated/coverage.arrow'}) for pid in range(101, 106)]
    src_path = oral/'oral_microbiome.parquet'
    pq.write_table(pa.Table.from_pylist(source), src_path)
    cov_path = inputs/'discovery_covariates_food_TRE_ONLY.parquet'
    covariates = [dict(participant_id=pid, cohort='10k', research_stage='00_00_visit', input_row=i, final_role='discovery', split_unit_id='PRIVATE_HH_'+str(0 if pid < 103 else pid)) for i,pid in enumerate(range(101,106))]
    pq.write_table(pa.Table.from_pylist(covariates), cov_path)
    y = np.full((2,5), np.nan)
    y[:,:3] = 987654321.012345
    reference = private/'final_predictions.npz'
    np.savez(reference, y=y, PRIVATE_UNUSED_SCORE=np.array([123456789.765432]))
    registration = private.parent/'registration.json'
    registration.write_text(json.dumps({'source_and_input_sha256':{str(cov_path):sha(cov_path)}}))
    reg_path = project/'analysis/shared/sample_registry_TRE_ONLY.parquet'
    registry = []
    for i,row in enumerate(source):
        registry.append(dict({k:row[k] for k in K}, site='oral', sample_lock_status='locked_W13_independent_full_source_verification_passed', linked_index_row_ordinal=i+1, primary_index_source_path=str(src_path), primary_index_source_sha256=sha(src_path)))
    registry.append(dict(registry[0], site='gut', primary_index_source_path='PRIVATE_GUT_SOURCE'))
    pq.write_table(pa.Table.from_pylist(registry), reg_path)
    lock = project/'config/w13_sample_selection_lock.json'
    lock.write_text(json.dumps({'sample_registry_sha256':sha(reg_path)}))
    paths = [oral/'aggregated/abundance.arrow', oral/'aggregated/coverage.arrow']
    for j,path in enumerate(paths):
        rows = []
        for pid in [106,103,105,101,102,104]:
            rows.append(dict(participant_id=pid, cohort='10k', research_stage='undefined', array_index=0, collection_date=dates[pid], **{
                'PWY-1234:demo':float(pid-100)/(10 if j else 1),
                'GLYCOLYSIS-PWY:demo':None if pid==101 and j==0 else float('nan') if pid==102 and j==0 else float(pid-99)/(10 if j else 1),
                'VALID-OTHER-FEATURE':0., 'UNMAPPED':1000., 'UNINTEGRATED:total':2000.}))
        write_arrow(path, rows)
    (base/'.python_path').write_text(sys.executable+'\n')
    return dict(raw=raw, oral=oral, project=project, package=package, source=src_path, cov=cov_path, registry=reg_path, lock=lock, reference=reference, registration=registration, paths=paths)
HPP_FUNCTION_SOURCE_007_END
hpp_function_write 'frozen.py' '783382e5bae6ddd26d97226569867be69ec7e7cacfd2ed29cbc32fe812c52eea' <<'HPP_FUNCTION_SOURCE_008_END'
"""Read-only access to the frozen V12 background and matched ridge code."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_CACHE = None


def modules():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = Path(__file__).resolve().parent.parent
    before = sys.path[:]
    sys.path.insert(0, str(root))
    try:
        values = {}
        for name in ('upload_check', 'bridge', 'features', 'models', 'engine', 'runtime'):
            value = importlib.import_module(name)
            if Path(value.__file__).resolve().parent != root:
                raise RuntimeError('Frozen module name collision: ' + name)
            values[name] = value
    finally:
        sys.path[:] = before
    values.update(root=root, config=json.loads((root/'config.json').read_text()))
    _CACHE = SimpleNamespace(**values)
    return _CACHE
HPP_FUNCTION_SOURCE_008_END
hpp_function_write 'functional.py' 'fcefca3fa10d03ac95abe9b286c573dfbcbd280371ed8d3fc5e7dc7c280d3d3a' <<'HPP_FUNCTION_SOURCE_009_END'
"""Training-only representations; no selection by held-out outcomes."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class FixedMemberTarget:
    """Reuse training robust limits, keeping the previously fixed eligibility."""
    def __init__(self, original):
        self.original = original

    def fit(self, y, participant_ids, *, endpoint, role='training'):
        self.original.fit(y, participant_ids, endpoint=endpoint, role=role)
        self.state = dict(self.original.state)
        self.state['original_kind'] = self.state.get('kind')
        self.state['kind'] = 'functional_fixed_membership_training_clip_only_v1'
        self.state['new_8sd_exclusions_applied'] = False
        return self

    def transform(self, y, *, endpoint):
        result = self.original.transform(y, endpoint=endpoint)
        values = np.asarray(y, float)
        finite = np.isfinite(values)
        cleaned = np.full(values.shape, np.nan)
        cleaned[finite] = np.clip(values[finite], self.state['clip_low'], self.state['clip_high'])
        result['values'] = cleaned
        result['original_rule_excluded_8sd'] = result.get('excluded_8sd')
        result['excluded_8sd'] = np.zeros(values.shape, bool)
        result['winsorized_low'] = finite & (values < self.state['clip_low'])
        result['winsorized_high'] = finite & (values > self.state['clip_high'])
        return result


class FunctionState:
    def __init__(self, mode, config):
        self.mode, self.config = mode, config

    def fit(self, x):
        x = np.asarray(x, float)
        self.keep = np.flatnonzero(np.mean(np.isfinite(x) & (x > 0), axis=0) >= self.config['minimum_training_signal_prevalence'])
        self.observed = []
        if self.mode == 'abundance_unknown_null':
            self.observed = [np.sort(c[np.isfinite(c)]) for c in x[:, self.keep].T]
            ranks = self._ranks(x)
            self.rank_medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else .5 for c in ranks.T])
        raw = self.raw(x)
        self.input_scaler = StandardScaler(with_std=self.mode == 'abundance_unknown_null').fit(raw) if raw.shape[1] else None
        z = self.input_scaler.transform(raw) if self.input_scaler is not None else raw
        self.pca, self.pc_scaler = None, None
        self.pc_keep = np.array([], int)
        count = min(self.config['functional_main_components'], z.shape[1], max(len(z)-1, 0))
        if count and np.any(z.std(0) > 1e-10):
            self.pca = PCA(n_components=count, svd_solver='full').fit(z)
            scores = self.pca.transform(z)
            self.pc_keep = np.flatnonzero(scores.std(0) > 1e-10)
            if len(self.pc_keep):
                self.pc_scaler = StandardScaler().fit(scores[:, self.pc_keep])
        return self

    def _ranks(self, x):
        result = np.full((len(x), len(self.keep)), np.nan)
        for j, (column, observed) in enumerate(zip(self.keep, self.observed)):
            present = np.isfinite(x[:, column])
            values = x[present, column]
            if len(observed):
                result[present, j] = (np.searchsorted(observed, values, 'left') + np.searchsorted(observed, values, 'right')) / (2.0*len(observed))
        return result

    def raw(self, x):
        x = np.asarray(x, float)
        if np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
            raise ValueError('Invalid nonnegative functional signals')
        if self.mode == 'abundance_relative':
            values = np.where(np.isfinite(x), x, 0.)
            sums = values.sum(1)
            if (sums <= 0).any():
                raise ValueError('Entire missing/zero profile must remain flagged')
            return np.sqrt(values[:, self.keep]/sums[:, None])
        if self.mode == 'coverage_confidence':
            if np.any(x[np.isfinite(x)] > 1+1e-8):
                raise ValueError('Supplied coverage score outside observed range')
            return np.sqrt(np.where(np.isfinite(x[:, self.keep]), x[:, self.keep], 0.))
        if self.mode != 'abundance_unknown_null':
            raise ValueError('Unknown functional route')
        ranks = self._ranks(x)
        return np.column_stack([np.where(np.isfinite(ranks), ranks, self.rank_medians), ~np.isfinite(ranks)]).astype(float)

    def transform(self, x):
        raw = self.raw(np.asarray(x, float))
        if self.pca is None or self.pc_scaler is None:
            return np.empty((len(raw), 0))
        z = self.input_scaler.transform(raw)
        return self.pc_scaler.transform(self.pca.transform(z)[:, self.pc_keep])

    def audit(self, names):
        result = dict(mode=self.mode, retained_fields=len(self.keep), pc_dimensions=len(self.pc_keep))
        if self.pca is not None:
            result['explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
            labels = [names[j] for j in self.keep]
            if self.mode == 'abundance_unknown_null':
                labels = ['observed_rank:'+n for n in labels] + ['reporting_mask:'+n for n in labels]
            result['unsupervised_top_loadings'] = []
            for pc in self.pca.components_[self.pc_keep[:self.config['functional_modifier_components']]]:
                selected = np.argsort(-abs(pc), kind='stable')[:8]
                result['unsupervised_top_loadings'].append([dict(field=labels[j], loading=float(pc[j])) for j in selected])
        return result


class TechnicalState:
    def raw(self, x):
        a = np.asarray(x, float)
        good = np.isfinite(a) & (a >= 0)
        result = np.full(a.shape, np.nan)
        result[good] = np.log1p(a[good])
        return result

    def fit(self, x):
        a = self.raw(x)
        self.median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in a.T])
        self.scaler = StandardScaler().fit(self.complete(a))
        return self

    def complete(self, a):
        return np.column_stack([np.where(np.isfinite(a), a, self.median), ~np.isfinite(a)]).astype(float)

    def transform(self, x):
        return self.scaler.transform(self.complete(self.raw(x)))


class Design:
    def __init__(self, base, mode, signals, technical, config, frozen_features):
        self.base, self.role, self.config = base, base.role, base.config
        self.functional_config, self.frozen_features = config, frozen_features
        self.mode = mode
        s = self.role.split
        self.function = FunctionState(mode, config).fit(signals[s.train_rows])
        self.pcs = [self.function.transform(signals[rows]) for rows in (s.train_rows, s.rows)]
        number = min(config['functional_curved_components'], self.pcs[0].shape[1])
        self.function_curves = frozen_features.Curves().fit(self.pcs[0][:,:number]) if number else None
        self.curved_components = number
        self.technical_state = TechnicalState().fit(technical[s.train_rows])
        self.technical = [self.technical_state.transform(technical[rows]) for rows in (s.train_rows, s.rows)]
        self.extra_scaler = StandardScaler().fit(self.extra(True))
        self.xt, self.xv = self.main(True), self.main(False)
        self.interaction_states = {}

    def extra(self, training, value=None):
        pos = 0 if training else 1
        diet, tech = self.base.diet_basis(training, value), self.technical[pos]
        curves = self.function_curves.transform(self.pcs[pos][:,:self.curved_components]) if self.function_curves is not None else np.empty((len(tech),0))
        return np.column_stack([self.pcs[pos], curves, tech, tech*diet[:, :1], tech*diet[:, 1:2]])

    def main(self, training=False, value=None):
        return np.column_stack([self.base.main(training, value), self.extra_scaler.transform(self.extra(training, value))])

    def interaction_raw(self, spec, training, value=None):
        pos = 0 if training else 1
        number = self.functional_config['functional_modifier_components']
        return self.pcs[pos][:, :number]*self.base.diet_basis(training, value)[:, :1]

    def interactions(self, spec):
        if self.pcs[0].shape[1] == 0:
            state = self.frozen_features.InteractionMap(np.empty((self.xt.shape[1], 0)), np.array([], int), None)
            return np.empty((len(self.xt), 0)), np.empty((len(self.xv), 0)), state
        return self.frozen_features.Design.interactions(self, spec)

    def support(self):
        pieces = []
        for pos in (0,1):
            background = self.base.background_t if pos == 0 else self.base.background_v
            pieces.append(np.column_stack([background, self.base.pcs['oral'][pos][:, :6], self.base.pcs['gut'][pos][:, :6], self.pcs[pos][:, :3], self.technical[pos]]))
        scale = StandardScaler().fit(pieces[0])
        n = min(self.functional_config['support_neighbors'], len(pieces[0]))
        neighbors = NearestNeighbors(n_neighbors=n).fit(scale.transform(pieces[0]))
        rows = neighbors.kneighbors(scale.transform(pieces[1]), return_distance=False)
        limits = np.quantile(self.base.diet(True)[rows], self.functional_config['support_quantiles'], axis=1)
        lo, hi = self.config['contrast']
        return (limits[0] <= lo) & (limits[1] >= hi)
HPP_FUNCTION_SOURCE_009_END
hpp_function_write 'native.py' '761492bdf30fb074d18d48a159a93d239089fe6ecd79e31927beab00dc271884' <<'HPP_FUNCTION_SOURCE_010_END'
"""Only native reader. All participant data and derived arrays stay inside TRE."""
import contextlib
import hashlib
import io
import json
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from frozen import modules
from functional import FixedMemberTarget


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(project, raw_root, config):
    project, raw_root = Path(project), Path(raw_root)
    root = Path(__file__).resolve().parent
    frozen = modules()
    if frozen.upload_check.verify(frozen.root) != 51:
        raise ValueError('Original V12 source verification failed')
    versions = frozen.runtime.environment(frozen.config)
    gate = frozen.runtime.readiness(frozen.config, prior_summary_reviewed=True)
    argv, handler = sys.argv[:], signal.getsignal(signal.SIGALRM)
    output = io.StringIO()
    try:
        sys.argv = [str(root/'audit_driver.py'), str(raw_root), str(project), str(config['original_people']), str(config['source_input_rows']), str(config['original_households'])]
        with contextlib.redirect_stdout(output):
            namespace = runpy.run_path(str(root/'audit_driver.py'), run_name='__hpp_function_audit__')
    except SystemExit as error:
        print('SOURCE_AUDIT_INCOMPLETE: ' + ' | '.join(output.getvalue().splitlines()[-3:]), flush=True)
        raise RuntimeError('Source audit did not complete') from error
    finally:
        sys.argv = argv
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
    if not namespace.get('complete') or 'PATHWAY_NULL_CONTRACT_DONE' not in output.getvalue():
        print('SOURCE_AUDIT_INCOMPLETE: ' + ' | '.join(output.getvalue().splitlines()[-3:]), flush=True)
        raise RuntimeError('Source audit did not establish complete original records')
    records, matrices = namespace['records'], namespace['matrices']
    if records['abundance']['columns'] != records['coverage']['columns']:
        raise ValueError('Function candidate labels changed')
    labels = records['abundance']['columns']
    if len(labels) != config['candidate_fields']:
        raise ValueError('Candidate field count changed')
    null_masks = namespace['ARROW_NULL_MASKS']
    if not np.array_equal(null_masks['abundance'], null_masks['coverage']):
        raise ValueError('Paired function reporting masks changed')
    for name, values in matrices.items():
        if not np.array_equal(np.isnan(values), null_masks[name]) or np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError('Non-null invalid functional values')
    coverage = matrices['coverage']
    if np.any(coverage[np.isfinite(coverage)] > 1+1e-8):
        raise ValueError('Coverage range changed')
    package = project/'collaborator_handoff_20261102_v1'
    dataset = frozen.bridge.load(package, frozen.config)
    original_target = dataset.api.PhenotypeOutliers
    dataset.api.PhenotypeOutliers = lambda:FixedMemberTarget(original_target())
    original = np.isfinite(dataset.reference['y'])
    if not np.array_equal(original, namespace['eligible']):
        raise ValueError('V12 membership differs from audited source')
    if not dataset.index[namespace['K']].reset_index(drop=True).equals(namespace['x'][namespace['K']].reset_index(drop=True)):
        raise ValueError('Native index keys differ from audit')
    positions = namespace['target'].input_row.to_numpy(int)
    usable = np.isfinite(matrices['abundance']).any(1) & np.isfinite(matrices['coverage']).any(1)
    if len(positions) != config['original_people'] or int(usable.sum()) != config['functional_people']:
        raise ValueError('Predefined functional population changed')
    available = np.zeros(len(dataset.index), bool)
    available[positions[usable]] = True
    signals = {}
    for role in ('abundance', 'coverage'):
        values = np.full((len(dataset.index), len(labels)), np.nan)
        values[positions] = matrices[role]
        signals[role] = values
    source_path = namespace['src_path']
    source_digest = namespace['src_hash']
    if sha(source_path) != source_digest:
        raise ValueError('Source index changed since audit')
    technical_names = ['raw_read_count', 'metaphlan4_aligned_read_count']
    with pq.ParquetFile(source_path, pre_buffer=False) as pf:
        present = [n for n in technical_names if n in pf.schema_arrow.names]
        metadata = pf.read(columns=present, use_threads=False).to_pydict() if present else {}
    technical = np.full((len(dataset.index), 3), np.nan)
    for j,name in enumerate(technical_names):
        if name in metadata:
            for output_row, source_row in zip(positions, namespace['approved']):
                value = metadata[name][source_row]
                try:
                    value = float(value)
                    if np.isfinite(value) and value >= 0:
                        technical[output_row,j] = value
                except (TypeError, ValueError, OverflowError):
                    pass
    ratio_ok = np.isfinite(technical[:,0]) & (technical[:,0] > 0) & np.isfinite(technical[:,1])
    technical[ratio_ok,2] = technical[ratio_ok,1]/technical[ratio_ok,0]
    paths = dataset.paths + frozen.bridge.adapter_sources()
    paths += [source_path, namespace['reg_path'], project/'config/w13_sample_selection_lock.json', namespace['v4']/'registration.json']
    paths += [v['path'] for v in records.values()]
    paths += [frozen.root/n for n in ('config.json','features.py','models.py','engine.py','bridge.py','runtime.py','upload_check.py','upload_manifest.json')]
    paths += [Path(gate['v11_summary'])]
    hashes = {str(Path(p)):sha(p) for p in dict.fromkeys(paths)}
    if hashes[str(source_path)] != source_digest:
        raise ValueError('Source changed during native loading')
    return SimpleNamespace(dataset=dataset, frozen=frozen, available=available, signals=signals, technical=technical,
        labels=labels, source_hashes=hashes, versions=versions, audit_text=output.getvalue(), synthetic_only=False,
        audit=dict(original_people=len(positions), functional_people=int(available.sum()), excluded_whole_candidate_null=int((~usable).sum()),
                   functional_households=int(dataset.index.loc[available,'split_unit_id'].nunique()), technical_fields=present,
                   partial_null_nonreporting_is_assumption=True, provider_abundance_unit_verified=False,
                   confirmation_outcomes_read=False, independent_confirmation=False))
HPP_FUNCTION_SOURCE_010_END
hpp_function_write 'pilot.py' '5740847289b97775e53ea90e10bd68b19aa75a197ee770360e5d287bc38c4ffd' <<'HPP_FUNCTION_SOURCE_011_END'
"""Bounded nested-household functional pilot. Native data never leave TRE."""
import argparse
import hashlib
import io
import json
import os
import re
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from functional import Design
from native import sha

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    os.replace(str(temporary), str(path))


def save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.stem+'.tmp-'+uuid.uuid4().hex+'.npz')
    # ZIP needs backwards seeks while it is assembled; the mounted destination
    # may only support sequential writes. Finish the archive before opening it.
    with io.BytesIO() as memory:
        np.savez_compressed(memory, **arrays)
        payload = memory.getvalue()
    try:
        with temporary.open('xb') as stream:
            written = stream.write(payload)
        if written != len(payload) or sha(temporary) != hashlib.sha256(payload).hexdigest():
            raise OSError('Sequential checkpoint readback mismatch')
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def design(base_engine, data, train, valid, config):
    data.dataset.api.separated(data.dataset.index.split_unit_id.astype(str).to_numpy(), train, valid)
    base = base_engine.design(data.dataset, train, valid)
    s = base.role.split
    if not np.array_equal(s.train_rows, np.flatnonzero(train)) or not np.array_equal(s.rows, np.flatnonzero(valid)):
        raise ValueError('Predefined functional members changed during frozen preprocessing')
    return base


def wrapper(base, data, route, config):
    role = 'coverage' if route == 'coverage_confidence' else 'abundance'
    return Design(base, route, data.signals[role], data.technical, config, data.frozen.features)


def outer(data, config, repeat, fold, log=print):
    ds, frozen = data.dataset, data.frozen
    groups = ds.index.split_unit_id.astype(str).to_numpy()
    assignment = ds.reference['outer'][repeat]
    train = data.available & (assignment != fold)
    valid = data.available & (assignment == fold)
    ds.api.separated(groups, train, valid)
    inner = np.full(len(groups), -1, int)
    inner[train] = ds.api.folds(groups[train], config['inner_folds'], config['outer_seed']+100*repeat+fold+1)
    engine = frozen.engine.Engine(frozen.config, None)
    log('START r%d_f%d inner training roles' % (repeat, fold))
    bases = [design(engine, data, train & (inner != k), train & (inner == k), config) for k in range(config['inner_folds'])]
    target = np.concatenate([b.role.split.yv for b in bases])
    selections = {}
    for route in config['routes']:
        views = [wrapper(base, data, route, config) for base in bases]
        predictions = [np.column_stack([frozen.models.fit_main(v, p).predict() for p in config['main_penalties']]) for v in views]
        loss = np.mean((target[:,None]-np.concatenate(predictions))**2, axis=0)
        chosen = int(loss.argmin())
        main_penalty = config['main_penalties'][chosen]
        spec = dict(name=route)
        candidates = [None]+config['interaction_penalties']
        augmented = [np.column_stack([predictions[k][:,chosen]] + [frozen.models.fit_augmented(v, spec, main_penalty, p).predict() for p in config['interaction_penalties']]) for k,v in enumerate(views)]
        interaction_loss = np.mean((target[:,None]-np.concatenate(augmented))**2, axis=0)
        selections[route] = dict(main_penalty=main_penalty, interaction_penalty=candidates[int(interaction_loss.argmin())],
                                 inner_main_MSE=loss.tolist(), inner_interaction_MSE=interaction_loss.tolist())
    del bases, views, predictions, augmented, target
    log('FIT r%d_f%d outer models; selection fixed' % (repeat, fold))
    base = design(engine, data, train, valid, config)
    s = base.role.split
    arrays = dict(rows=s.rows, y=s.yv, raw_y=ds.y[s.rows])
    main, full, total, delta, component, supported = [], [], [], [], [], []
    audits = {}
    for route in config['routes']:
        view = wrapper(base, data, route, config)
        setting = selections[route]
        baseline = frozen.models.fit_main(view, setting['main_penalty'])
        model = frozen.models.fit_augmented(view, dict(name=route), setting['main_penalty'], setting['interaction_penalty'], baseline)
        main.append(baseline.predict())
        full.append(model.predict())
        baseline_contrast, _ = baseline.contrasts()
        contrast, raw_component = model.contrasts()
        total.append(contrast)
        delta.append(contrast-baseline_contrast)
        component.append(raw_component)
        supported.append(view.support())
        audits[route] = view.function.audit(data.labels)
    arrays.update(main=np.asarray(main), interaction=np.asarray(full), total_contrast=np.asarray(total),
                  delta_contrast=np.asarray(delta), raw_component=np.asarray(component), support=np.asarray(supported, bool))
    if not all(np.isfinite(arrays[k]).all() for k in ('y','raw_y','main','interaction','total_contrast','delta_contrast','raw_component')):
        raise ValueError('Nonfinite outer values')
    metadata = dict(repeat=repeat, fold=fold, people=len(s.rows), train_people=len(s.train_rows), selections=selections, function_audits=audits,
                    endpoint_preprocessing=s.preprocessing.get('target',{}) if hasattr(s,'preprocessing') else dict(synthetic_only=True))
    log('OUTER_COMPLETE r%d_f%d people=%d routes=%d' % (repeat, fold, len(s.rows), len(config['routes'])))
    return arrays, metadata


def checkpoint(private, tag, identity_hash, compute):
    complete = private/(tag+'_COMPLETE.json')
    array_path, meta_path = private/(tag+'.npz'), private/(tag+'.json')
    if complete.exists():
        proof = json.loads(complete.read_text())
        if proof['identity_hash'] != identity_hash or proof['arrays_sha256'] != sha(array_path) or proof['metadata_sha256'] != sha(meta_path):
            raise ValueError('Checkpoint identity or checksum mismatch')
        with np.load(str(array_path), allow_pickle=False) as archive:
            return {name:archive[name] for name in archive.files}, json.loads(meta_path.read_text())
    start = time.monotonic()
    arrays, metadata = compute()
    metadata['seconds'] = time.monotonic()-start
    save_arrays(array_path, arrays)
    write_json(meta_path, metadata)
    write_json(complete, dict(identity_hash=identity_hash, arrays_sha256=sha(array_path), metadata_sha256=sha(meta_path)))
    return arrays, metadata


def metric(y, prediction):
    sst = float(np.sum((y-y.mean())**2))
    if not np.isfinite(sst) or sst <= 0:
        raise ValueError('Outcome variance unavailable')
    mse = float(np.mean((y-prediction)**2))
    return dict(R2=1-len(y)*mse/sst, MSE=mse)


def paired_intervals(y, main, full, groups, draws, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 30:
        raise ValueError('Too few households for conditional bootstrap')
    delta = (y[None,:]-main)**2-(y[None,:]-full)**2
    table = np.column_stack([np.bincount(inverse), np.bincount(inverse,weights=y), np.bincount(inverse,weights=y*y)] + [np.bincount(inverse,weights=v) for v in delta])
    rng = np.random.default_rng(seed)
    result = []
    for begin in range(0,draws,100):
        weights = rng.multinomial(len(unique), np.full(len(unique),1./len(unique)), size=min(100,draws-begin))
        totals = weights @ table
        denominator = totals[:,2]-totals[:,1]**2/totals[:,0]
        good = denominator > 0
        result.append(totals[good,3:]/denominator[good,None])
    samples = np.concatenate(result)
    if len(samples) < .95*draws:
        raise ValueError('Insufficient nondegenerate household draws')
    return np.quantile(samples,[.025,.975],axis=0).T


def summarize(data, config, chunks):
    n, routes = len(data.available), config['routes']
    names = ('y','raw_y','main','interaction','total_contrast','delta_contrast','raw_component','support')
    all_arrays = {}
    for name in names:
        shape = (config['outer_repeats'],n) if name in ('y','raw_y') else (config['outer_repeats'],len(routes),n)
        all_arrays[name] = np.full(shape,np.nan)
    seen = np.zeros((config['outer_repeats'],n),int)
    selected = {name:[0]*config['outer_repeats'] for name in routes}
    for arrays, meta in chunks:
        repeat, rows = meta['repeat'], arrays['rows']
        seen[repeat,rows] += 1
        for name in names:
            if name in ('y','raw_y'):
                all_arrays[name][repeat,rows] = arrays[name]
            else:
                all_arrays[name][repeat][:,rows] = arrays[name]
        for route in routes:
            selected[route][repeat] += meta['selections'][route]['interaction_penalty'] is not None
    if not np.all(seen[:,data.available] == 1) or np.any(seen[:,~data.available]):
        raise ValueError('Outer evaluation membership missing or duplicated')
    groups = data.dataset.index.loc[data.available,'split_unit_id'].astype(str).to_numpy()
    scores = []
    for repeat in range(config['outer_repeats']):
        y = all_arrays['y'][repeat,data.available]
        main = all_arrays['main'][repeat][:,data.available]
        full = all_arrays['interaction'][repeat][:,data.available]
        intervals = paired_intervals(y,main,full,groups,config['prediction_bootstraps'],config['seed']+repeat)
        for j,route in enumerate(routes):
            baseline, augmented = metric(y,main[j]), metric(y,full[j])
            raw_y = all_arrays['raw_y'][repeat,data.available]
            raw_delta = metric(raw_y,full[j])['R2']-metric(raw_y,main[j])['R2']
            scores.append(dict(route=route,repeat=repeat,N=int(data.available.sum()),baseline=baseline,interaction=augmented,
                delta_R2=augmented['R2']-baseline['R2'],MSE_reduction=baseline['MSE']-augmented['MSE'],
                conditional_household_delta_R2_CI95=intervals[j].tolist(),raw_y_same_prediction_delta_R2=raw_delta,
                interaction_selected_folds=selected[route][repeat],
                total_model_contrast_SD=float(all_arrays['total_contrast'][repeat,j,data.available].std()),
                added_model_contrast_SD=float(all_arrays['delta_contrast'][repeat,j,data.available].std()),
                local_contrast_support_fraction=float(all_arrays['support'][repeat,j,data.available].mean())))
    stability = {}
    for j,route in enumerate(routes):
        a,b = [all_arrays['delta_contrast'][r,j,data.available] for r in (0,1)]
        stability[route] = float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None
    return dict(status='COMPLETE',synthetic_only=data.synthetic_only,claim_type=config['claim_type'],
                scope='Same discovery cohort; fixed predictions conditional intervals; no formal interaction P value or causal response.',
                population=data.audit,primary_route=config['primary_route'],primary_repeat=0,scores=scores,
                same_people_added_contrast_split_correlation=stability,
                historical_discovery_search=True,independent_confirmation=False,confirmation_outcomes_read=False), all_arrays


def report(summary):
    print(('SYNTHETIC ONLY' if summary['synthetic_only'] else 'DISCOVERY FUNCTION PILOT')+' | fiber_density -> D_Glu (mg/dL)')
    print('Population:',json.dumps(summary['population'],sort_keys=True))
    print('PRIMARY abundance_relative, repeat 0; other encodings and repeat 1 are sensitivity only.')
    print('route repeat base_R2 full_R2 delta_R2 conditional_HH_CI95 selected_folds rawY_delta support added_contrast_SD')
    for row in summary['scores']:
        lo,hi = row['conditional_household_delta_R2_CI95']
        print('%s %d %.6f %.6f %+.6f [%+.6f,%+.6f] %d %+.6f %.3f %.3f' % (
            row['route'],row['repeat'],row['baseline']['R2'],row['interaction']['R2'],row['delta_R2'],lo,hi,
            row['interaction_selected_folds'],row['raw_y_same_prediction_delta_R2'],row['local_contrast_support_fraction'],row['added_model_contrast_SD']))
    print('Same-person split contrast correlation:',json.dumps(summary['same_people_added_contrast_split_correlation'],sort_keys=True))
    print('Matched baselines are refit within each route on the same functional subset; do not compare with old N=3343 R2.')
    print('CIs resample households with OOF predictions fixed; no learning/selection uncertainty or formal interaction P value.')
    print('Null encoding is an assumption; contrasts are model-generated associations, not measured individual diet responses.')
    print('Eligibility stays fixed; outcome clipping limits use training roles only, with no new validation-outcome exclusions.')


def execute(data, config, run, threads=2, resume=False, log=print):
    run = Path(run)
    run.mkdir(parents=True,exist_ok=True)
    private = run/'TRE_ONLY'
    private.mkdir(exist_ok=True)
    code = {str(p):sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.sh') and p.name not in ('terminal_install_and_run.sh','verification.json')}
    identity = dict(config=config,code_hashes=code,source_hashes=data.source_hashes,versions=data.versions,
                    python=sys.version,threads=threads,population=data.audit,synthetic_only=data.synthetic_only)
    identity_hash = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    registration, lock = run/'registration.json', run/'RUNNING.lock'
    if registration.exists():
        if json.loads(registration.read_text()) != identity:
            raise ValueError('Resume source/config/environment identity mismatch')
        if (run/'summary.json').exists():
            summary = json.loads((run/'summary.json').read_text())
            proof = json.loads((run/'COMPLETE.json').read_text())
            if proof['identity_hash'] != identity_hash or proof['summary_sha256'] != sha(run/'summary.json') or proof['arrays_sha256'] != sha(private/'final_predictions.npz') or proof['cohort_flags_sha256'] != sha(private/'cohort_flags.npz'):
                raise ValueError('Completed summary checksum mismatch')
            return summary
        if not resume:
            raise ValueError('Existing incomplete run; use identical sources with --resume')
    else:
        if resume:
            raise ValueError('No registered run to resume')
        write_json(registration,identity)
        (run/'source_audit.txt').write_text(data.audit_text)
    if lock.exists():
        previous = json.loads(lock.read_text())
        if not resume or previous['host'] != socket.gethostname():
            raise RuntimeError('Run lock exists; inspect its owner')
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError('Run lock owner still exists')
    with lock.open('x') as stream:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname()),stream)
    chunks=[]
    try:
        flag_path=private/'cohort_flags.npz'
        expected_flags=dict(original_eligible=np.isfinite(data.dataset.reference['y'][0]),functional_available=data.available,
                            original_eligible_without_function=np.isfinite(data.dataset.reference['y'][0]) & ~data.available)
        if flag_path.exists():
            with np.load(str(flag_path),allow_pickle=False) as previous_flags:
                if any(not np.array_equal(previous_flags[n],v) for n,v in expected_flags.items()):
                    raise ValueError('Stored cohort flags differ from registered population')
        else:
            save_arrays(flag_path,expected_flags)
        with threadpool_limits(limits=threads):
            for repeat in range(config['outer_repeats']):
                for fold in range(config['outer_folds']):
                    tag='r%d_f%d' % (repeat,fold)
                    chunks.append(checkpoint(private,tag,identity_hash,lambda repeat=repeat,fold=fold:outer(data,config,repeat,fold,log)))
            for path,digest in data.source_hashes.items():
                if sha(path) != digest:
                    raise ValueError('Source changed during run; summary not published')
            summary, arrays = summarize(data,config,chunks)
            save_arrays(private/'final_predictions.npz',arrays)
            write_json(run/'summary.json',summary)
            write_json(run/'COMPLETE.json',dict(identity_hash=identity_hash,summary_sha256=sha(run/'summary.json'),arrays_sha256=sha(private/'final_predictions.npz'),cohort_flags_sha256=sha(flag_path)))
    except BaseException:
        raise
    else:
        lock.unlink()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
    parser.add_argument('--raw-root',default='/home/ec2-user/studies/hpp_datasets')
    parser.add_argument('--run-id',default='function_20261110_v1_iofix1')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.run_id) or not 1 <= args.threads <= 8:
        raise ValueError('Invalid run ID or thread count')
    run=ROOT/'runs'/args.run_id
    if args.report:
        proof=json.loads((run/'COMPLETE.json').read_text())
        if sha(run/'summary.json') != proof['summary_sha256']:
            raise ValueError('Summary checksum mismatch')
        report(json.loads((run/'summary.json').read_text()))
        return
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).name != name or sha(ROOT/name) != digest:
            raise ValueError('Functional source manifest mismatch')
    config=json.loads((ROOT/'config.json').read_text())
    from native import load
    print('START native functional pilot; source audit first; no confirmation outcomes.',flush=True)
    data=load(args.project,args.raw_root,config)
    print('SOURCE_READY functional_people=%d original_people=%d' % (data.audit['functional_people'],data.audit['original_people']),flush=True)
    summary=execute(data,config,run,args.threads,args.resume,lambda message:print(message,flush=True))
    report(summary)


if __name__ == '__main__':
    main()
HPP_FUNCTION_SOURCE_011_END
hpp_function_write 'server_run.sh' '8e7d81c72f09a45d9dc5a55e0772410a007dad2bf48e19d6018c5ab9461cc1fd' <<'HPP_FUNCTION_SOURCE_012_END'
#!/usr/bin/env bash
set -euo pipefail
hpp_function_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
hpp_function_python=python
if [ -s "$hpp_function_dir/../.python_path" ]; then
    IFS= read -r hpp_function_python < "$hpp_function_dir/../.python_path"
fi
cd "$hpp_function_dir"
export PYTHONDONTWRITEBYTECODE=1
"$hpp_function_python" -B - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd()
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files'].items():
    path=root/name
    if Path(name).name != name or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('FUNCTIONAL_UPLOAD_CHECK_FAILED')
print('FUNCTIONAL_UPLOAD_CHECK_OK files=%d' % len(manifest['files']))
PY
hpp_function_report_only=false
for hpp_function_arg in "$@"; do
    if [ "$hpp_function_arg" = --report ]; then
        hpp_function_report_only=true
    fi
done
if [ "$hpp_function_report_only" = false ]; then
    "$hpp_function_python" -B check_synthetic.py
fi
"$hpp_function_python" -B -u pilot.py "$@"
HPP_FUNCTION_SOURCE_012_END
hpp_function_write 'manifest.json' '3a99e664252c1ecd82c48d3c0cabb2f7098f03bd8724c18f84a684eb2d7820bd' <<'HPP_FUNCTION_SOURCE_013_END'
{
  "analysis_configuration_unchanged": true,
  "files": {
    "IMPLEMENTATION_PLAN.md": "ba8fb78c33145fe08eac0ce7895519b59a6463e6cd87c64a6ccbb24bd7ca58fb",
    "PROTOCOL.md": "bb845b41d395cf6748a401e7077b1c86f52432e2f16c4bfc815326c1091ab123",
    "README.md": "e9b1879e6af5c3999c33e4090b3b44c93b506f1b78ce5aaa0496ef34c48e3158",
    "audit_driver.py": "0c2186ecfc4b42456437f5145847addf807d4cff9979088e30f57bcccfbcf92c",
    "check_synthetic.py": "a41f7f96ab80dc55204ed14511a456e1725d14b65cca65dad70804d258f31f8b",
    "config.json": "5f4fe8a03443b1623a9c2a980e3e8aadf3188148ab65290982ddf3cd898e3102",
    "demo_data.py": "4a25c6af3289ed250f42307ffa1ccebbe9ddc72abee81f293b0ebedf94a2efab",
    "demo_files.py": "ec6f1e5a4f8a972b1473c425827e7a4424a15e4cfaa5d4f11cfa4d9a9c57b965",
    "frozen.py": "783382e5bae6ddd26d97226569867be69ec7e7cacfd2ed29cbc32fe812c52eea",
    "functional.py": "fcefca3fa10d03ac95abe9b286c573dfbcbd280371ed8d3fc5e7dc7c280d3d3a",
    "native.py": "761492bdf30fb074d18d48a159a93d239089fe6ecd79e31927beab00dc271884",
    "pilot.py": "5740847289b97775e53ea90e10bd68b19aa75a197ee770360e5d287bc38c4ffd",
    "server_run.sh": "8e7d81c72f09a45d9dc5a55e0772410a007dad2bf48e19d6018c5ab9461cc1fd"
  },
  "native_HPP_inputs_embedded": false,
  "original_V12_files_modified": false,
  "storage_repair": "In-memory ZIP assembly followed by sequential hash-checked writes; fresh IO-fix run ID preserves failed registration.",
  "version": "functional_followup_20261110_v1_iofix1"
}
HPP_FUNCTION_SOURCE_013_END
echo 'FUNCTIONAL_SOURCES_READY: isolated child folder; previous V12 files retained.'
if [ "${HPP_FUNCTIONAL_INSTALL_ONLY:-0}" = 1 ]; then
    echo 'INSTALL_ONLY_DONE; no models started.'
    exit 0
fi
cd "$hpp_function_target"
bash server_run.sh --threads 2
HPP_FUNCTION_INSTALL_AND_RUN
