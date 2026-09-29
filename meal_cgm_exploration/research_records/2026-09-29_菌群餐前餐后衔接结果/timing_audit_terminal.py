"""Aggregate timing audit. Run from meal_cgm_exploration; no models or source writes."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from hppmeal.common import read_csv, header, norm, unique, sha256, require
from gut_meal_bridge_v2.inputs import date_only


def audit(r):
    m = json.loads((r / 'manifest.json').read_text())
    spec = json.loads((r / 'analysis_spec.json').read_text())
    state = m['status']['gut_meal_bridge_v2']
    require(state['status'] in ['COMPLETE', 'COMPLETE_WITH_LIMITATIONS'], 'Run incomplete')
    c = m['config']
    tz = c['time']['local_timezone']
    f = r / 'private/person_inputs.csv'
    require(sha256(f) == state['result_sha256']['private/person_inputs.csv'], 'Person file changed')
    p = read_csv(f)
    for key in ['participant_id', 'reference_connection']:
        p[key] = norm(p[key])
    unique(p, ['participant_id'], 'Person table')
    require(p.reference_connection.notna().all(), 'Missing connection ID')
    require(p.strict_eligible.isin(['True', 'False']).all(), 'Invalid strict flag')
    for key in ['sample_date', 'first_date', 'last_date']:
        raw = p[key]
        p[key] = pd.to_datetime(raw.apply(lambda x: date_only(x, tz)))
        require(not (raw.notna() & p[key].isna()).any(), 'Invalid date: ' + key)
    require(p.first_date.le(p.last_date).all(), 'Invalid meal date range')
    for endpoint, key in [('first_date', 'stool_minus_first_meal_days'),
                          ('last_date', 'stool_minus_last_meal_days')]:
        gap = (p.sample_date - p[endpoint]).dt.days
        old = pd.to_numeric(p[key], errors='coerce')
        require((gap.eq(old) | (gap.isna() & old.isna())).all(), 'Saved gap mismatch')

    ids = ['participant_id', 'cohort', 'research_stage', 'connection_id']
    fields = ['cgm_connection_first_date', 'cgm_connection_last_date']
    p['cgm_start'], p['cgm_end'] = pd.NaT, pd.NaT
    source = 'unavailable'
    notes = []
    candidates = [Path(spec['sources']['whole_cgm']), Path(m['root']) / 'Transfer/cgm/cgm.csv']
    for src in candidates:
        if not src.is_file() or not set(ids + fields).issubset(header(src)):
            continue
        digest = sha256(src)
        pins = [x['sha256'] for x in m['inputs'] if Path(x['path']) == src]
        require(not pins or all(x == digest for x in pins), 'CGM metadata changed')
        d = read_csv(src, usecols=ids + fields)
        for key in ids:
            d[key] = norm(d[key])
        d = d.loc[d.cohort.eq(c['cohort']) & d.research_stage.eq(c['research_stage'])]
        d = d.merge(p[['participant_id', 'reference_connection']],
                    left_on=['participant_id', 'connection_id'],
                    right_on=['participant_id', 'reference_connection'], how='inner')
        unique(d, ['participant_id'], 'Matched CGM metadata')
        d = d.set_index('participant_id')
        for dest, col in zip(['cgm_start', 'cgm_end'], fields):
            p[dest] = pd.to_datetime(p.participant_id.map(d[col]).apply(lambda x: date_only(x, tz)))
        source = src.name + (' (pinned)' if pins else ' (current metadata)') + ' SHA256=' + digest
        if p.cgm_start.notna().all() and p.cgm_end.notna().all():
            break

    if not (p.cgm_start.notna().all() and p.cgm_end.notna().all()):
        oldrun = Path(spec['meal_run'])
        cache = oldrun / 'private/cache.sqlite'
        if cache.is_file():
            oldm = json.loads((oldrun / 'manifest.json').read_text())
            expected = oldm['status']['audit'].get('cache_sha256')
            print('正在核对旧 CGM 缓存完整性；缓存较大时需稍等。', flush=True)
            require(expected and sha256(cache) == expected, 'CGM cache changed/unpinned')
            stamp = (cache.stat().st_size, cache.stat().st_mtime_ns)
            con = sqlite3.connect(cache.resolve().as_uri() + '?mode=ro', uri=True)
            try:
                sql = ('SELECT t FROM cgm INDEXED BY cgm_key WHERE participant_id=? '
                       'AND cohort=? AND research_stage=? AND connection_id=? '
                       'ORDER BY t {} LIMIT 1')
                for order, dest in [('ASC', 'cgm_start'), ('DESC', 'cgm_end')]:
                    values = []
                    for row in p.itertuples():
                        hit = con.execute(sql.format(order), (row.participant_id,
                            c['cohort'], c['research_stage'], row.reference_connection)).fetchone()
                        values.append(pd.NaT if not hit else pd.to_datetime(hit[0], unit='s', utc=True)
                                      .tz_convert(tz).tz_localize(None).normalize())
                    p[dest] = pd.to_datetime(values)
            finally:
                con.close()
            require(stamp == (cache.stat().st_size, cache.stat().st_mtime_ns), 'Cache modified during audit')
            source = 'frozen cache; first/last timestamped records; SHA256=' + expected
        else:
            notes.append('CGM 缓存不存在；仅报告可核对的元数据日期，缺失不补填。')
    valid = p.cgm_start.notna() & p.cgm_end.notna()
    require(p.loc[valid, 'cgm_start'].le(p.loc[valid, 'cgm_end']).all(), 'Invalid CGM range')
    outside = valid & (p.first_date.lt(p.cgm_start) | p.last_date.gt(p.cgm_end))
    require(not outside.any(), 'Meal dates outside matched CGM range; check source dates')

    lines = ['TIMING AUDIT | ' + r.name, 'DATA=' + m['data_kind'] + ' | 只核对时间；无新模型',
             'CGM source: ' + source, '日期匹配状态: ' + '; '.join(
                 '{}={}'.format(k, v) for k, v in p.timing_status.value_counts(dropna=False).items())]
    rows, gaps, bins = [], [], []
    for name, q in [('Main', p), ('Strict', p.loc[p.strict_eligible.eq('True')])]:
        for window, start, end in [('Meals', 'first_date', 'last_date'), ('CGM', 'cgm_start', 'cgm_end')]:
            ok = q.sample_date.notna() & q[start].notna() & q[end].notna()
            before = ok & q.sample_date.lt(q[start])
            after = ok & q.sample_date.gt(q[end])
            rows.append([name, window, len(q), int(before.sum()), int((ok & ~before & ~after).sum()),
                         int(after.sum()), int((~ok).sum())])
            for label, col in [('first', start), ('last', end)]:
                g = (q.sample_date - q[col]).dt.days.dropna()
                gaps.append([name, window + '_' + label, len(g)] +
                            ([round(float(g.quantile(x)), 1) for x in [.1, .25, .5, .75, .9]] if len(g) else ['NA'] * 5))
            g = (q.sample_date - q[end]).dt.days
            bins.append([name, window, int(g.lt(0).sum()), int(g.eq(0).sum()),
                         int(g.between(1, 7).sum()), int(g.between(8, 14).sum()),
                         int(g.between(15, 30).sum()), int(g.gt(30).sum()), int(g.isna().sum())])
    for title, data, cols in [
        ('菌群日期相对窗口（边界同日计入 within）:', rows, ['sample', 'window', 'N', 'before', 'within', 'after', 'unknown']),
        ('间隔天数 = 菌群日期 - 对应端点；负值表示更早:', gaps, ['sample', 'endpoint', 'N', 'P10', 'P25', 'median', 'P75', 'P90']),
        ('相对窗口结束日期的分组人数:', bins, ['sample', 'window', '<0d', '=0d', '1-7d', '8-14d', '15-30d', '>30d', 'NA'])]:
        lines.extend([title, pd.DataFrame(data, columns=cols).to_string(index=False)])
    lines.extend(notes + ['Meals=本轮纳入餐次日期；CGM=对应连接可核对的记录范围。',
        '同日不能判定日内先后；菌群日期语义仍需区分实际采集与交样；仅本轮样本。'])
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    try:
        run = Path('outputs/gut_meal_bridge_v2_20260929T044951_469958Z')
        require(run.is_dir(), '请在 meal_cgm_exploration 目录运行；指定结果文件夹缺失')
        require(json.loads((run / 'manifest.json').read_text())['data_kind'] == 'REAL', 'Expected REAL run')
        report = audit(run)
        dest = Path('outputs') / datetime.now(timezone.utc).strftime('gut_timing_audit_%Y%m%dT%H%M%S_%fZ.txt')
        with dest.open('x', encoding='utf-8') as handle:
            handle.write(report)
        print(report + '汇总已保存: ' + str(dest))
    except Exception as error:
        raise SystemExit('TIMING AUDIT ERROR: ' + str(error))
