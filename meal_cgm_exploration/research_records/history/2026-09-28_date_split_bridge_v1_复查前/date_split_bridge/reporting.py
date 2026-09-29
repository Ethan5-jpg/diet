"""Exactly three key summaries, hash-verified before display."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hppmeal.common import require,sha256
from response_followup.support import SETTINGS,OUTCOMES
from .metrics import METRICS
from . import VERSION

PRIVATE = ['person_features','daily_grid_audit','first_members','read_separation','model_members']


def save(run,result):
    for name,table in result.items(): run.table(table,name+'.csv',private=name in PRIVATE)


def one(d,**criteria):
    for name,value in criteria.items(): d=d.loc[d[name].eq(value)]
    require(len(d)==1,'Missing/duplicate result row')
    return d.iloc[0]


def text_table(d):
    return d.to_string(index=False,float_format=lambda x:'{:.3g}'.format(x),na_rep='NA')


def cell(r):
    if r.status!='ESTIMATED': return 'NE'
    return '{:.3g}[{:.3g},{:.3g}]'.format(r.estimate,r.ci_low,r.ci_high)


def qtext(r):
    return 'NA' if r.status!='ESTIMATED' else ('0(display)' if r.q==0 else '{:.3g}'.format(r.q))


def pages(run,result,state):
    models=result['models'];flow=result['flow'];p=result['person_features']
    require(len(models)==72 and len(result['prior_count_checks'])==14,'Incomplete summaries')
    quality=[]
    for metric in METRICS:
        a=one(flow,setting='source_primary',cgm=metric);b=one(flow,setting='joint',cgm=metric)
        cor=one(result['legacy_concordance'],setting='source_primary',cgm=metric)
        quality.append(dict(metric=metric,N_original=int(a.metric_ready_people),N_joint=int(b.metric_ready_people),
            late_days_p50=a.late_days_p50,legacy_rho=cor.spearman))
    counts=[]
    for setting in SETTINGS:
        d=p.loc[p.setting.eq(setting)]
        counts.append(dict(sample='Original' if setting=='source_primary' else 'Joint',initial=len(d),prior_basic=int(d.candidate_basic.sum()),
            crossing_meals=int(d.excluded_meals_crossing_cut.sum()),timezone_meals=int(d.excluded_first_meals_timezone.sum()),
            DST_Y_days=int(d.dst_second_days.sum())))
    bodies=[('状态与指标核对',
        '旧桥接成员/日期/前窗餐已核对；14项可行性人数与上轮一致。\n'
        '主分析完成 {}/24；全期对照 {}/24；配对差值 {}/24。\n'.format(*[int(models.loc[models.target.eq(k),'status'].eq('ESTIMATED').sum()) for k in ['late_primary','full_comparator','late_minus_full']])+
        text_table(pd.DataFrame(counts))+'\n'+text_table(pd.DataFrame(quality))+'\n'
        '指标采用新固定窗口规则；legacy_rho仅为新全期指标与旧全期Z的相关。\n'
        'Mean/CV/SDw:完整日原始覆盖>=90%，日内15min网格，插值<=45min；需>=3日。\n'
        'MODD15:相邻非DST日同钟点绝对差，>=48配对格/日对，至少2对；不是原iglu精确复现。\n'
        '前窗餐实际读数<切分点；后窗网格实际支撑读数>=切分点，未检出共享读数。\n'
        '原/joint共24主规格合并BH；全期/差值各单独24项次要家族。NE保留在家族。\n'
        '全期对照含后窗，也可能含前窗读数；差值并非纯粹共享读数造成的效应。\n'
        '样本量指指标就绪人数；具体模型人数见后两份。完整明细留服务器。')]
    for setting,title in zip(SETTINGS,['原样本12项','joint12项']):
        lines=[]
        for measure,short in zip(OUTCOMES,['Net','Peak','Positive']):
            for metric in METRICS:
                a=one(models,setting=setting,meal_measure=measure,cgm=metric,target='late_primary')
                b=one(models,setting=setting,meal_measure=measure,cgm=metric,target='full_comparator')
                c=one(models,setting=setting,meal_measure=measure,cgm=metric,target='late_minus_full')
                lines.append(dict(X=short,Y=metric,N=int(a.people),late_B_CI=cell(a),q24=qtext(a),
                    full_B='{:.3g}'.format(b.estimate) if b.status=='ESTIMATED' else 'NE',delta_CI=cell(c),qD24=qtext(c)))
        reasons=models.loc[models.setting.eq(setting)&models.status.ne('ESTIMATED'),'reason'].value_counts()
        reason_text='；'.join('{} ({})'.format(k,int(v)) for k,v in reasons.head(3).items())
        bodies.append((title,'X=前窗个人餐反应；Y=后窗CGM。单位：新全期参考Y的SD/前窗X参考SD。\n'
            '同一批人、同一X/背景协变量/尺度；CI为逐项95%，q24为全部24主规格BH。\n'+text_table(pd.DataFrame(lines))+'\n'
            'full_B=同一X关联全期新指标；delta=后窗B−全期B，配对模型CI，qD24为次要24家族。\n'
            '两样本重叠；不是独立验证/因果/预测测试，不能与旧原Z系数直接相减。\n'
            +('NE原因：'+reason_text+'。其余详见tables/models.csv。' if len(reasons) else '本套预定模型均可估计。')))
    folder=run.out/'pages';folder.mkdir()
    for i,(title,body) in enumerate(bodies,1):
        content='KEY {}/3 | {}\nRUN={}\nDATA={} | STATUS={}\n\n{}\n'.format(i,title,run.out.name,run.c['data_kind'],state['status'],body)
        require(len(content.splitlines())<=32,'Summary exceeds screenshot height')
        (folder/('{:02d}.txt'.format(i))).write_text(content,encoding='utf-8')


def summary(run,state):
    (run.out/'SUMMARY.md').write_text('# 日期分离CGM衔接\n\nDATA={}；STATUS={}。\n\n{}\n\n'
        '新固定窗口定义；旧iglu导出精确等价尚未确认。24主规格及配套同人全期对照/配对差值。\n'
        '只回传pages/中的3份关键摘要；完整表和private/留服务器。\n'.format(run.c['data_kind'],state['status'],state.get('reason','')),encoding='utf-8')


def show(source,output_root,page=None,brief=False):
    require(not (brief and page is not None),'Use --brief or --page, not both')
    if source=='latest':
        candidates=sorted(p for p in Path(output_root).glob('date_split_bridge_*') if p.is_dir())
        require(candidates,'No date-split run found');source=candidates[-1]
    source=Path(source).expanduser().resolve()
    require((source/'manifest.json').is_file(),'Latest/selected run has no completed manifest; inspect ERROR.txt')
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'));state=m['status'].get('date_split_bridge',{})
    require(state.get('version')==VERSION and state.get('status') in ['COMPLETE','COMPLETE_WITH_LIMITATIONS'],'Latest/selected run is not complete; no older run substituted')
    chunks=[]
    for i in ([1,2,3] if brief else [page or 1]):
        require(i in [1,2,3],'Only3 key summaries')
        rel='pages/{:02d}.txt'.format(i);p=source/rel
        require(p.is_file() and state.get('result_sha256',{}).get(rel)==sha256(p),'Summary changed/missing: '+rel)
        chunks.append(p.read_text(encoding='utf-8'))
    print('\n'.join(chunks),end='')
