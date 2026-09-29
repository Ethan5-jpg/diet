"""Aggregate-only short pages, figures, and verified readback."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hppmeal.common import require,sha256
from .support import SETTINGS,OUTCOMES


def formatted(d):
    return d.to_string(index=False,float_format=lambda v:'{:.4g}'.format(v),na_rep='NA')


def save(run,result):
    for key,t in result.items():
        run.table(t,key+'.csv',private=key in ['person_windows','first_window_members'])


BRIEF_TABLES = ['support','contrasts','membership','feasibility','window_distributions']


def one_row(table, **criteria):
    selected=table
    for col,value in criteria.items():selected=selected.loc[selected[col].eq(value)]
    require(len(selected)==1,'Missing/duplicate summary row: '+str(criteria))
    return selected.iloc[0]


def number_text(value,digits=3):
    if pd.isna(value):return 'NA'
    return format(float(value),'.{}g'.format(digits))


def reason_code(row):
    if row['support_status']=='SUPPORTED':return 'OK'
    text=str(row.get('reason',''))
    rules=[('Empty frozen','empty'),('Score IQR','Z_ties'),('No overlap of group central meal energy','E_disjoint'),
        ('Too few people/meals','E_sparse'),('No overlap of central carbohydrate','C_disjoint'),
        ('No observed meals','empty_range'),('Carbohydrate contrast','C_narrow'),
        ('Fewer than 50 people','cell_N<50'),('Fewer than 30 within','cross_N<30')]
    return next((code for prefix,code in rules if text.startswith(prefix)),'NO')


def effect_cell(row):
    q='0(display)' if row['original_q']==0 else number_text(row['original_q'],4)
    if row['status']!='SUPPORTED_CONTRAST':return 'NA; q='+q
    require(all(np.isfinite(row[x]) for x in ['estimate','ci_low','ci_high','original_q']), 'Invalid summary estimate')
    return '{}[{},{}];q={}'.format(number_text(row['estimate']),number_text(row['ci_low']),number_text(row['ci_high']),q)


def endpoint_minimum(row,fields):
    # Early rejection means these checks were not reached. Also masks old v1 zero placeholders.
    if reason_code(row) not in ['OK','cell_N<50','cross_N<30']:return 'NA'
    values=[row[k] for k in fields]
    require(all(np.isfinite(v) and v>=0 and float(v).is_integer() for v in values),'Invalid endpoint support count')
    return str(int(min(values)))


def brief_sections(run_name,data_kind,result,state,legacy=False):
    from seven_diet_meals.inputs import DIETS
    require(data_kind in ['REAL','SYNTHETIC'],'Unknown data label')
    require(len(result['support'])==14 and len(result['contrasts'])==42 and len(result['membership'])==14,'Incomplete result tables for concise summary')
    require(result['membership'].status.eq('MATCHED').all(),'Membership checks were not all matched')
    supported=int(result['contrasts'].status.eq('SUPPORTED_CONTRAST').sum())
    require(supported==state['supported_contrasts'],'Run/table supported contrast counts disagree')
    f=result['feasibility'];w=result['window_distributions']
    rows=[]
    stages=[('original_bridge_members','Original_members'),('first_window_5meals_3days','X_5meals_3days'),
        ('second_window_3full_days_coverage90','Y_3days_cov90'),('both_basic_requirements','X_and_Y'),
        ('both_and_positive_complete','Plus_positive'),('both_and_MODD_clock_proxy','Plus_MODD_proxy'),
        ('both_MODD_proxy_and_positive','MODD_and_positive')]
    for key,label in stages:
        row={'stage':label}
        for setting,short in zip(SETTINGS,['Original','Joint']):row[short]=int(one_row(f,setting=setting,stage=key)['people'])
        rows.append(row)
    medians=[];exclusions=[]
    for setting,short in zip(SETTINGS,['Original','Joint']):
        base=one_row(f,setting=setting,stage='original_bridge_members')['people']
        kept=one_row(f,setting=setting,stage='both_basic_requirements')['people']
        row=dict(sample=short,kept_pct=100*kept/base if base else np.nan)
        for field,label in [('first_qualified_meals','X_meals_p50'),('first_meal_dates','X_days_p50'),('second_good90_days','Y90_days_p50'),('adjacent_day_pairs_48bins','pairs_p50')]:
            row[label]=one_row(w,setting=setting,field=field)['median']
        medians.append(row)
        exclusions.append(dict(sample=short,
            crossing_meals=int(one_row(w,setting=setting,field='excluded_meals_crossing_cut')['total']),
            timezone_meals=int(one_row(w,setting=setting,field='excluded_first_meals_timezone')['total']),
            DST_Y_days=int(one_row(w,setting=setting,field='dst_second_days')['total'])))
    bodies=[('状态与日期分离',
        '冻结B样本成员核对14/14；有观测支持、可换算的对比 {}/42。'.format(supported)+'\n'
        '只转换已有交互系数；本次新增效应模型=0。完整明细保留服务器。\n'
        '日期分离：X为前窗餐；Y为后窗同连接CGM。人数如下：\n'+formatted(pd.DataFrame(rows))+'\n'
        '门槛：X>=5餐/3日；Y>=3个完整且覆盖>=90%的日期。以下中位数按原桥接成员：\n'+formatted(pd.DataFrame(medians))+'\n'
        '排除/时区诊断（计数）：\n'+formatted(pd.DataFrame(exclusions))+'\n'
        'MODD_proxy: >=2个相邻日对、每对>=48共同钟点格；这不是原MODD重算。\n'
        '计数仅用于可行性判断。后续CGM窗口定义/缩放仍需确认，无新β/p/q。')]
    for setting,title in zip(SETTINGS,['原样本关键结果','joint关键结果']):
        anchors=[];effects=[]
        for diet in DIETS:
            r=one_row(result['support'],setting=setting,diet=diet)
            short='Carb_pct' if diet=='Carbohydrate_pct' else diet
            anchors.append(dict(diet=short,N=int(r['people']),dZ=number_text(r['score_high']-r['score_low']),
                C_g=number_text(r['carb_low'])+'->'+number_text(r['carb_high']),
                E_kcal=number_text(r['energy_low'],4)+'..'+number_text(r['energy_high'],4),
                cellN_min=endpoint_minimum(r,['low_score_lowC_people','low_score_highC_people','high_score_lowC_people','high_score_highC_people']),
                crossN_min=endpoint_minimum(r,['low_score_crossers','high_score_crossers']),support=reason_code(r)))
            row=dict(diet=short)
            for outcome,label in zip(OUTCOMES,['Net_D[CI];q','Peak_D[CI];q','Positive_D[CI];q']):
                row[label]=effect_cell(one_row(result['contrasts'],setting=setting,diet=diet,outcome=outcome))
            effects.append(row)
        bodies.append((title,
            'D=[高评分的高C−低C反应]−[低评分的高C−低C反应]，单位mg/dL。\n'
            'dZ=原评分Q75−Q25；C为碳水参照->高值(g)，E为共同能量范围。\n'+formatted(pd.DataFrame(anchors))+'\n'
            '每项为 D[逐项95%CI];原q。NA表示不换算，原q仍列出；Carb_pct=Carbohydrate_pct。\n'+formatted(pd.DataFrame(effects))+'\n'
            'OK需每端点单元>=50人、每组同人跨端点>=30人；NA人数表示未进入该检查。\n'
            '净/峰各沿用原14项q家族，正增量沿用21项；CI条件于观测参照。\n'
            '负D不是碳水总斜率为负；不同评分/样本的参照不同，不能直接排名。'))
    result_text=[]
    for i,(title,body) in enumerate(bodies,1):
        prefix='KEY {}/3 | {}\nRUN={}\nDATA={} | STATUS={}\n'.format(i,title,run_name,data_kind,state['status'])
        if legacy:prefix+='旧版结果只读压缩显示，未重算；旧版能量零宽度判定未修正。\n'
        result_text.append(prefix+'\n'+body+'\n')
    return result_text


def pages(run,result,state):
    folder=run.out/'pages';folder.mkdir()
    for i,content in enumerate(brief_sections(run.out.name,run.c['data_kind'],result,state),1):
        (folder/('{:02d}.txt'.format(i))).write_text(content,encoding='utf-8')


def figures(run,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from seven_diet_meals.inputs import DIETS
    for setting in SETTINGS:
        fig,axes=plt.subplots(7,3,figsize=(13,16))
        for i,diet in enumerate(DIETS):
            for j,outcome in enumerate(OUTCOMES):
                ax=axes[i,j]
                d=result['curves'].loc[lambda x:x.setting.eq(setting)&x.diet.eq(diet)&x.outcome.eq(outcome)]
                if len(d):
                    x=d.carb_g.to_numpy(float);lo=d.ci_low.to_numpy(float);hi=d.ci_high.to_numpy(float)
                    ax.plot(x,d.estimate.to_numpy(float),color='#21669a');ax.fill_between(x,lo,hi,color='#21669a',alpha=.16)
                else:ax.text(.5,.5,'No supported contrast',ha='center',va='center',transform=ax.transAxes,fontsize=8)
                ax.axhline(0,c='gray',ls='--',lw=.7);ax.set_title(diet+' | '+['Net','Peak','Positive'][j],fontsize=9,loc='left')
                if i==6:ax.set_xlabel('Meal carbohydrate (g)')
                if j==0:ax.set_ylabel('Relative change difference\n(mg/dL)',fontsize=8)
                ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        fig.suptitle('{} | {} | Frozen-interaction transformation'.format(run.c['data_kind'],setting),fontsize=14)
        fig.text(.07,.012,'Pointwise CI conditional on observed anchors. Not absolute glucose curves; original q retained. Unsupported grid cells are omitted.',fontsize=8)
        fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.055,hspace=.8,wspace=.35)
        for ext in ['png','pdf']:fig.savefig(run.out/'figures'/('relative_changes_'+setting+'.'+ext),dpi=170)
        plt.close(fig)


def summary(run,state):
    (run.out/'SUMMARY.md').write_text('# 后续：交互量级与日期分离可行性\n\nDATA={}；STATUS={}。\n\n{}\n\n'
        '仅转换原B系数并检查日期分离；没有新拟合效应，没有重算q。\n'
        '原CGM指标采用既有iglu导出及清洗结果；本包未确认窗口重算契约，MODD仅输出相邻日钟点格代理计数。\n'
        '只需pages/的3份关键摘要；完整明细和private/保留服务器。\n'.format(run.c['data_kind'],state['status'],state.get('reason','')),encoding='utf-8')


def resolve_run(source,output_root):
    if source=='latest':
        # Never silently hide a newer failed/incomplete run by selecting an older successful run.
        candidates=sorted(p for p in Path(output_root).glob('response_followup_*') if p.is_dir())
        require(candidates,'No follow-up run found')
        source=candidates[-1]
    source=Path(source).expanduser().resolve()
    require((source/'manifest.json').is_file(),'Selected/latest run has no completed manifest; inspect its ERROR.txt or specify an earlier run explicitly')
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    state=m['status'].get('response_followup',{})
    require(state.get('status') in ['COMPLETE','COMPLETE_WITH_LIMITATIONS'],'Selected/latest run is not complete; no older run was substituted')
    return source,m,state


def show(source,output_root,page=None,brief=False):
    source,m,state=resolve_run(source,output_root)
    require(page is None or 1<=page<=3,'Only key summaries 1..3 are available; details stay in tables/')
    require(not (brief and page is not None),'Use --brief or --page, not both')
    selected=[1,2,3] if brief else [page or 1]
    version=state.get('version')
    require(version in ['2026.09.27.1','2026.09.27.2'],'Unsupported follow-up result version')
    chunks=[]
    if version=='2026.09.27.1':
        # Backward-compatible display from immutable aggregate tables; no old run is rewritten.
        tables={}
        for key in BRIEF_TABLES:
            rel='tables/'+key+'.csv';p=source/rel
            require(p.is_file() and state.get('result_sha256',{}).get(rel)==sha256(p),'Legacy summary source changed/missing: '+rel)
            tables[key]=pd.read_csv(p)
        rendered=brief_sections(source.name,m['data_kind'],tables,state,legacy=True)
        chunks=[rendered[i-1] for i in selected]
    else:
        for i in selected:
            rel='pages/{:02d}.txt'.format(i);p=source/rel
            require(p.is_file() and state.get('result_sha256',{}).get(rel)==sha256(p),'Summary page changed/missing')
            chunks.append(p.read_text(encoding='utf-8'))
    # Verify every requested part before printing any of them.
    print('\n'.join(chunks),end='')
