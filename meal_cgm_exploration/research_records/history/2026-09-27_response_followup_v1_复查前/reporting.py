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


def pages(run,result,state):
    sections=[]
    sections.append(('运行与目录', (
        '冻结成员核对：14/14；原B规格：42项（只做线性换算，不重拟合）。\n'
        '有观测支持的差异：{}/42；其余保留原因。\n'
        '日期分离仅检查可行性，不计算新CGM指标或效应。\n'
        '2-3: 原/joint评分、餐碳水和能量参照范围\n4-5: 原/joint端点人数及同人跨端点支持\n'
        '6-8: 原样本净/峰/正差异；9-11: joint对应差异\n'
        '12: 日期分离筛选人数；13-14: 两窗分布\n15-16: 24项拟议衔接可用人数\n'
        '17-18: 保留/排除者比较；19-20: 原B样本分布；21: 支持不足原因\n'
        '差异=原交互β×评分IQR×餐碳水差/10，单位mg/dL。\n'
        'CI为固定观测参照下逐项95%，不含参照分位数估计不确定性。\n'
        'q沿用原家族；负值不表示碳水总斜率为负。').format(state['supported_contrasts'])))
    for setting in SETTINGS:
        s=result['support'].loc[lambda x:x.setting.eq(setting)]
        view=s[['diet','people','meals','score_low','score_high','carb_low','carb_high','energy_low','energy_high']].rename(columns={'score_low':'Z25','score_high':'Z75','carb_low':'C_ref','carb_high':'C_high','energy_low':'E_low','energy_high':'E_high'})
        sections.append(('交互观测范围 | '+setting,'Z为原评分，C单位g，E单位kcal。端点由共同能量带中的观测决定。\n'+formatted(view)))
    for setting in SETTINGS:
        s=result['support'].loc[lambda x:x.setting.eq(setting)]
        view=s[['diet','low_score_lowC_people','low_score_highC_people','high_score_lowC_people','high_score_highC_people','low_score_crossers','high_score_crossers','support_status']].rename(columns={
            'low_score_lowC_people':'L_lowC','low_score_highC_people':'L_highC','high_score_lowC_people':'H_lowC','high_score_highC_people':'H_highC',
            'low_score_crossers':'L_both','high_score_crossers':'H_both','support_status':'support'})
        sections.append(('端点支持 | '+setting,'L/H=评分<=Q25/>=Q75；端点附近±5g，截于共同范围；均限制在共同能量带。\n每格>=50人且每评分组同人跨两端点>=30人，才换算差异。\n'+formatted(view)))
    for setting in SETTINGS:
        for outcome in OUTCOMES:
            t=result['contrasts'].loc[lambda x:x.setting.eq(setting)&x.outcome.eq(outcome)]
            sections.append(('相对变化差 | {} | {}'.format(setting,outcome),
                'D=[高评分的高C−低C反应]−[低评分的高C−低C反应]。\n固定其它模型协变量的线性对比；非绝对曲线；仅作观测支持初筛。\n'+
                formatted(t[['diet','original_beta','score_delta','estimate','ci_low','ci_high','original_q','status']].rename(columns={'original_beta':'beta_B','score_delta':'delta_Z','estimate':'D_mgdl','original_q':'old_q'}))))
    sections.append(('日期分离筛选人数',
        '同原CGM连接；按连接首末本地日期中点切分。先餐反应，后CGM。\n'
        'X至少5餐/3日；Y至少3个完整且覆盖>=90%的日期，均为诊断阈值。\n'
        'MODD proxy另需>=2个相邻日对，每对>=48个共同15分钟钟点格；不是原MODD算法。\n'+formatted(result['feasibility'])))
    for setting in SETTINGS:
        t=result['window_distributions'].loc[lambda x:x.setting.eq(setting)].drop(columns='setting')
        sections.append(('窗口分布 | '+setting,formatted(t)))
    for setting in SETTINGS:
        t=result['specifications'].loc[lambda x:x.setting.eq(setting)]
        sections.append(('候选可用人数 | '+setting,'仅可行性计数；原CGM定义、清洗与窗口重算需下一阶段确认。尚无新β/p/q。\n'+formatted(t[['meal','cgm','potential_people']])))
    for setting in SETTINGS:
        t=result['selection'].loc[lambda x:x.setting.eq(setting)&x.field.str.startswith('original_')]
        sections.append(('日期分离保留者比较 | '+setting,'分组按两窗基本门槛；数值为原全观测期特征，不是新窗反应。\n'+formatted(t[['group','field','people','valid','mean','sd']])))
    for setting in SETTINGS:
        t=result['descriptive'].loc[lambda x:x.setting.eq(setting)]
        sections.append(('原B样本分布 | '+setting,'评分每人等权；餐营养分位数为每人总权重1的餐分布。\n'+formatted(t[['diet','field','p10','p25','median','p75','p90']])))
    t=result['support'][['setting','diet','support_status','reason']].copy()
    t['reason']=t.reason.where(t.support_status.ne('SUPPORTED'),'Passed screen; full covariate overlap not established')
    sections.append(('支持状态与原因',formatted(t)))
    require(len(sections)==21,'Update fixed page directory')
    folder=run.out/'pages';folder.mkdir()
    for i,(title,body) in enumerate(sections,1):
        (folder/('{:02d}.txt'.format(i))).write_text('PAGE {}/21 | {}\nRUN={}\nDATA={} | FOLLOWUP={}\n\n{}\n'.format(i,title,run.out.name,run.c['data_kind'],state['status'],body),encoding='utf-8')


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
        '结果以pages/的21页汇总回传；private/保留服务器，包含原标识和窗口日期。\n'.format(run.c['data_kind'],state['status'],state.get('reason','')),encoding='utf-8')


def show(source,output_root,page):
    if source=='latest':
        candidates=[]
        for p in Path(output_root).glob('response_followup_*'):
            if not (p/'manifest.json').is_file():continue
            m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
            if m['status'].get('response_followup',{}).get('status') in ['COMPLETE','COMPLETE_WITH_LIMITATIONS']:candidates.append(p)
        require(candidates,'No completed follow-up run')
        source=sorted(candidates)[-1]
    source=Path(source).expanduser().resolve()
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    state=m['status'].get('response_followup',{})
    require(state.get('status') in ['COMPLETE','COMPLETE_WITH_LIMITATIONS'],'Follow-up run is not complete')
    require(1<=page<=21,'Page must be 1..21')
    rel='pages/{:02d}.txt'.format(page);p=source/rel
    require(state.get('result_sha256',{}).get(rel)==sha256(p),'Summary page changed/missing')
    print(p.read_text(encoding='utf-8'),end='')
