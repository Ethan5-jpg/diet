#!/usr/bin/env python3
"""Render archived screenshot precision results; never fit or re-test models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIETS = ['AHEI', 'AMED', 'hPDI', 'rEDIH', 'EAT13', 'NOVA4', 'Carbohydrate_pct']
LABELS = DIETS[:-1] + ['Carbohydrate %']
SETTINGS = ['source_primary', 'joint']
COLORS = ['#21669a', '#ce673b']
OUTCOMES = ['mean_increment', 'peak_rise_0_120', 'positive_mean']
NAMES = ['Net increment', 'Peak rise (0–120 min)', 'Positive increment']
CGM = ['Mean','Median','IQR','GMI','CV','MAD','MAG','MODD','SD_ROC','SDhhmm','SDw','SDwsh','TAR140','TAR180','HBGI','LBGI','ADRR','COGI','GRADE','GRADE_eugly']
STEPS = ['background','plus_energy','plus_carb','plus_both','meal_adjusted']
STEP_LABELS = ['BG','BG + E','BG + C','BG + E + C','Full (+ C²)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(rows, **criteria):
    found = [r for r in rows if all(r.get(k) == v for k,v in criteria.items())]
    assert len(found) == 1, (criteria, len(found))
    r = found[0]
    assert all(r.get(k) is not None and np.isfinite(r[k]) for k in ['estimate','ci_low','ci_high','q'])
    assert r['ci_low'] <= r['estimate'] <= r['ci_high'] and 0 <= r['q'] <= 1
    return r


def mark(ax, r, y, color, marker='o'):
    ax.errorbar(r['estimate'],y,xerr=[[r['estimate']-r['ci_low']],[r['ci_high']-r['estimate']]],
        fmt=marker, color=color, markerfacecolor=color if r['q']<.05 else 'white',
        markeredgecolor=color, ms=5.2, lw=1.25, capsize=2.1, zorder=3)


def style(ax):
    ax.axvline(0, color='#7f8a93', lw=.8, ls='--')
    ax.grid(axis='x', color='#e6e9ed', lw=.6)
    ax.set_axisbelow(True)


def legend(fig):
    handles=[Line2D([],[],color=COLORS[i],marker=['o','s'][i],ls='',label=s) for i,s in enumerate(['Original sample','Joint quality sample'])]
    handles += [Line2D([],[],color='#555',marker='o',ls='',label='Filled: q < 0.05'),Line2D([],[],color='#555',marker='o',mfc='white',ls='',label='Open: q ≥ 0.05')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--records',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    base=Path(args.records).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    files=[base/'nutrition_bridge_累计已展示结果_截图精度.json',base/'quality_timecourse_累计133项已展示结果_截图精度.json']
    sources=[json.loads(p.read_text(encoding='utf-8')) for p in files]
    n,q=[d['rows'] for d in sources]
    assert len(n)==302 and sources[0]['data_kind']=='REAL'
    row_refs={id(r):dict(source_file=files[j].name,row_index=i,run_id=sources[j]['run_id'],row=r) for j,d in enumerate(sources) for i,r in enumerate(d['rows'])}
    provenance=dict(data_kind='REAL',precision='Archived terminal screenshot precision; not full-precision server export',
        inputs={p.name:digest(p) for p in files},script_sha256=digest(__file__),artifacts={},notes=[
            'No raw p values reconstructed; no FDR recomputed; original family q retained.',
            'Pointwise 95% intervals. Original/joint overlap; shared CGM readings.',
            'q displayed as zero is a display/numerical limitation, not a probability of exactly zero.'])
    def save(fig,name,rows,caption):
        for ext in ['png','pdf','svg']:
            fig.savefig(out/(name+'.'+ext),dpi=200,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        provenance['artifacts'][name]=dict(caption=caption,rows=[row_refs[id(r)] for r in rows])
    # Figure 1: complete A models, all seven diets.
    fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True);used=[]
    for k,ax in enumerate(axes):
        for j,s in enumerate(SETTINGS):
            for i,d in enumerate(DIETS):
                r=one(n,analysis='positive' if k==2 else 'adjustment',setting=s,branch='A',outcome=OUTCOMES[k],model='meal_adjusted',diet=d)
                used.append(r);mark(ax,r,i+(-.14 if j==0 else .14),COLORS[j],['o','s'][j])
        ax.set_title(NAMES[k],loc='left',fontweight='bold');style(ax);ax.set_xlabel('β (mg/dL per original diet Z)')
        ax.set_xlim(-1.55,.5);ax.set_yticks(range(7));ax.set_yticklabels(LABELS);ax.invert_yaxis()
    fig.suptitle('1  |  Dietary scores and personal average meal responses',x=.06,ha='left',fontweight='bold',fontsize=15)
    fig.text(.06,.07,'Pointwise 95% CI. Net/peak: q family 70; positive: q family 21, separately within each sample.',fontsize=9)
    fig.subplots_adjust(left=.12,right=.98,top=.83,bottom=.2,wspace=.18);legend(fig)
    save(fig,'Figure1_full_models',used,'七项完整A模型；净/峰q家族70、正增量q家族21；两样本重叠。')
    # Figure 2: independent adjustment specifications, no connecting trajectory.
    for sidx,s in enumerate(SETTINGS):
        fig,axes=plt.subplots(7,2,figsize=(11.6,15.8));used=[]
        for i,d in enumerate(DIETS):
            for k,o in enumerate(OUTCOMES[:2]):
                ax=axes[i,k]
                for j,m in enumerate(STEPS):
                    r=one(n,analysis='adjustment',setting=s,diet=d,outcome=o,model=m);used.append(r);mark(ax,r,j,COLORS[sidx])
                ax.set_yticks(range(5));ax.set_yticklabels(STEP_LABELS,fontsize=8);ax.invert_yaxis();style(ax)
                ax.set_xlim(-1.5,2.85 if k else 1.6)
                ax.set_title(LABELS[i]+'  ·  '+NAMES[k],loc='left',fontsize=10,fontweight='bold')
                if i==6:ax.set_xlabel('β (mg/dL per original diet Z)')
        fig.suptitle('2{}  |  Adjustment dependence · {}'.format('A' if sidx==0 else 'B',['Original sample','Joint quality sample'][sidx]),x=.08,ha='left',fontweight='bold',fontsize=15)
        fig.text(.08,.945,'+E and +C are parallel additions to BG. All five specifications use the same people/meals per diet.',fontsize=9)
        fig.text(.08,.017,'E: mean meal energy; C: mean meal carbohydrate. Full adds C². Pointwise 95% CI; filled q < 0.05 (family 70).',fontsize=9)
        fig.subplots_adjust(left=.13,right=.98,top=.915,bottom=.065,hspace=.65,wspace=.38)
        save(fig,'Figure2'+('A_original' if sidx==0 else 'B_joint'),used,'五规格相互独立展示；+E和+C是平行调整，Full在两者上加碳水平方；每项饮食样本固定，q家族70。')
    # Figure 3: four prespecified phenotypes plus exhaustive 20-item matrix.
    fig,axes=plt.subplots(2,2,figsize=(11,7));used=[]
    for ax,c in zip(axes.flat,['Mean','CV','MODD','SDw']):
        for j,s in enumerate(SETTINGS):
            for k,meal in enumerate(OUTCOMES):
                r=one(n,analysis='bridge',setting=s,cgm=c,meal_measure=meal);used.append(r);mark(ax,r,k+(-.13 if j==0 else .13),COLORS[j],['o','s'][j])
        ax.set_title(c,loc='left',fontweight='bold');ax.set_yticks(range(3));ax.set_yticklabels(['Net','Peak','Positive']);ax.invert_yaxis();style(ax);ax.set_xlim(0,.82)
        ax.set_xlabel('β (original whole-CGM Z / 1 SD meal feature)')
    fig.suptitle('3A  |  Meal features and whole-period CGM',x=.08,ha='left',fontweight='bold',fontsize=15)
    fig.text(.08,.068,'Shared readings; internal association. SDw = mean within-day SD. Pointwise CI; q family 60.',fontsize=9)
    fig.subplots_adjust(left=.1,right=.98,top=.88,bottom=.20,wspace=.35,hspace=.55);legend(fig)
    save(fig,'Figure3A_key_CGM',used,'原预定四指标的24项衔接；X为配对样本餐特征1SD，Y为整体CGM原Z。共享读数，不是独立验证。')
    fig,axes=plt.subplots(1,2,figsize=(10,10.8));used=[]
    for j,(ax,s) in enumerate(zip(axes,SETTINGS)):
        mat=np.zeros((20,3));rs=[]
        for i,c in enumerate(CGM):
            for k,meal in enumerate(OUTCOMES):
                r=one(n,analysis='bridge',setting=s,cgm=c,meal_measure=meal);used.append(r);rs.append((i,k,r));mat[i,k]=r['estimate']
        im=ax.imshow(mat,cmap='RdBu_r',vmin=-.8,vmax=.8,aspect='auto')
        for i,k,r in rs:ax.text(k,i,'{:.3f}{}'.format(r['estimate'],'*' if r['q']<.05 else ''),ha='center',va='center',fontsize=9,color='white' if abs(r['estimate'])>.5 else '#142333')
        ax.set_yticks(range(20));ax.set_yticklabels([('* ' if c in ['Mean','CV','MODD','SDw'] else '')+c for c in CGM]);ax.set_xticks(range(3));ax.set_xticklabels(['Net','Peak','Positive']);ax.set_title(['Original sample','Joint quality sample'][j],loc='left',fontweight='bold')
    fig.suptitle('3B  |  All 20 CGM phenotypes · all 120 associations',x=.08,ha='left',fontweight='bold',fontsize=15)
    fig.text(.08,.045,'Cell *: q < 0.05 (family 60). Row *: prespecified key phenotype. Mean and GMI are redundant.',fontsize=9)
    fig.text(.08,.022,'β: original CGM Z / paired-sample meal SD. Shared readings; coefficients are not correlations or independent replications.',fontsize=8.5)
    fig.subplots_adjust(left=.18,right=.91,top=.9,bottom=.1,wspace=.65)
    cax=fig.add_axes([.94,.18,.018,.61]);fig.colorbar(im,cax=cax,label='β')
    save(fig,'Figure3B_all20_CGM',used,'完整120项，数值为标准化回归系数而非相关系数；原始q=0保留在追溯JSON中，图中只标阈值；Mean/GMI冗余。')
    # S1: direct AMED contrast alongside its two components.
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True);used=[]
    for col,(ax,o,title) in enumerate(zip(axes,['early_mean','late_mean','late_minus_early'],['Early (0–60 min)','Late (60–120 min)','Direct late − early contrast'])):
        for j,s in enumerate(SETTINGS):
            r=one(q,analysis='timecourse',setting=s,diet='AMED',outcome=o);used.append(r);mark(ax,r,j,COLORS[j],['o','s'][j])
            fig.text(.075+col*.31,.10 if j else .15,'{}: {:.3f} [{:.3f}, {:.3f}]; q={:.4g}'.format('Joint' if j else 'Original',r['estimate'],r['ci_low'],r['ci_high'],r['q']),fontsize=8)
        style(ax);ax.set_ylim(1.6,-.6);ax.set_xlim(-.52,.42);ax.set_yticks([0,1]);ax.set_yticklabels(['Original','Joint']);ax.set_title(title,fontsize=10,fontweight='bold',loc='left');ax.set_xlabel('β (mg/dL per original AMED Z)')
    fig.suptitle('S1  |  AMED time distribution · existing models',x=.065,ha='left',fontweight='bold',fontsize=15)
    fig.text(.065,.02,'Direct contrast, not a comparison of two significance labels. Original timecourse q family 21. Original/joint overlap.',fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,top=.78,bottom=.38,wspace=.25)
    save(fig,'Supplement1_AMED_timecourse',used,'使用原quality_timecourse模型的AMED早/晚和晚减早直接对比；未新拟合四时间段，q家族21。')
    # Exhaustive numerical tables, with row-level provenance.
    lines=['# 七项饮食证据总表','', '数据：真实研究结果的截图精度。未重拟合、未重算FDR；95%CI为逐项区间。原/joint人群重叠。', '',
        'A为个人平均餐反应；B为评分×餐碳水交互；时间项为晚减早直接对比。负B系数不代表碳水总斜率为负。', '']
    table_refs=[]
    for s in SETTINGS:
        lines += ['## '+('原样本' if s=='source_primary' else '联合质量样本'),'']
        for branch in ['A','B','timecourse']:
            outs=OUTCOMES if branch!='timecourse' else ['late_minus_early']
            lines += ['### '+{'A':'A：完整调整，单位mg/dL/原饮食Z；净/峰q家族70、正增量q家族21','B':'B：碳水交互，单位mg/dL/10g/原饮食Z；净/峰各沿用原14项家族、正增量21项','timecourse':'时间：晚段减早段，单位mg/dL/原饮食Z；q家族21'}[branch],'','| 饮食 | 结局 | N | β [95%CI] | q | q<0.05 |','|---|---|---:|---|---:|:---:|']
            for d in DIETS:
                for o in outs:
                    if branch=='timecourse':r=one(q,analysis='timecourse',setting=s,diet=d,outcome=o)
                    elif branch=='A':r=one(n,analysis='positive' if o=='positive_mean' else 'adjustment',setting=s,branch='A',model='meal_adjusted',diet=d,outcome=o)
                    elif o=='positive_mean':r=one(n,analysis='positive',setting=s,branch='B',diet=d,outcome=o)
                    else:r=one(q,analysis='quality',setting=s,branch='B',diet=d,outcome=o)
                    table_refs.append(row_refs[id(r)])
                    ds=r.get('display',{})
                    vals=[ds.get(k,format(r[k],'.6g')) for k in ['estimate','ci_low','ci_high','q']]
                    lines.append('| {} | {} | {} | {} [{}, {}] | {} | {} |'.format(d,o,r['people'],*vals,'是' if r['q']<.05 else '否'))
            lines.append('')
    lines += ['## 可用于汇报的定位','',
        '- rEDIH：三类完整A反应均负向，原/joint一致；是当前最稳定的饮食信号，但依赖调整语境，不能写成因果保护。',
        '- 个人碳水比例：背景正向、完整调整后负向；评分剩余变异约23%，优先明确条件比较的含义。',
        '- NOVA4：背景平均反应正向，营养调整后减弱；B交互仍是另一类问题，不与A混为一谈。',
        '- EAT13：正增量完整模型原/joint获得支持，净/峰未获相同支持；结局间差异未作正式检验。',
        '- AHEI：部分结果处于阈值附近。joint峰值CI未跨零但q未通过；原正增量q约0.05155、joint约0.038，不能宣称两样本效应有差异。',
        '- hPDI：部分背景结果有信号，完整调整支持不足；不能把背景与完整模型差异直接称为中介。',
        '- AMED：平均反应支持不足，但原晚减早直接对比负向；可保留时间分布线索，不解释为胰岛素机制。','',
        '完整出处与逐行字段见 `artifact_provenance.json`。字段中的旧截图文件名由原累计JSON保留；行索引可直接定位冻结副本。']
    (out/'七项饮食证据总表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    provenance['artifacts']['七项饮食证据总表']=dict(rows=table_refs)
    (out/'artifact_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    intro=['# 本地结果图表','', '已完成三组主图、七项完整总表和AMED补充图。数据来自已归档截图；未新增真实模型。','',
        '图1、图2使用nutrition_bridge的q家族，旧B与时间模型沿用quality_timecourse的原家族。原/joint不是独立复现。', '',
        '[七项饮食证据总表](七项饮食证据总表.md) · [逐行追溯](artifact_provenance.json)', '']
    for name,meta in provenance['artifacts'].items():
        if name=='七项饮食证据总表':continue
        intro+=['## '+name,'',meta['caption'],'','![{}]({}.png)'.format(name,name),'','[矢量PDF]({}.pdf) · [SVG]({}.svg)'.format(name,name),'']
    (out/'README.md').write_text('\n'.join(intro),encoding='utf-8')
    print('LOCAL_FIGURES_COMPLETE: {} figures, {} table rows; {}'.format(6,len(table_refs),out))

if __name__=='__main__':main()
