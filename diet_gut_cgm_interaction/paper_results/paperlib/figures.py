"""Editable vector figures and exact plotted-source tables."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from .data import KEY, MODELS, CODES, OUTCOMES, TAXA, SPECIES, DIETS, SHORT, UNITS, BETA, LOW, HIGH, SLOPE_FIELDS, public, true

COLORS = ['#20252b', '#0072b2', '#999999', '#009e73', '#e69f00', '#cc79a7', '#d55e00']


def setup():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7, 'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6,
        'ytick.labelsize':6, 'legend.fontsize':6, 'axes.linewidth':.6,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'savefig.facecolor':'white', 'axes.spines.top':False, 'axes.spines.right':False})


def caption_mode(meta):
    if meta['mode']=='SCREENSHOT_PREVIEW':
        return 'PREVIEW | Screenshot-rounded values; replace with verified server export before submission.'
    if meta['mode']=='SYNTHETIC_TEST':
        return 'SYNTHETIC SOFTWARE TEST | Not HPP research results.'
    return 'Observational estimates | Pointwise 95% confidence intervals'


def finish(fig, name, frame, out, bundle, figures, footer):
    fig.text(.02,.015,caption_mode(bundle['meta']),fontsize=5.5,ha='left',color='black')
    fig.text(.02,.055,footer,fontsize=5.7,ha='left',va='bottom',color='black')
    fig.canvas.draw()
    # Audit text bounds before export; figure dimensions remain fixed (no tight cropping).
    renderer=fig.canvas.get_renderer()
    outside=[]
    # Matplotlib creates locator labels beyond the visible limits but does not draw them.
    hidden_ticks=set()
    for axis in fig.axes:
        for coordinate in [axis.xaxis,axis.yaxis]:
            lo,hi=sorted(coordinate.get_view_interval())
            for tick in coordinate.get_major_ticks()+coordinate.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    hidden_ticks.update([id(tick.label1),id(tick.label2)])
    for artist in fig.findobj(matplotlib.text.Text):
        if id(artist) in hidden_ticks or not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2):
            outside.append(artist.get_text())
    if outside:
        raise ValueError('Text outside figure '+name+': '+repr(outside[:5]))
    for ext in ['pdf','svg','png']:
        fig.savefig(out/'figures'/(name+'.'+ext),dpi=300)
    source=frame.copy()
    source['evidence_precision']=bundle['meta']['mode']
    source.to_csv(out/'source_data'/(name+'.csv'),index=False,encoding='utf-8-sig')
    figures.append(dict(name=name,rows=len(source),width_mm=round(fig.get_figwidth()*25.4,2),height_mm=round(fig.get_figheight()*25.4,2),text_bounds='passed'))
    plt.close(fig)


def point(ax, x, lo, hi, y, color='#0072b2', filled=True, size=3):
    ax.errorbar(x,y,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',mfc=color if filled else 'white',
                mec=color,ecolor=color,ms=size,lw=.7,capsize=1.5,mew=.65)


def make_figures(bundle,out):
    setup(); figures=[]
    primary=bundle['primary']
    focus=primary[primary.family.eq('focus')].copy()
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,105/25.4),sharey=True)
    combinations=[(e,t) for e in ['hPDI','NOVA4'] for t in TAXA]
    for j,(ax,o) in enumerate(zip(axes,OUTCOMES)):
        for y,(e,t) in enumerate(combinations):
            row=focus[(focus.exposure==e)&(focus.taxon==t)&(focus.outcome==o)].iloc[0]
            if y>=4:ax.axhspan(y-.5,y+.5,color='#f2f5f7',zorder=0)
            if row.status=='OK':point(ax,row[BETA],row[LOW],row[HIGH],y,filled=row.q_focus24<.05)
            else:ax.text(.98,y,'NE',transform=ax.get_yaxis_transform(),ha='right')
        ax.axvline(0,color='#666666',lw=.6,ls='--');ax.set_title(chr(97+j)+'  '+o,loc='left',fontweight='bold')
        ax.set_xlabel('Interaction ('+('pp' if o=='CV' else 'mg/dL')+')')
        ax.set_yticks(range(8));ax.set_ylim(7.7,-.7)
    axes[0].set_yticklabels([e+' | '+t for e,t in combinations])
    fig.legend(handles=[Line2D([],[],marker='o',color='#0072b2',ls='',ms=3,label='qFocus24 < 0.05'),Line2D([],[],marker='o',mfc='white',mec='#0072b2',ls='',ms=3,label='qFocus24 ≥ 0.05')],loc='upper center',bbox_to_anchor=(.59,.98),ncol=2,frameon=False)
    fig.subplots_adjust(left=.17,right=.98,top=.83,bottom=.25,wspace=.30)
    fields=KEY+['species','n',BETA,LOW,HIGH,'q_focus24','q_all84','unit','status']
    finish(fig,'Figure_I1_focus_interactions',focus.reindex(columns=fields),out,bundle,figures,
           'Per original Diet Z × microbial CLR-Z. All 24 focus tests shown.\nT068: E. rectale; T096: R. lactatiformans. Full species names and model N: Source Data.')
    # These are continuous-model slopes evaluated at quantiles, not subgroup estimates.
    slopes=bundle['slopes']; nova=slopes[(slopes.exposure=='NOVA4')&(slopes.taxon=='T068')&(slopes.model=='primary')&slopes.outcome.isin(['CV','SDw'])].copy()
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    for j,(ax,o) in enumerate(zip(axes,['CV','SDw'])):
        r=nova[nova.outcome.eq(o)].sort_values('microbial_quantile')
        ax.errorbar(r.microbial_quantile,r.beta_native,yerr=[r.beta_native-r.ci_low_native,r.ci_high_native-r.beta_native],fmt='o',ms=3,color='#0072b2',capsize=2,lw=.8)
        ax.axhline(0,color='#666666',lw=.6,ls='--');ax.set_xticks([25,50,75]);ax.set_xticklabels(['Q25','Q50','Q75']);ax.set_xlim(17,83)
        ax.set_ylabel('NOVA4 slope ('+('pp' if o=='CV' else 'mg/dL')+' / Diet Z)')
        ax.set_xlabel('E. rectale reference quantile');ax.set_title(chr(97+j)+'  '+o+' | N='+str(int(r.n.iloc[0])),loc='left',fontweight='bold')
    fig.subplots_adjust(left=.10,right=.98,top=.86,bottom=.34,wspace=.4)
    finish(fig,'Figure_I2_NOVA4_conditional_slopes',public(nova,SLOPE_FIELDS),out,bundle,figures,'Slopes evaluated within the same continuous primary model; participants are not divided into three groups.')
    # Discrete bins retain an explicit 0.05 threshold; numbers remain primary for interpretation.
    st=bundle['stability'].sort_values(KEY).reset_index(drop=True)
    if len(st):
        qfields=['reference_q_all84']+[m+'__q_all84' for m in MODELS]
        values=st[qfields].to_numpy(float)
        bins=np.select([~np.isfinite(values),values<.001,values<.01,values<.05],[4,0,1,2],default=3)
        fig,ax=plt.subplots(figsize=(183/25.4,min(170,max(105,60+5.3*len(st)))/25.4))
        ax.imshow(bins,cmap=ListedColormap(['#173f5f','#2f7e9a','#badce5','#f2f2f2','#ead1ae']),vmin=0,vmax=4,aspect='auto')
        for i in range(len(st)):
            for j in range(7):
                value=values[i,j];txt='NE' if not np.isfinite(value) else '{:.3g}'.format(value)
                ax.text(j,i,txt,ha='center',va='center',fontsize=5.7,color='white' if bins[i,j]<=1 else 'black')
            ax.text(7.05,i,str(int(st.retained_sensitivity_count.iloc[i]))+'/6',ha='left',va='center',fontsize=6,fontweight='bold' if st.retained_sensitivity_count.iloc[i]==6 else 'normal')
        ax.set_yticks(range(len(st)));ax.set_yticklabels([SHORT.get(r.exposure,r.exposure)+' | '+r.taxon+' | '+r.outcome for _,r in st.iterrows()])
        ax.set_xticks(range(7));ax.set_xticklabels(['Primary']+CODES);ax.xaxis.tick_top();ax.tick_params(length=0)
        ax.set_xlim(-.5,7.75);ax.set_title('Original significant extension targets | qAll84',loc='left',fontweight='bold',pad=25)
        ax.text(7.05,1.025,'Retained',fontsize=6,ha='left',va='bottom',transform=ax.get_xaxis_transform(),clip_on=False)
        for y in np.arange(.5,len(st),1):ax.axhline(y,color='white',lw=.6)
        fig.subplots_adjust(left=.275,right=.965,top=.86,bottom=.21)
        finish(fig,'Figure_I3_extension_stability',st.reindex(columns=KEY+['primary_n','min_sensitivity_n']+qfields+['retained_sensitivity_count','same_direction_all_six']),out,bundle,figures,
            'Dark / medium / light blue: q < 0.001 / 0.01 / 0.05; grey: q ≥ 0.05; tan: not estimable.\nS: strict; P: protocol sample; A: + medication; Q: quadratic; C: covariate interactions; U: all-seven sample.\nRetention requires estimability, same interaction sign and qAll84 < 0.05. Cohorts overlap.')
    selected=st.loc[true(st.retained_all_six),KEY] if len(st) else st.reindex(columns=KEY)
    picked=slopes[slopes.model.eq('primary')&slopes.microbial_quantile.isin([25,75])].merge(selected,on=KEY,validate='many_to_one')
    if len(selected):
        # At most nine panels per page; selection is stated explicitly in the legend.
        for page,start in enumerate(range(0,len(selected),9),1):
            part=selected.iloc[start:start+9]
            nrows=int(np.ceil(len(part)/3));fig,axes=plt.subplots(nrows,3,figsize=(183/25.4,min(170,42*nrows+32)/25.4),squeeze=False)
            plotted=[]
            for idx,(_,key) in enumerate(part.iterrows()):
                ax=axes.flat[idx];r=picked.copy()
                for k in KEY:r=r[r[k]==key[k]]
                r=r.sort_values('microbial_quantile');plotted.append(r)
                ax.errorbar(r.microbial_quantile,r.beta_native,yerr=[r.beta_native-r.ci_low_native,r.ci_high_native-r.beta_native],fmt='o',ms=3,color='#0072b2',lw=.7,capsize=2)
                ax.axhline(0,color='#666666',lw=.6,ls='--');ax.set_xticks([25,75]);ax.set_xticklabels(['Q25','Q75']);ax.set_xlim(15,85)
                ax.set_title(chr(97+idx)+'  '+SHORT.get(key.exposure,key.exposure)+' | '+key.taxon+' | '+key.outcome,loc='left',fontweight='bold',fontsize=6.5)
                ax.set_ylabel(('pp' if key.outcome=='CV' else 'mg/dL')+' / Diet Z')
            for ax in axes.flat[len(part):]:ax.axis('off')
            fig.subplots_adjust(left=.095,right=.98,top=.91,bottom=.20,hspace=.63,wspace=.5)
            name='Figure_I4_selected_extension_slopes'+('_'+str(page) if len(selected)>9 else '')
            finish(fig,name,public(pd.concat(plotted),SLOPE_FIELDS),out,bundle,figures,'Primary-model slopes at microbial Q25 / Q75, for targets retained in all six sensitivities. Panel y-scales differ.\nSelected after inspecting sensitivity results; these panels are not independent validation.')
    # Full unselected extension forests are only possible from full server tables.
    ext=bundle['extension']
    if len(ext):
        for o in OUTCOMES:
            for page,diets in enumerate([DIETS[2:5],DIETS[5:]],1):
                pairs=[(e,t) for e in diets for t in TAXA]
                frame=ext[ext.exposure.isin(diets)&ext.outcome.eq(o)]
                fig,ax=plt.subplots(figsize=(183/25.4,160/25.4))
                for y,(e,t) in enumerate(pairs):
                    if y%2==0:ax.axhspan(y-.48,y+.48,color='#f3f5f7',zorder=0)
                    for m,offset,color in zip(['primary']+MODELS,np.linspace(-.32,.32,7),COLORS):
                        r=frame[(frame.exposure==e)&(frame.taxon==t)&(frame.model==m)].iloc[0]
                        if r.status=='OK':point(ax,r[BETA],r[LOW],r[HIGH],y+offset,color,filled=r.q_all84<.05,size=2.5)
                        else:ax.text(.99,y+offset,'NE',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=5)
                ax.axvline(0,color='#666666',lw=.6,ls='--');ax.set_yticks(range(len(pairs)));ax.set_yticklabels([SHORT.get(e,e)+' | '+t for e,t in pairs]);ax.set_ylim(len(pairs)-.4,-.6)
                ax.set_title(o+' | All extension combinations, part '+str(page),loc='left',fontweight='bold',pad=25)
                ax.set_xlabel('Interaction per Diet Z × microbial CLR-Z ('+('pp' if o=='CV' else 'mg/dL')+')')
                handles=[Line2D([],[],color=c,marker='o',ms=2.5,lw=.7,label=l) for c,l in zip(COLORS,['Primary']+CODES)]
                fig.legend(handles=handles,ncol=7,loc='upper center',bbox_to_anchor=(.58,.94),frameon=False)
                fig.subplots_adjust(left=.21,right=.98,top=.85,bottom=.19)
                finish(fig,'Figure_S_'+o+'_all_extensions_'+str(page),frame.reindex(columns=KEY+['model','n',BETA,LOW,HIGH,'q_all84','status','reason']),out,bundle,figures,'All combinations retained, including nulls and non-estimable models. Filled: qAll84 < 0.05; open: qAll84 ≥ 0.05.\nPrimary + S/P/A/Q/C/U; each specification controls BH across all 84 tests. Units differ across outcomes.')
    return figures
