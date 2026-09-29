"""Render all24 screenshot-precision associations and paired contrasts; no new inference."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 source=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 d=json.loads(source.read_text());assert d['data_kind']=='REAL' and len(d['rows'])==24
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,2,figsize=(12.6,8),sharey=True)
 colors=['#21669a','#ce673b'];settings=['source_primary','joint'];marks=['o','s']
 combos=[(x,y) for x in ['Net','Peak','Positive'] for y in ['Mean','CV','SDw','MODD15']]
 for j,setting in enumerate(settings):
  for i,(x,y) in enumerate(combos):
   r=next(r for r in d['rows'] if r['setting']==setting and r['display']['X']==x and r['cgm']==y)
   for ax,prefix,significant in [(axes[0],'late',r['q24_below_005']),(axes[1],'delta',r['qD24_below_005'])]:
    b,lo,hi=[r[prefix+'_'+k] for k in ['estimate','ci_low','ci_high']]
    ax.errorbar(b,i+(-.13 if j==0 else .13),xerr=[[b-lo],[hi-b]],fmt=marks[j],color=colors[j],
     markerfacecolor=colors[j] if significant else 'white',markeredgecolor=colors[j],ms=5.2,lw=1.15,capsize=2,zorder=3)
 for ax in axes:
  ax.set_ylim(11.65,-.65)
  for lo,hi in [(-.5,3.5),(7.5,11.5)]:ax.axhspan(lo,hi,color='#f2f5f7',zorder=0)
  for y in [3.5,7.5]:ax.axhline(y,color='#d0d7dc',lw=.7)
  ax.axvline(0,color='#77838d',lw=.9,ls='--');ax.grid(axis='x',lw=.5,color='#e1e6eb');ax.set_axisbelow(True)
 axes[0].set_yticks(range(12));axes[0].set_yticklabels(['{}  /  {}'.format(x,y) for x,y in combos])
 axes[1].tick_params(axis='y',labelleft=False)
 axes[0].set_xlim(-.025,.74);axes[1].set_xlim(-.075,.016)
 axes[0].set_title('A  |  Early meal response → later CGM',loc='left',fontweight='bold',pad=12)
 axes[1].set_title('B  |  Paired change: later − full window',loc='left',fontweight='bold',pad=12)
 axes[0].set_xlabel('Later-window β (pointwise 95% CI)')
 axes[1].set_xlabel('Δβ (paired pointwise 95% CI)')
 fig.suptitle('Meal responses remain associated with CGM on separate dates',x=.03,y=.995,ha='left',fontsize=16,fontweight='bold')
 fig.text(.03,.935,'All 24 prespecified primary associations are positive with q < 0.05; the samples overlap.',fontsize=10.5)
 handles=[Line2D([],[],ls='',marker=marks[i],color=colors[i],label=label) for i,label in enumerate(['Original sample','Joint quality sample'])]
 handles += [Line2D([],[],ls='',marker='o',color='#505b64',label='Filled: q < 0.05'),Line2D([],[],ls='',marker='o',color='#505b64',mfc='white',label='Open: q ≥ 0.05')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.912),ncol=4,frameon=False)
 fig.subplots_adjust(left=.17,right=.98,top=.81,bottom=.16,wspace=.23)
 fig.text(.03,.083,'Unit: full-window reference Y SD per reference SD of early meal response. Original N = 4,435 / 4,433 (MODD15); joint N = 4,136 / 4,134.',fontsize=8.6)
 fig.text(.03,.058,'BH families: 24 primary tests in A; a separate secondary family of 24 paired contrasts in B. Full window includes later dates.',fontsize=8.6)
 fig.text(.03,.033,'Screenshot display precision; no re-fitting or re-testing. New fixed grid definitions; MODD15 is not asserted equal to legacy iglu MODD.',fontsize=8.6)
 files=[]
 for ext in ['png','pdf','svg']:
  dest=out/('Figure4_date_split.'+ext);fig.savefig(dest,dpi=180,facecolor='white',bbox_inches='tight');files.append(dest.name)
 plt.close(fig)
 provenance=dict(run_id=d['run_id'],data_kind='REAL',input_name=source.name,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),all_primary_rows=24,all_paired_contrasts=24,
  q_recomputed=False,new_models_fitted=False,precision=d['precision'],outputs={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in files})
 (out/'artifact_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
