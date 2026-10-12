"""Paper-facing aggregate tables, bilingual draft and a local reading page."""
import html
import json
import numpy as np
import pandas as pd
from .data import KEY, MODELS, CODES, BETA, LOW, HIGH, UNITS, SPECIES, MODEL_FIELDS, SLOPE_FIELDS, public, true

CSS='''body{font:15px/1.65 Arial,"PingFang SC",sans-serif;color:#1c2b35;max-width:1200px;margin:40px auto;padding:0 25px}h1{font-size:28px}h2{margin-top:36px;font-size:21px}p{max-width:950px}.notice{background:#eef4f7;border-left:4px solid #377d96;padding:12px 18px}table{border-collapse:collapse;font-size:12px;width:100%;margin:15px 0}th{background:#edf3f6;text-align:left}td,th{border-bottom:1px solid #dce4e8;padding:7px 9px;white-space:nowrap}tr:nth-child(even){background:#f8fafb}.scroll{overflow:auto}img{max-width:100%;height:auto}a{color:#175c81}pre{white-space:pre-wrap;font:13px/1.5 monospace}.fig{page-break-inside:avoid;max-width:1000px}footer{margin-top:35px;color:#647681}@media print{body{margin:8mm;max-width:none}table{font-size:8pt}.notice{font-size:10pt}}'''


def num(value,digits=4):
    if pd.isna(value):return 'NA'
    return ('{:.%dg}'%digits).format(float(value))


def format_effect(row):
    if pd.isna(row.get(BETA)):return 'NA'
    return '{} [{}, {}]'.format(num(row[BETA]),num(row[LOW]),num(row[HIGH]))


def prepare_tables(bundle):
    primary=bundle['primary'];st=bundle['stability'];slopes=bundle['slopes']
    focus=primary[primary.family.eq('focus')].copy()
    selected=primary[primary.family.eq('extension')].merge(st,on=KEY,validate='one_to_one',suffixes=('','_stability'))
    desired=KEY+['n',BETA,LOW,HIGH,'interaction_p','q_all84','min_sensitivity_n']+[m+'__q_all84' for m in MODELS]+['same_direction_all_six','retained_sensitivity_count','retained_all_six']
    tables={
      'Table_1_focus_primary':public(focus,MODEL_FIELDS),
      'Table_2_original_extension_targets':public(selected,desired),
      'Table_3_extension_sensitivity_summary':bundle['summary'].copy(),
      'Table_S1_primary_all_available':public(primary,MODEL_FIELDS),
      'Table_S2_original_models':public(bundle['original'],MODEL_FIELDS),
      'Table_S4_conditional_slopes':public(slopes,SLOPE_FIELDS)}
    if len(bundle['all_followup']):tables['Table_S3_followup_all_models']=public(bundle['all_followup'],MODEL_FIELDS)
    if len(bundle['fdr']):tables['Table_S5_fdr_families']=bundle['fdr'].copy()
    if len(bundle['reproduction']):tables['Table_S6_reference_reproduction']=public(bundle['reproduction'],KEY+['model','matches','differing_fields','old_n','current_n'])
    if len(bundle['extension']):
        keys=set(st[KEY].itertuples(index=False,name=None))
        ext=bundle['extension']
        extra=ext[~ext.model.eq('primary')&ext.status.eq('OK')&ext.q_all84.lt(.05)].copy()
        extra=extra[[tuple(r) not in keys for r in extra[KEY].itertuples(index=False,name=None)]]
        tables['Table_S7_new_signals_outside_original']=public(extra,MODEL_FIELDS)
    stable=selected[true(selected.retained_all_six)].copy()
    stable['max_sensitivity_q_all84']=stable[[m+'__q_all84' for m in MODELS]].max(axis=1)
    tables['Table_4_targets_retained_all_six']=public(stable,KEY+['n',BETA,LOW,HIGH,'interaction_p','q_all84','max_sensitivity_q_all84','retained_sensitivity_count'])
    return tables


def prose(bundle,tables):
    p=bundle['primary']; st=bundle['stability']; focus=p[p.family.eq('focus')]
    f=int(focus.q_focus24.lt(.05).sum());all_sig=int(p.q_all84.lt(.05).sum());stable=int(true(st.retained_all_six).sum())
    text=['# 饮食与菌群交互的论文结果草稿','',
      '来源状态：'+bundle['meta']['mode']+'。'+('本地截图精度预览；用于讨论和排版，正式引用请使用服务器导出替换。' if bundle['meta']['mode']=='SCREENSHOT_PREVIEW' else '由已核验的汇总结果导出；投稿前仍需作者确认研究定义与全文语境。'),'',
      '## 中文 Results','',
      '在参与者层面的整体CGM分析中，预先固定的24项重点交互中有{}项通过重点家族的BH校正。完整84项主交互共有{}项通过整体校正，其中{}项来自60项扩展检验。上述检验涉及在同一队列既往分析中筛选的四个菌群特征，结果解释为探索性效应修饰关联。'.format(f,all_sig,len(st)),'']
    for outcome in ['CV','SDw']:
        r=focus[(focus.exposure=='NOVA4')&(focus.taxon=='T068')&(focus.outcome==outcome)].iloc[0]
        text.append('NOVA4×E. rectale 对{}的交互系数为{}（逐项95%CI {}至{}；N={}；qFocus24={}；qAll84={}），单位为每1原饮食Z×1菌CLR-Z对应的{}。'.format(outcome,num(r[BETA]),num(r[LOW]),num(r[HIGH]),int(r.n),num(r.q_focus24),num(r.q_all84),UNITS[outcome]))
    text+=['','在查看主结果之后，对全部60项扩展组合补充六组敏感性分析；每个设定均连同24项重点组合在完整84项内校正。原显著{}项中，{}项在全部六组中可估计且方向一致，{}项进一步在六组中均满足qAll84<0.05。通过设定的数量仅作为描述性稳定性指标，不能视为相互独立的复制。'.format(len(st),int(true(st.same_direction_all_six).sum()),stable),'',
       '六组均保留的组合及其原主分析效应见Table 4；原全部显著扩展见Table 2，包含未能在全部设定保留者。条件斜率展示的是同一连续模型在菌群参考分位点处的估计，并非将人群分组重新拟合。统计显著性不代表效应具有临床重要性，也不证明菌群造成饮食效应改变。','',
       '## English Results','',
       'In participant-level analyses of overall CGM phenotypes, {} of 24 designated focus interactions met the focus-family Benjamini–Hochberg threshold (qFocus24 < 0.05). Across the full set of 84 primary interactions, {} met qAll84 < 0.05, including {} of the 60 extension tests. The four microbial features had been selected in earlier analyses of the same cohort; these analyses therefore assess exploratory effect modification rather than independent validation.'.format(f,all_sig,len(st)),'']
    for outcome in ['CV','SDw']:
        r=focus[(focus.exposure=='NOVA4')&(focus.taxon=='T068')&(focus.outcome==outcome)].iloc[0]
        text.append('The NOVA4-by-E. rectale interaction for {} was {} (pointwise 95% CI {}, {}; N={}; qFocus24={}; qAll84={}), expressed in {} per original dietary-score Z unit by microbial CLR-Z unit.'.format(outcome,num(r[BETA]),num(r[LOW]),num(r[HIGH]),int(r.n),num(r.q_focus24),num(r.q_all84),UNITS[outcome]))
    text+=['','Following inspection of the primary results, six sensitivity specifications were applied to all extension combinations, with BH adjustment across 84 tests within each specification. Of the {} originally significant extension interactions, {} were estimable and retained the same sign in all six specifications, and {} additionally met qAll84 < 0.05 in every specification. This consistency across overlapping samples provides within-cohort sensitivity evidence and does not establish replication or causal effects.'.format(len(st),int(true(st.same_direction_all_six).sum()),stable),'',
      '## 可用于 Methods 的英文段落','',
      'We fitted participant-level ordinary least-squares models for mean CGM glucose, glucose coefficient of variation (CV), and the mean of daily within-day glucose standard deviations (SDw), including the dietary score, microbial feature, their product term, and the prespecified covariates. Dietary scores and microbial centred-log-ratio features retained their original standardised units. Diet and microbial features were centred using their corresponding primary-model sample means; these reference values were retained in sensitivity analyses. HC3 heteroskedasticity-robust covariance estimates and residual-degrees-of-freedom t inference were used. Coefficients and pointwise 95% confidence intervals were expressed on the original clean CGM scale (mg/dL for mean glucose and SDw; percentage points for CV). Models used complete cases for each combination. Conditional dietary slopes were evaluated at microbial reference quantiles using the full coefficient covariance; these were not subgroup regressions.','',
      'Covariates comprised age, sex, education, smoking, sleep duration, physical activity, vitamin use, hormone use, BMI and CGM device. hPDI models additionally adjusted for alcohol consumption, whereas EAT13, NOVA4 and carbohydrate energy-percentage models additionally adjusted for mean energy intake. Primary analyses excluded participants with an affirmative combined diabetes/A10 medication flag; the strict specification required an explicitly negative flag. Further sensitivity specifications used a common medication-information sample with and without antibiotic/PPI adjustment, added dietary and microbial quadratic terms, or added interactions of diet and microbiota with age, sex and BMI. The original focus common-sample analysis required completeness for two dietary scores; the extension follow-up required all seven scores, all four taxa, all three outcomes and their required covariates.','',
      'Benjamini–Hochberg adjustment was performed separately for the designated 24-test focus family and the full 84-test primary family. Original focus sensitivities used their own 24-test families. The post-primary extension follow-up used a fixed 84-test family within each of seven specifications (primary plus six sensitivities). Unavailable tests remained in each denominator with p=1, while their own q values were left missing. Extension retention required an estimable interaction, the same sign as its original primary estimate and qAll84<0.05 in each sensitivity. No multiplicity correction over only the selected extension hits was performed.','',
      '## 作者需保留的解释边界','',
      '- SDw按项目已核对定义表述为各日内葡萄糖SD的平均值，不称日间SD。历史计算版本和参数的溯源边界仍沿用原项目。定义参考[IGLU官方SD说明](https://irinagain.github.io/iglu/reference/sd_measures.html)。',
      '- rEDIH 沿用项目原变量，不在本图表包内翻转或重算。其原始计量单位及评分定义溯源仍须结合上游文档确认，正文暂保留变量名，不据名称自动添加健康方向解释。',
      '- 交互系数描述饮食斜率随菌群CLR-Z变化；CLR-Z不是绝对菌量。不能由交互正负直接给菌贴有益或有害标签。',
      '- 四菌同队列预选、单次菌群测量、测量时间关系和未测量混杂限制继续保留。相同队列的敏感性结果不是独立验证。',
      '- 六组全部保留的筛选发生于看到结果之后；正文展示选择必须披露，同时保留全部阴性结果。不同结局及饮食之间相关，不能将条目数作为独立发现数。',
      '- 如需报告临床量级，应同时给出交互原单位、条件斜率和双IQR对比；不将较小q值等同于较大或更重要的效应。','']
    if bundle['meta']['mode']=='SYNTHETIC_TEST':text.insert(2,'SYNTHETIC SOFTWARE TEST ONLY — all numerical paragraphs below are synthetic.')
    selected=tables['Table_4_targets_retained_all_six']
    descriptions=[]
    for (diet,taxon),group in selected.groupby(['exposure','taxon'],sort=False):
        descriptions.append('{} × {} ({})'.format(diet,SPECIES[taxon],', '.join(group.outcome)))
    position=text.index('## 可用于 Methods 的英文段落')
    text[position:position]=['The extension combinations meeting all six sensitivity criteria were '+('; '.join(descriptions) if descriptions else 'none')+'.','']
    for outcome in ['CV','SDw']:
        rows=bundle['slopes']
        rows=rows[(rows.exposure=='NOVA4')&(rows.taxon=='T068')&(rows.outcome==outcome)&(rows.model=='primary')].set_index('microbial_quantile')
        if 25 in rows.index and 75 in rows.index:
            a,b=rows.loc[25],rows.loc[75]
            sentence='For {}, the conditional NOVA4 slope was {} (95% CI {}, {}) at E. rectale Q25 and {} (95% CI {}, {}) at Q75, in {} per original NOVA4 Z unit.'.format(outcome,num(a.beta_native),num(a.ci_low_native),num(a.ci_high_native),num(b.beta_native),num(b.ci_low_native),num(b.ci_high_native),UNITS[outcome])
            position=text.index('## 可用于 Methods 的英文段落')
            text[position:position]=[sentence,'']
    return '\n'.join(text)


def legends():
    return '''# Figure and table legends

## Figure I1 Dietary score by microbial-feature interactions in the focus family

Points show primary interaction coefficients and horizontal bars show pointwise 95% HC3 confidence intervals for all 24 hPDI/NOVA4-by-taxon-by-outcome tests. Panels a–c show Mean, CV and SDw (the mean of daily within-day standard deviations), respectively. Coefficients are expressed per original Diet Z × microbial CLR-Z, in mg/dL for Mean/SDw and percentage points for CV. Filled symbols denote qFocus24<0.05 after BH adjustment across the 24 focus tests; open symbols denote qFocus24≥0.05. Dashed lines denote zero interaction. Individual N, both qFocus24 and qAll84, and exact coefficients are provided in the figure Source Data. T023=GGB6012_SGB8640; T046=Clostridium phoceensis; T068=Eubacterium rectale; T096=Ruthenibacterium lactatiformans. All taxa were selected previously in this cohort.

## Figure I2 Conditional NOVA4 slopes at reference Eubacterium rectale quantiles

Points and vertical bars show the primary-model NOVA4 slope and pointwise 95% HC3 confidence interval evaluated at the microbial Q25, Q50 and Q75 values of the corresponding primary complete-case sample. Panels a and b show CV and SDw. Slopes are per original NOVA4 Z unit; units and N are indicated on each panel. These estimates are obtained from one continuous interaction model using the coefficient covariance, not from separate participant groups. Quantiles are within-model reference positions; equal spacing on the horizontal axis does not imply equally spaced microbial CLR-Z values. No new FDR procedure is applied to these conditional intervals. A significant slope at one quantile and a nonsignificant slope at another does not itself establish interaction.

## Figure I3 Sensitivity of the original extension interaction targets

Rows show every extension combination whose original primary qAll84 was below 0.05. Columns show the original primary q and six sensitivity q values: S, strict negative diabetes/A10 flag; P, medication-information sample; A, the same sample with antibiotic/PPI adjustment; Q, additional diet and microbial quadratic terms; C, additional interactions of diet and microbial features with age, sex and BMI; U, a common complete sample for all seven diets. BH adjustment includes all 84 tests within each specification, rather than only the displayed targets. The right column counts sensitivity specifications with estimable, same-sign interactions and qAll84<0.05. Row-specific primary N and minimum sensitivity N are provided in Source Data; Nmin is not the N of every model. Cell colour bands represent q<0.001, q<0.01, q<0.05, q≥0.05 or non-estimability; numeric q values are shown. Samples overlap and this is post-primary within-cohort sensitivity analysis.

## Figure I4 Primary conditional slopes for extensions retained in all six sensitivities

Panels show original primary-model dietary slopes at microbial Q25 and Q75 for extension targets that were estimable, same-sign and qAll84<0.05 in all six sensitivity specifications. This panel selection is based on the observed sensitivity results; all original targets remain shown in Figure I3 and all tested extensions in the supplementary figures. Points and vertical bars denote slopes per original dietary-score Z unit and pointwise 95% HC3 intervals. T068=E. rectale; T096=R. lactatiformans. Exact coefficients, model N and confidence intervals appear in the matching Source Data. Intervals are not simultaneous and slope sign alone should not be interpreted as evidence of microbial protection or harm.

## Supplementary figures All extension interaction estimates

Six pages cover all 60 extension combinations and all seven model specifications, grouped by outcome and dietary score. Points and horizontal bars show interaction coefficients and pointwise 95% HC3 confidence intervals on the native outcome scale. Colours identify specifications; filled/open markers indicate qAll84 below/at least 0.05, and NE denotes a non-estimable fit. No combination is omitted because of its p or q value. Each Source Data file includes the coefficients, confidence limits, qAll84, status and model-specific N plotted on that page. These pages require the full server aggregate tables and are intentionally absent from the screenshot preview.

## Tables

Table 1 reports all 24 focus primary estimates, N, intervals and both FDR families. Table 2 reports all originally significant extensions, their primary estimates and each sensitivity q, including targets not retained in all specifications. Table 3 describes all 60 extensions within each sensitivity. Table 4 is the selected subset retained in all six specifications; it must be cited with its post-result selection rule. Supplementary tables contain all 84 primary tests, all 228 original model rows, all 588 follow-up rows, conditional slopes, FDR-family definitions, reproduction checks and any sensitivity-only hits outside the frozen primary set. Failed or skipped tests retain their status and reason; blank numeric values mean unavailable, not zero. All intervals are pointwise. The screenshot preview includes only observed rows and explicitly reports incomplete coverage.
'''


def write_report(bundle,out,figures):
    tables=prepare_tables(bundle)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Diet–gut–CGM paper results</title><style>'+CSS+'</style><body>',
           '<h1>饮食 × 菌群 × 整体 CGM：论文图表</h1>',
           '<p class="notice">'+html.escape(bundle['meta']['mode'])+' · 原显著扩展 '+str(len(bundle['stability']))+' 项；六组均保留 '+str(int(true(bundle['stability'].retained_all_six).sum()))+' 项。各图和表均有可追溯汇总数据。</p>',
           '<p><a href="Results_and_methods.md">中英文结果与方法</a> · <a href="Figure_legends.md">英文图表说明</a> · <a href="Source_Data_README.md">Source Data口径</a></p>']
    for name,frame in tables.items():
        frame.to_csv(out/'tables'/(name+'.csv'),index=False,encoding='utf-8-sig')
    for name in ['Table_1_focus_primary','Table_2_original_extension_targets','Table_3_extension_sensitivity_summary','Table_4_targets_retained_all_six']:
        frame=tables[name].copy()
        if BETA in frame:
            frame['Beta [95% CI]']=frame.apply(format_effect,axis=1)
            frame=frame.drop(columns=[c for c in [BETA,LOW,HIGH,'interaction_se_native','reason','species','family','model','status'] if c in frame])
        frame=frame.rename(columns={m+'__q_all84':c+' qAll84' for m,c in zip(MODELS,CODES)})
        parts.append('<h2>'+name.replace('_',' ')+'</h2><p><a href="tables/'+name+'.csv">CSV source</a></p><div class="scroll">'+frame.to_html(index=False,na_rep='NA',float_format=lambda x:num(x),escape=True)+'</div>')
    parts.append('<h2>Figures</h2>')
    for figure in figures:
        name=figure['name'];parts.append('<section class="fig"><h3>'+name.replace('_',' ')+'</h3><p><a href="figures/'+name+'.pdf">PDF</a> · <a href="figures/'+name+'.svg">Editable SVG</a> · <a href="source_data/'+name+'.csv">Source Data</a></p><img src="figures/'+name+'.png" alt="'+name+'"></section>')
    parts.append('<h2>All exported tables</h2><ul>')
    for name,frame in tables.items():parts.append('<li><a href="tables/'+name+'.csv">'+name+'</a> · '+str(len(frame))+' rows</li>')
    parts.append('</ul><footer>Observational, within-cohort analysis. All figures use aggregate saved estimates, not participant data.</footer></body></html>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    (out/'Results_and_methods.md').write_text(prose(bundle,tables),encoding='utf-8')
    (out/'Figure_legends.md').write_text(legends(),encoding='utf-8')
    mode=bundle['meta']['mode']
    note='''# Source Data and precision

Mode: %s.

Each figure has one CSV under source_data with the exact numerical estimates used for plotting. The tables directory retains all available combinations, including nulls and failures. Native units are mg/dL (Mean/SDw) or percentage points (CV); interactions are per original Diet Z × microbial CLR-Z. Slope estimates are per original Diet Z. No participant rows, participant identifiers, raw input matrices or sample hashes are exported.

SCREENSHOT_PREVIEW uses only recorded displayed values. The 41 unshown primary extension tests, unshown p values and unshown sensitivity coefficients are unavailable, not zero. In this mode, Table S1 contains 43 observed primary rows rather than 84, and the all-extension supplemental forests are omitted. The 17:40 and 19:10 corrections take precedence. Do not use the preview as the final numerical source for submission.

SERVER_FULL_PRECISION reads saved CSV values without recomputing or rounding them before CSV export. Display tables and labels use rounded formatting for readability. Source file hashes are checked before and after export, and the two run identities and aggregate hashes are saved in export_manifest.json. No new model fits or significance tests are performed.

SYNTHETIC_TEST is software validation only; never use it as HPP evidence. Publication use still requires author confirmation of study definitions, particularly rEDIH provenance, and the applicable data-access agreement.

Layout reference: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/ (accessed 2026-10-10). PDF embeds TrueType fonts; SVG preserves text; PNG is for viewing. Figures use an 183 mm width and at most 170 mm height, following the general Nature figure guide as a design reference, not a claim of journal approval. Each figure has its matching local source-data file. Server files remain on the server; this workflow does not create an archive or require a download. Final numbering and main/supplement allocation depend on the full manuscript.
'''%mode
    (out/'Source_Data_README.md').write_text(note,encoding='utf-8')
    return tables
