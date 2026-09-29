"""Verify archived displayed results and plot them; no model fits or new inference."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCES = {
    'nutrition': ROOT / 'nutrition_bridge_累计已展示结果_截图精度.json',
    'response': ROOT / '2026-09-28_response_followup_三份摘要_截图精度.json',
    'date_split': ROOT / '2026-09-28_date_split_bridge_三份摘要_截图精度.json',
    'gut': ROOT / '2026-09-29_菌群餐前餐后衔接结果/截图数值转录.json',
    'timing': ROOT / '2026-09-29_菌群餐前餐后衔接结果/时间核对_截图转录.json',
    'early': OUT / '早期真实结果_核对转录.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = {key: digest(path) for key, path in SOURCES.items()}
    d = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}
    checks = []

    def check(name, actual, expected):
        if actual != expected:
            raise ValueError('{}: {} != {}'.format(name, actual, expected))
        checks.append(dict(check=name, actual=actual, expected=expected, status='PASS'))

    for key, n in [('nutrition', 302), ('response', 42), ('date_split', 24)]:
        check(key + '_displayed_rows', len(d[key]['rows']), n)
    nutrition = d['nutrition']['rows']
    fields = ['analysis', 'setting', 'branch', 'outcome', 'model', 'diet', 'meal_measure', 'cgm']
    check('unique_nutrition_specs', len({tuple(row.get(k) for k in fields) for row in nutrition}), 302)
    check('nutrition_CI_contains_estimate', all(r['ci_low'] <= r['estimate'] <= r['ci_high'] for r in nutrition), True)
    date = d['date_split']['rows']
    check('unique_date_specs', len({(r['setting'], r['meal_measure'], r['cgm']) for r in date}), 24)
    check('positive_date_associations', sum(r['late_estimate'] > 0 for r in date), 24)
    check('date_primary_q24_pass', sum(r['q24_below_005'] for r in date), 24)
    check('date_delta_negative', sum(r['delta_estimate'] < 0 for r in date), 24)
    check('date_delta_q24_pass', sum(r['qD24_below_005'] for r in date), 20)
    volatility = [r for r in date if r['cgm'] != 'Mean']
    check('volatility_specs', len(volatility), 18)
    check('volatility_delta_q_pass', sum(r['qD24_below_005'] for r in volatility), 18)
    meals = [r for r in d['gut']['models'] if r['outcome'] in ['pre', 'net', 'peak']]
    check('gut_displayed_models', len(d['gut']['models']), 36)
    check('gut_meal_models', len(meals), 24)
    check('unique_gut_meal_specs', len({(r['setting'], r['outcome'], r['taxon']) for r in meals}), 24)
    check('gut_CI_contains_estimate', all(r['ci_low'] <= r['estimate'] <= r['ci_high'] for r in meals), True)
    counts = {}
    for setting, expected in [('primary_common', [3, 0, 0]), ('strict_common', [3, 2, 0])]:
        counts[setting] = [sum(r['setting'] == setting and r['outcome'] == out and r['q_family'] < .05 for r in meals)
                           for out in ['pre', 'net', 'peak']]
        check(setting + '_q_support_pre_net_peak', counts[setting], expected)
    for row in d['timing']['rows']:
        check('timing_bins_total_' + row['sample'] + '_' + row['window'],
              sum(row['days_from_window_end'].values()), row['total'])
        check('timing_position_total_' + row['sample'] + '_' + row['window'],
              sum(row[k] for k in ['before', 'within', 'after', 'unknown']), row['total'])
    main_cgm = next(r for r in d['timing']['rows'] if r['sample'] == 'Main' and r['window'] == 'CGM')
    check('known_gut_dates', main_cgm['date_known'], 3575)
    check('after_CGM', main_cgm['after'], 3532)
    check('after_CGM_median_IQR', main_cgm['days_from_end_quantiles'][1:4], [7, 9, 12])
    check('main_single_visit_match', d['timing']['main_date_match_status']['SINGLE_VISIT_RECORD'], 3575)

    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'font.size': 10,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.7), sharey=True)
    fig.subplots_adjust(left=.215, right=.98, top=.78, bottom=.27, wspace=.16)
    taxa = list(d['gut']['taxa'])
    colors = ['#126A87', '#B66723']
    labels = ['主样本 N=3,631', '严格样本 N=3,398']
    for index, (outcome, title) in enumerate([('pre', '餐前水平 · 次要'), ('net', '餐后平均净增量 · 主要'), ('peak', '峰值升幅 · 次要')]):
        ax = axes[index]
        ax.axvline(0, color='#7B8792', linewidth=1, linestyle='--', zorder=0)
        for y in [0, 2]:
            ax.axhspan(y-.48, y+.48, color='#F1F4F6', zorder=0)
        for j, setting in enumerate(['primary_common', 'strict_common']):
            for i, taxon in enumerate(taxa):
                row = next(r for r in meals if r['setting'] == setting and r['outcome'] == outcome and r['taxon'] == taxon)
                y = 3-i + (.13 if j == 0 else -.13)
                beta = row['estimate']
                ax.errorbar(beta, y, xerr=[[beta-row['ci_low']], [row['ci_high']-beta]],
                    fmt='o' if j == 0 else 's', color=colors[j], markersize=6.3,
                    markerfacecolor=colors[j] if row['q_family'] < .05 else 'white',
                    markeredgewidth=1.3, elinewidth=1.3, capsize=2.5, zorder=3)
        ax.set_title('{}  {}'.format('ABC'[index], title), loc='left', fontsize=12, pad=28)
        ax.text(0, 1.02, '通过原定q：主 {}/4；严格 {}/4'.format(counts['primary_common'][index], counts['strict_common'][index]),
                transform=ax.transAxes, fontsize=9, color='#43515C')
        ax.set_xlim(-1.12, .85)
        ax.set_xticks([-1, -.5, 0, .5])
        ax.set_ylim(-.48, 3.48)
        ax.grid(axis='x', color='#DFE5E9', linewidth=.65, zorder=0)
        ax.tick_params(axis='y', length=0, pad=12)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B3BBC2')
        ax.set_xlabel('β（mg/dL / 原菌 CLR-Z）', labelpad=10)
    species = ['T023  GGB6012_SGB8640', 'T046  Clostridium phoceensis',
               'T068  Eubacterium rectale', 'T096  Ruthenibacterium lactatiformans']
    axes[0].set_yticks([3, 2, 1, 0], labels=species, fontsize=9)
    fig.suptitle('四个既有菌候选与餐前、餐后血糖：全部餐次结果', x=.035, y=.955,
                 ha='left', fontsize=17, fontweight='bold', color='#17313F')
    handles = [Line2D([0], [0], color=colors[j], marker='o' if j == 0 else 's', linewidth=1.3,
                      label=labels[j], markersize=6) for j in [0, 1]]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.03, .91), frameon=False, ncol=2)
    foot = ('点和线为β及逐项95%CI；实心：原家族q<0.05，空心：q≥0.05。CI不跨零不保证通过多重校正。\n'
            '净增量采用4项主家族；餐前与峰值共同采用8项次要家族；两样本分别校正且成员重叠。\n'
            '结局定义与调整集不同，未进行结局间效应差异检验。图用已归档截图精度，不是新模型或完整精度结果。')
    fig.text(.035, .14, foot, ha='left', va='top', fontsize=9, color='#43515C', linespacing=1.8)
    figs = OUT / 'figures'
    figs.mkdir(exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(figs / ('Figure5_gut_meal.' + ext), dpi=180, facecolor='white')
    plt.close(fig)
    check('source_JSON_unchanged', {key: digest(path) for key, path in SOURCES.items()}, before)
    attenuation = [-r['delta_estimate']/r['full_estimate']*100 for r in volatility]
    body = dict(status='PASS', scope='Local archived displayed-value checks and figure; not server model validation',
                new_model_fits=0, new_p_q_computations=0, existing_source_JSON_modified=False,
                displayed_precision_only=True, checks=checks,
                derived_descriptions=dict(volatility_decrease_percent_min=min(attenuation),
                    volatility_decrease_percent_max=max(attenuation),
                    after_CGM_percent_known=main_cgm['after']/main_cgm['date_known']*100),
                sources=[dict(label=k, path=str(p.relative_to(ROOT)), sha256=before[k]) for k,p in SOURCES.items()],
                figure_files=[dict(path=str(p.relative_to(OUT)), sha256=digest(p)) for p in sorted(figs.glob('Figure5_gut_meal.*'))],
                limitations=['No raw individual data or full precision model files locally',
                             'No exact stool sample/date linkage verified',
                             'Strict/joint are overlapping sensitivity samples',
                             'CI and q are retained from original declared families'])
    (OUT / '本轮整合核查.json').write_text(json.dumps(body, ensure_ascii=False, indent=2)+'\n')
    print('PASS:', len(checks), 'archive checks; saved PNG/SVG; no new model fits.')


if __name__ == '__main__':
    main()
