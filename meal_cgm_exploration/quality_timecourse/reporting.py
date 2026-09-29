"""Complete, nonselective summaries; identifiers remain in private outputs."""
import numpy as np
import pandas as pd
from seven_diet_meals.inputs import DIETS


def num(value):
    return 'NA' if pd.isna(value) else '{:.4g}'.format(value)


def model_table(d, qcol):
    lines = ['| 饮食 | 人数 | 餐数 | 估计 | 95% CI | q | 状态 |', '|---|---:|---:|---:|---|---:|---|']
    for diet in DIETS:
        for r in d.loc[d.diet.eq(diet)].to_dict('records'):
            lines.append('| {} | {} | {} | {} | {} 至 {} | {} | {} |'.format(diet, r['people'], r['meals'], num(r['estimate']), num(r['ci_low']), num(r['ci_high']), num(r.get(qcol)), r['status']))
    return lines


def summary(run, state, result=None):
    lines = ['# 七项饮食：记录覆盖与餐后时间分段', '', '数据：**{}**；状态：**{}**。'.format(run.c['data_kind'], state['status']), '',
        '本轮回答三个问题：记录覆盖筛选后关联是否保留；关联位于前一小时还是后一小时；平均净增量/峰值及营养调整前后是否一致。', '',
        '日表存在表示该日期有饮食记录，不能确认每餐都记录完整。CGM缺口保留为缺失，不用血糖曲线反推并补造饮食。', '',
        '质量阈值为本轮预先固定的敏感性定义，不是官方完整性判定标准。完整CGM参考日来自观测跨度，不等于已确认连续佩戴。', '',
        'source_primary：复现原主样本；aligned：完整单连接本地日+同日/同时区日志；cgm90：再要求日CGM覆盖≥90%；log80：aligned且个人≥7参考日/日志覆盖≥80%；joint：两者同时满足；joint_people_all_meals：joint同一批人的原合格餐。', '']
    if state.get('reason'):
        lines += ['说明：' + state['reason'], '']
    if result is not None:
        table, times = result['models'], result['timecourse']
        lines += ['原主样本42个模型的状态、成员、系数及区间已复现。', '', '## 质量筛选：平均餐后净增量', '', '单位：每原始饮食Z单位对应mg/dL；保持原评分方向。', '']
        for setting in ['source_primary', 'joint_people_all_meals', 'joint']:
            lines += ['### ' + setting, ''] + model_table(table.loc[table.setting.eq(setting) & table.branch.eq('A') & table.model.eq('meal_adjusted') & table.outcome.eq('mean_increment')], 'q_family14') + ['']
        lines += ['## 质量筛选：个体内碳水斜率交互', '', '单位：每饮食Z、每10g餐次碳水的斜率差，不能将交互系数当作碳水总斜率。', '']
        for setting in ['source_primary', 'joint']:
            lines += ['### ' + setting, ''] + model_table(table.loc[table.setting.eq(setting) & table.branch.eq('B') & table.outcome.eq('mean_increment')], 'q_family14') + ['']
        lines += ['## 前一小时与后一小时', '', '均相对原餐前基线；晚减早直接拟合差值，不以两项显著性不同判断差异。', '']
        for setting in ['source_primary', 'joint']:
            for outcome in ['early_mean', 'late_mean', 'late_minus_early']:
                lines += ['### {} / {}'.format(setting, outcome), ''] + model_table(times.loc[times.setting.eq(setting) & times.outcome.eq(outcome)], 'q_family21') + ['']
        lines += ['## 文件阅读顺序', '',
            '1. tables/01_diary_audit.csv、02_coverage_overview.csv、02_coverage_quantiles.csv：日志人群、参考日期、CGM与饮食日期覆盖。',
            '2. tables/03_model_reproduction.csv：原42模型复现；04_diary_overlap.csv：各评分/分支与日表交集；04_sample_flow.csv：样本损失；04_selection_balance.csv：保留/排除人群描述。',
            '3. tables/05_quality_models.csv：全部252规格，含两种结局和分步筛选；05_quality_comparison.csv：并排比较。',
            '4. tables/06_timecourse_models.csv：原样本及joint的42个时间分段规格。',
            '5. tables/07_existing_adjustment_comparison.csv：原三套样本营养调整前后及峰值结果；系数变化不等同于中介比例，也未正式检验系数变化。',
            '6. figures/：全部七项评分的质量对照和时间分段森林图。', '',
            '本轮固定294个规格（252质量/复现+42时间分段），不是294个独立研究问题。每套质量样本沿用原三组14项FDR；时间分段每套21项FDR；未估计规格仍保留在计划分母。区间是逐项95%区间，不是多重校正区间。', '',
            '这些是重叠样本的探索性比较，不是独立复制。应结合效应大小、区间和样本损失判断稳定性；不能只看q是否跨0.05。早晚分段不直接测量胰岛素分泌或清除机制。', '',
            'private/保存身份、日期、连接及逐餐衍生数据，manifest.json含输入路径；对外汇报使用本摘要、tables/、figures/。']
    (run.out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_results(run, result, previous):
    from .analysis import adjusted_comparison
    mapping = {'reproduction': '03_model_reproduction.csv', 'flow': '04_sample_flow.csv', 'balance': '04_selection_balance.csv', 'diary_overlap': '04_diary_overlap.csv',
        'models': '05_quality_models.csv', 'coefficients': '05_quality_coefficients.csv',
        'timecourse': '06_timecourse_models.csv', 'time_coefficients': '06_timecourse_coefficients.csv'}
    for key, name in mapping.items():
        run.table(result[key], name)
    run.table(result['joint_membership'], '04_joint_membership.csv', private=True)
    run.table(adjusted_comparison(previous), '07_existing_adjustment_comparison.csv')
    comp = result['models'].pivot(index=['diet', 'branch', 'outcome', 'model'], columns='setting', values=['estimate', 'ci_low', 'ci_high', 'q_family14', 'people', 'meals', 'status'])
    comp.columns = ['__'.join(x) for x in comp.columns]
    run.table(comp.reset_index(), '05_quality_comparison.csv')


def figures(run, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    model = result['models']
    time = result['timecourse']
    def draw(ax, table, groups, group_col, title):
        colors = ['#2166ac', '#b35806', '#5e3c99']
        for j, (key, label) in enumerate(groups):
            d = table.loc[table[group_col].eq(key)].set_index('diet')
            for i, diet in enumerate(DIETS):
                r = d.loc[diet]
                pos = i + (j - (len(groups) - 1) / 2) * .18
                if r['status'] == 'ESTIMATED':
                    ax.errorbar(r.estimate, pos, xerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]], fmt='o', ms=4, color=colors[j], capsize=2)
                else:
                    ax.text(.99, pos, 'NA', transform=ax.get_yaxis_transform(), ha='right', fontsize=7, color=colors[j])
            ax.plot([], [], 'o', color=colors[j], label=label)
        ax.axvline(0, color='grey', lw=.8)
        ax.set_yticks(range(7))
        ax.set_yticklabels(DIETS)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('mg/dL per original diet Z (95% CI)')
        ax.legend(fontsize=8)
    a = model.loc[model.branch.eq('A') & model.model.eq('meal_adjusted') & model.outcome.eq('mean_increment')]
    draw(axes[0], a, [('source_primary', 'Original'), ('joint_people_all_meals', 'Joint people, original meals'), ('joint', 'Joint quality')], 'setting', 'Quality sensitivity: mean increment')
    for ax, setting in zip(axes[1:], ['source_primary', 'joint']):
        draw(ax, time.loc[time.setting.eq(setting)], [('early_mean', '0-60 min'), ('late_mean', '60-120 min'), ('late_minus_early', 'Late minus early')], 'outcome', 'Timecourse: ' + setting)
    fig.suptitle(run.c['data_kind'] + ' | Seven dietary variables; exploratory associations')
    fig.tight_layout()
    fig.savefig(run.out / 'figures/01_quality_timecourse.png', dpi=180)
    fig.savefig(run.out / 'figures/01_quality_timecourse.pdf')
    plt.close(fig)
