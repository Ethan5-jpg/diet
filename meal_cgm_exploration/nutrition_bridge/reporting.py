"""Readable aggregate results, paginated terminal output, and static figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hppmeal.common import require, sha256
from seven_diet_meals.inputs import DIETS
from .analysis import SETTINGS, STEPS, MEASURES


def formatted(d):
    return d.to_string(index=False, float_format=lambda x: '{:.4g}'.format(x), na_rep='NA')


def ordered(d):
    return d.set_index('diet').reindex(DIETS).reset_index()


def save_tables(run, result):
    names = dict(reproduction='02_reproduction', original='02_original_models', flow='03_sample_flow',
        adjustment='04_adjustment_models', correlations='04_score_nutrient_correlations',
        positive='05_positive_models', distribution='05_response_distribution',
        bridge_flow='06_bridge_flow', bridge='06_bridge_models', coefficients='07_coefficients')
    for key, name in names.items():
        run.table(result[key], name + '.csv')
    for key in ['members', 'bridge_members', 'bridge_people']:
        run.table(result[key], key + '.csv', private=True)
    old = result['original'].loc[result['original'].outcome.eq('mean_increment')]
    comparison = result['positive'].merge(old[['setting', 'diet', 'branch', 'model', 'estimate', 'person_sha256', 'meal_sha256']],
        on=['setting', 'diet', 'branch', 'model'], suffixes=('_positive', '_net'), validate='one_to_one')
    require(comparison.person_sha256_positive.eq(comparison.person_sha256_net).all()
            and comparison.meal_sha256_positive.eq(comparison.meal_sha256_net).all(), 'Positive/net memberships differ')
    comparison['coefficient_change'] = comparison.estimate_positive - comparison.estimate_net
    comparison['interpretation'] = 'Descriptive coefficient change, not a tested contrast; q belongs to separate families'
    run.table(comparison, '05_positive_net_comparison.csv')


def summary(run, state):
    text = '''# 营养调整、正增量与整体CGM衔接

数据：**{kind}**；状态：**{status}**。
原模型复现：{reproduced}/84。扩展规格估计：{estimated}/302。
{reason}

本轮读取原始主样本及联合质量样本。七项评分方向和原Z单位不变。
背景、平均餐能量、餐碳水线性、两者、再加餐碳水平方，使用完全相同的各评分人群与餐次。
个人日能量与平均餐能量是不同变量；EAT13、NOVA4、碳水评分的背景本来就含个人日能量。

正增量=positive iAUC/120；净增量=net iAUC/120，单位均为mg/dL。正增量用于检查负面积抵消；它不是原指标的独立验证。
A为每人等权的个人平均反应；B为个人固定效应、每人总权重1的餐碳水交互，单位mg/dL/10g/饮食Z，不能把交互当作碳水总斜率。
正面积缺失时保留固定样本和未估计状态，不自动删餐或补值；原输入数值不合法则停止。

整体CGM衔接不按饮食评分完整性选人；同访视同连接且至少5餐/3日。三项餐反应分别与全部20项整体原Z表型回归。
衔接模型只调整原BASE背景，餐特征按每个配对样本的1SD表达；另列未经调整的Spearman相关。
整体和餐次血糖共享读数，不能当作外部验证或菌群机制验证。SDw为日内SD的跨日平均；Mean/GMI冗余不算独立发现。

每套样本分别对调整70项、正增量21项、衔接60项做BH校正；未估计项保留计划分母。区间为逐项95%，不是同时区间。
不要按q跨0.05比较两模型差异；系数变化不是中介比例。影响诊断只标记，不自动删除参与者。
COMPLETE表示计算完成，不代表有显著或有意义的发现。合成验证不包含真实研究结果。

截图：运行本包 --show latest --page 1 查看目录，再按编号逐页查看。页内包含全部七项或全部20项的分段，不按显著性排序。
tables/ 为可汇总表，figures/ 为静态图；private/ 含个体成员/特征，manifest.json 含路径及哈希，保留在服务器。
'''.format(kind=run.c['data_kind'], status=state['status'], reproduced=state.get('reproduced', 0),
           estimated=state.get('estimated_models', 0), reason=state.get('reason', ''))
    (run.out / 'SUMMARY.md').write_text(text, encoding='utf-8')


def pages(run, result, state):
    sections = []
    for outcome in ['mean_increment', 'peak_rise_0_120']:
        for setting in SETTINGS:
            t = result['adjustment'].loc[lambda x: x.setting.eq(setting) & x.outcome.eq(outcome)]
            p = t.pivot(index='diet', columns='model', values='estimate').reindex(DIETS)
            p = p.reindex(columns=[s for s, _ in STEPS]).rename(columns={'background': 'BG', 'plus_energy': '+E',
                'plus_carb': '+C', 'plus_both': '+E+C', 'meal_adjusted': '+E+C+C2'})
            final = t.loc[t.model.eq('meal_adjusted')].set_index('diet').reindex(DIETS)
            p['N'] = final.people
            p['q_full'] = final.q
            p['status'] = final.status
            sections.append(('调整 | {} | {}'.format(setting, outcome),
                '每饮食原Z单位的mg/dL；固定成员。E=平均餐能量，C=平均餐碳水，C2=餐碳水平方的平均值。\n'
                'q_full来自每样本70项家族；其他步骤完整区间/q见表04。\n' + formatted(p.reset_index())))
    for setting in SETTINGS:
        for branch, model in [('A', 'background'), ('A', 'meal_adjusted'), ('B', 'carb_interaction')]:
            t = ordered(result['positive'].loc[lambda x: x.setting.eq(setting) & x.branch.eq(branch) & x.model.eq(model)])
            o = result['original'].loc[lambda x: x.setting.eq(setting) & x.branch.eq(branch) & x.model.eq(model) & x.outcome.eq('mean_increment')]
            t = t.merge(o[['diet', 'estimate']].rename(columns={'estimate': 'net_est'}), on='diet', validate='one_to_one')
            sections.append(('正增量 | {} | {} {}'.format(setting, branch, model),
                'A: mg/dL/饮食Z；B: mg/dL/10g/饮食Z。net_est为原净增量系数；此处q来自新21项家族。\n' +
                formatted(t[['diet', 'people', 'net_est', 'estimate', 'ci_low', 'ci_high', 'q', 'status']])))
    for setting in SETTINGS:
        t = ordered(result['adjustment'].loc[lambda x: x.setting.eq(setting) & x.model.eq('meal_adjusted') & x.outcome.eq('mean_increment')])
        names = ['diet', 'people', 'score_residual_variance_fraction', 'score_vif', 'max_leverage', 'max_cook', 'max_abs_leave1_coefficient_change']
        t = t.reindex(columns=names).rename(columns={'score_residual_variance_fraction': 'score_remain', 'score_vif': 'VIF',
            'max_leverage': 'max_hat', 'max_cook': 'max_Cook', 'max_abs_leave1_coefficient_change': 'max_dBeta'})
        sections.append(('诊断 | {} | 净增量完整调整'.format(setting),
            'score_remain=控制其他变量后评分剩余方差比例；max_dBeta=删除单人所致最大系数变化。\n不据此自动删人。全部步骤/结局诊断见表04，正增量诊断见表05。\n' + formatted(t)))
    for setting in SETTINGS:
        t = result['bridge'].loc[lambda x: x.setting.eq(setting) & x.cgm.isin(['Mean', 'CV', 'MODD', 'SDw'])].copy()
        t['meal'] = t.meal_measure.map({'mean_increment': 'net', 'peak_rise_0_120': 'peak', 'positive_mean': 'positive'})
        sections.append(('衔接重点 | ' + setting,
            '系数: 整体CGM原Z/每1SD个人餐特征；背景调整。rho为未调整Spearman；q家族60。\n' +
            formatted(t.rename(columns={'spearman': 'rho'})[['meal', 'cgm', 'people', 'estimate', 'ci_low', 'ci_high', 'q', 'rho']])))
    for setting in SETTINGS:
        for measure in MEASURES:
            t = result['bridge'].loc[lambda x: x.setting.eq(setting) & x.meal_measure.eq(measure)]
            for half in [0, 1]:
                sections.append(('衔接全部20项 | {} | {} | {}/2'.format(setting, measure, half + 1),
                    '单位=整体CGM原Z/1SD个人餐特征；原始rho未经背景调整；共享读数，非独立验证。\n' +
                    formatted(t.iloc[half * 10:(half + 1) * 10].rename(columns={'spearman': 'rho'})[
                        ['cgm', 'people', 'estimate', 'ci_low', 'ci_high', 'q', 'rho', 'status']])))
    for setting in SETTINGS:
        flow = result['bridge_flow'].loc[lambda x: x.setting.eq(setting)]
        d = result['distribution'].loc[lambda x: x.setting.eq(setting) & x.branch.eq('A') & x.level.eq('person_equal_weight') & x.field.eq('positive_minus_net')]
        sections.append(('样本与面积抵消 | ' + setting, formatted(flow[['stage', 'people', 'meals']]) +
            '\n个人正增量减净增量的分布（mg/dL，每人等权）：\n' + formatted(d[['diet', 'valid', 'missing', 'mean', 'median', 'p90', 'p99']])))
    failed = []
    for key in ['adjustment', 'positive', 'bridge']:
        t = result[key].loc[lambda x: ~x.status.eq('ESTIMATED')]
        if len(t):
            failed.append(t.assign(analysis=key).groupby(['analysis', 'setting', 'reason'], dropna=False).size().rename('specs').reset_index())
    failure_text = formatted(pd.concat(failed, ignore_index=True)) if failed else '所有计划规格均成功估计。'
    sections.append(('未估计原因', failure_text))
    for setting in SETTINGS:
        for outcome in ['mean_increment', 'peak_rise_0_120']:
            for model, _ in STEPS:
                t = ordered(result['adjustment'].loc[lambda x: x.setting.eq(setting) & x.outcome.eq(outcome) & x.model.eq(model)])
                sections.append(('调整详细区间 | {} | {} | {}'.format(setting, outcome, model),
                    '单位=mg/dL/原饮食Z；各步骤固定同一批人和餐；q来自70项家族，区间为逐项95%。\n' +
                    formatted(t[['diet', 'people', 'estimate', 'ci_low', 'ci_high', 'q', 'status']])))
    for setting in SETTINGS:
        t = result['correlations'].loc[lambda x: x.setting.eq(setting)].copy()
        t['field'] = t.field.map({'carb_10': 'C', 'carb_10_squared': 'C2', 'energy_100': 'E'})
        t = t.rename(columns={'background_residual_pearson': 'resid_r', 'pearson': 'raw_r', 'spearman': 'raw_rho'})
        sections.append(('评分与餐营养相关 | ' + setting,
            'raw为个人平均值的原始相关；resid_r为评分和餐营养分别扣除背景后残差相关。\n' +
            formatted(t[['diet', 'field', 'people', 'raw_r', 'raw_rho', 'resid_r']])))
    for setting in SETTINGS:
        t = result['distribution'].loc[lambda x: x.setting.eq(setting) & x.branch.eq('A') & x.level.eq('person_equal_weight') & x.field.eq('positive_mean')]
        sections.append(('正增量个人分布 | ' + setting,
            'mg/dL，每人等权；任一合格餐正面积缺失则该人平均正增量记缺失。\n' +
            formatted(t[['diet', 'valid', 'missing', 'zero_fraction', 'mean', 'sd', 'median', 'p90', 'p99']])))
    overview = ('DATA={} | STATUS={}\n复现={}/84；扩展估计={}/302。\n'
        '页码目录（每次只查看一页，按固定顺序，未按显著性挑选）：\n'
        '2-3: 净增量分步调整；4-5: 峰值分步调整\n'
        '6-8: 原样本正增量A背景/A完整/B交互；9-11: joint对应结果\n'
        '12-13: 原/joint影响与评分剩余变异诊断\n'
        '14-15: 原/joint的Mean、CV、MODD、SDw衔接\n'
        '16-21: 原样本全部20项（净/峰/正，各两页）\n'
        '22-27: joint全部20项（净/峰/正，各两页）\n'
        '28-29: 样本损失及正净面积差；30: 未估计原因\n'
        '31-40: 原样本各调整模型完整区间/q（净5页、峰5页）\n'
        '41-50: joint各调整模型完整区间/q（净5页、峰5页）\n'
        '51-52: 评分与餐营养相关；53-54: 正增量零值及分布\n'
        '先截图第2、3、6、7、9、10、14、15页，再看其余页。\n'
        '新家族q：调整70、正增量21、衔接60；逐项95%区间。\n'
        '原/joint是重叠样本。整体与餐次共享CGM，不是独立验证。\n'
        'COMPLETE仅表示计算完成，不保证有新结论。').format(run.c['data_kind'], state['status'], state['reproduced'], state['estimated_models'])
    sections.insert(0, ('运行与目录', overview))
    require(len(sections) == 54, 'Page directory must be updated')
    folder = run.out / 'pages'
    folder.mkdir(exist_ok=True)
    for i, (title, body) in enumerate(sections, 1):
        (folder / '{:02d}.txt'.format(i)).write_text('PAGE {}/54 | {}\nRUN={}\nDATA={}\n\n{}\n'.format(
            i, title, run.out.name, run.c['data_kind'], body), encoding='utf-8')


def figures(run, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for outcome in ['mean_increment', 'peak_rise_0_120']:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True)
        for ax, setting in zip(axes, SETTINGS):
            for diet in DIETS:
                d = result['adjustment'].loc[lambda x: x.setting.eq(setting) & x.outcome.eq(outcome) & x.diet.eq(diet)]
                d = d.set_index('model').reindex([s for s, _ in STEPS])
                ax.plot(range(5), d.estimate, '-o', label=diet, linewidth=1.2, markersize=3)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set_xticks(range(5))
            ax.set_xticklabels(['BG', '+E', '+C', '+E+C', '+E+C+C2'])
            ax.set_title(setting)
            ax.set_ylabel('mg/dL per original diet Z')
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
        fig.suptitle('{} | {} | fixed samples; estimates only, CIs in tables'.format(run.c['data_kind'], outcome))
        fig.tight_layout(rect=[0, .1, 1, 1])
        fig.savefig(run.out / 'figures' / (outcome + '_adjustment.png'), dpi=160)
        plt.close(fig)


def show(target, output_root, page):
    if target == 'latest':
        candidates = []
        for f in Path(output_root).expanduser().glob('nutrition_bridge_*/manifest.json'):
            m = json.loads(f.read_text(encoding='utf-8'))
            if m.get('status', {}).get('nutrition_bridge', {}).get('status') in ['COMPLETE', 'COMPLETE_WITH_LIMITATIONS']:
                candidates.append(f.parent)
        require(candidates, 'No completed nutrition_bridge run found in ' + str(output_root))
        path = sorted(candidates)[-1]
    else:
        path = Path(target).expanduser().resolve()
    if page == 0:
        print((path / 'SUMMARY.md').read_text(encoding='utf-8'))
        return
    require(1 <= page <= 54, 'Page must be 1..54 (0 for full summary)')
    rel = 'pages/{:02d}.txt'.format(page)
    m = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    require(m['status']['nutrition_bridge'].get('result_sha256', {}).get(rel) == sha256(path / rel), 'Page changed after run')
    print((path / rel).read_text(encoding='utf-8'))
