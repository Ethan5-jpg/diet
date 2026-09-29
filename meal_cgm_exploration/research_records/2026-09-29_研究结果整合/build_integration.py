#!/usr/bin/env python3
"""Build a local manuscript evidence packet. No model fitting or new inference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIETS = ['AHEI', 'AMED', 'hPDI', 'rEDIH', 'EAT13', 'NOVA4', 'Carbohydrate_pct']
LABELS = DIETS[:-1] + ['Carbohydrate %']
SETTINGS = ['source_primary', 'joint']
OUTCOMES = ['mean_increment', 'peak_rise_0_120', 'positive_mean']
COLORS = ['#246C96', '#BF5B35']
SRC = {
    'nutrition': 'nutrition_bridge_累计已展示结果_截图精度.json',
    'quality': 'quality_timecourse_累计133项已展示结果_截图精度.json',
    'contrast': '2026-09-28_response_followup_三份摘要_截图精度.json',
    'date': '2026-09-28_date_split_bridge_三份摘要_截图精度.json',
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def one(rows, **criteria):
    found = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    assert len(found) == 1, (criteria, len(found))
    return found[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    base, out = args.records.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    figs = out / 'figures'; figs.mkdir()
    sources = {k: json.loads((base / f).read_text()) for k, f in SRC.items()}
    assert all(d['data_kind'] == 'REAL' for d in sources.values())
    assert [len(sources[k]['rows']) for k in SRC] == [302, 133, 42, 24]
    n, q, b, d = [sources[k]['rows'] for k in SRC]
    refs = {id(r): {'source_file': SRC[k], 'row_index': i, 'run_id': sources[k]['run_id']}
            for k, doc in sources.items() for i, r in enumerate(doc['rows'])}
    a = [r for r in n if r.get('branch') == 'A' and r.get('model') in ['background', 'meal_adjusted']]
    t = [r for r in q if r['analysis'] == 'timecourse']
    assert len(a) == 84 and len(t) == 42
    for rows in [a, b, t]:
        for r in rows:
            assert r['ci_low'] <= r['estimate'] <= r['ci_high']
            assert 0 <= r['q'] <= 1 and r['people'] > 0
    assert sum(r['q24_below_005'] for r in d) == 24
    variability = [r for r in d if r['cgm'] != 'Mean']
    assert len(variability) == 18
    assert all(r['delta_estimate'] < 0 and r['qD24_below_005'] for r in variability)
    assert sum(r['qD24_below_005'] for r in d) == 20
    # Include every diet and outcome; no filtering by significance.
    for setting in SETTINGS:
        for diet in DIETS:
            for outcome in OUTCOMES:
                for model in ['background', 'meal_adjusted']:
                    one(a, setting=setting, diet=diet, outcome=outcome, model=model)
                one(b, setting=setting, diet=diet, outcome=outcome)
            for outcome in ['early_mean', 'late_mean', 'late_minus_early']:
                one(t, setting=setting, diet=diet, outcome=outcome)
    evidence = dict(scope='Screenshot precision; existing REAL results only; not new independent model counts',
                    dietary_A=[dict(provenance=refs[id(r)], result=r) for r in a],
                    within_person_B_contrasts=[dict(provenance=refs[id(r)], result=r) for r in b],
                    contrast_support=sources['contrast']['support_rows'],
                    timecourse=[dict(provenance=refs[id(r)], result=r) for r in t],
                    date_split=[dict(provenance=refs[id(r)], result=r) for r in d])
    write_json(out / 'evidence_rows.json', evidence)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.2))
    used = []
    for k, ax in enumerate(axes.flat):
        if k < 2:
            outcome = OUTCOMES[k]
            for j, setting in enumerate(SETTINGS):
                for m, model in enumerate(['background', 'meal_adjusted']):
                    for i, diet in enumerate(DIETS):
                        r = one(a, setting=setting, diet=diet, outcome=outcome, model=model)
                        used.append({'panel': 'AB'[k], **refs[id(r)]})
                        y = i + [-.27, -.09, .09, .27][j * 2 + m]
                        ax.errorbar(r['estimate'], y,
                            xerr=[[r['estimate'] - r['ci_low']], [r['ci_high'] - r['estimate']]],
                            fmt=['^', 'o'][m], color=COLORS[j],
                            mfc=COLORS[j] if r['q'] < .05 else 'white', ms=5.7, lw=1.15, capsize=2)
            ax.set_xlim(-1.55, 2.65)
            ax.set_xlabel('β (mg/dL per original dietary score Z)', labelpad=9)
            ax.set_title(f"{'AB'[k]}   {'Net increment' if k == 0 else 'Peak rise'}: between-person association",
                         loc='left', fontweight='bold', fontsize=11, pad=14)
        else:
            outcome = OUTCOMES[k - 2]
            for j, setting in enumerate(SETTINGS):
                for i, diet in enumerate(DIETS):
                    r = one(b, setting=setting, diet=diet, outcome=outcome)
                    used.append({'panel': 'CD'[k - 2], **refs[id(r)]})
                    ax.errorbar(r['estimate'], i + [-.13, .13][j],
                        xerr=[[r['estimate'] - r['ci_low']], [r['ci_high'] - r['estimate']]],
                        fmt='D', color=COLORS[j], mfc=COLORS[j] if r['q'] < .05 else 'white',
                        ms=5.5, lw=1.2, capsize=2)
            ax.set_xlim(-.56, .24)
            ax.set_xlabel('D: difference in carbohydrate-related response changes (mg/dL)', fontsize=9, labelpad=9)
            ax.set_title(f"{'CD'[k - 2]}   {'Net increment' if k == 2 else 'Peak rise'}: carbohydrate interaction",
                         loc='left', fontweight='bold', fontsize=11, pad=14)
        ax.set_yticks(range(7), LABELS)
        ax.set_ylim(6.6, -.6)
        ax.axvline(0, lw=.9, ls='--', color='#7B858D')
        ax.grid(axis='x', color='#DFE5E9', lw=.65)
        ax.set_axisbelow(True)
        for i in [0, 2, 4, 6]:
            ax.axhspan(i - .47, i + .47, color='#F3F5F7', zorder=-5)
    fig.suptitle('Dietary scores relate to distinct aspects of meal responses',
                 x=.06, y=.98, ha='left', fontsize=18, fontweight='bold')
    fig.text(.06, .948, 'All seven scores retained • Original and joint samples overlap • Pointwise 95% confidence intervals',
             fontsize=10, color='#52616A')
    leg = [Line2D([], [], color=COLORS[j], lw=2, label=s) for j, s in enumerate(['Original', 'Joint'])]
    leg += [Line2D([], [], color='#4A555E', marker=m, ls='', label=s) for m, s in [('^', 'Background (A–B)'), ('o', 'Full adjustment (A–B)'), ('D', 'Contrast D (C–D)')]]
    fig.legend(handles=leg, loc='upper left', bbox_to_anchor=(.057, .929), ncol=5, frameon=False, fontsize=9.5)
    fig.text(.06, .072, 'Filled: q < 0.05; open: q ≥ 0.05. A–B: original 70-test family per sample. C–D: original 14-test families.', fontsize=9)
    fig.text(.06, .051, 'D compares carbohydrate-associated changes at score Q75 versus Q25; carbohydrate anchors vary by score and sample.', fontsize=9)
    fig.text(.06, .030, 'D is neither an absolute between-group glucose difference nor the total carbohydrate slope. Do not rank scores by D.', fontsize=9)
    fig.subplots_adjust(left=.145, right=.978, top=.866, bottom=.13, hspace=.40, wspace=.39)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(figs / ('MainFigure1_diet_response.' + ext), dpi=210, facecolor='white')
    plt.close(fig)
    copies = {}
    for prefix, dest in [('2026-09-28_日期分离结果图/Figure4_date_split', 'MainFigure2_date_split'),
                         ('2026-09-27_结果图表/Supplement1_AMED_timecourse', 'Supplement_AMED_timecourse')]:
        for ext in ['png', 'pdf', 'svg']:
            source, target = base / (prefix + '.' + ext), figs / (dest + '.' + ext)
            shutil.copyfile(source, target)
            assert sha(source) == sha(target)
            copies[str(target.relative_to(out))] = {'source': str(source.relative_to(base)), 'sha256': sha(source)}

    lines = ['# 完整数值表：全部评分与预定日期分离关联', '',
             '截图显示精度；不重算q、不补猜p。逐项95%CI与多重校正q同时保留。原/joint重叠。', '',
             'A、B、时间和日期分离回答不同问题，不能把本文件行数相加当作独立发现数。', '']
    for title, rows, kind in [('饮食评分与个人平均餐反应：背景及完整模型', a, 'A'),
                               ('个人内碳水交互：按观测参照换算', b, 'B'),
                               ('全部七项评分的时间分布', t, 'T')]:
        lines += ['## ' + title, '',
                  ('单位mg/dL；D定义及评分/碳水/能量参照见下表；沿用原q。' if kind == 'B' else '单位mg/dL/原评分Z；净/峰A使用每样本70项q家族，正增量A每样本21项，时间模型每样本21项。'), '',
                  '| 样本 | 评分 | 结局 | 模型 | N | 估计 [95%CI] | q |',
                  '|---|---|---|---|---:|---|---:|']
        for r in rows:
            ds = r['display']; e, lo, hi, qq = [ds[x] for x in ['estimate', 'ci_low', 'ci_high', 'q']]
            lines.append(f"| {r['setting']} | {r['diet']} | {r['outcome']} | {r.get('model', 'frozen_B_transform')} | {r['people']} | {e} [{lo}, {hi}] | {qq} |")
        lines.append('')
    lines += ['## 交互参照与观测支持', '',
              'D＝[评分Q75时高C−低C的模型反应差]−[评分Q25时相同C变化的模型反应差]。参照在每个评分和样本内分别确定。净/峰各沿用每样本14项q家族，正增量沿用每样本21项家族。', '',
              '| 样本 | 评分 | N | 评分差（原Z） | C低→高（g） | 共同能量范围（kcal） | 最小端点N | 最小同人跨端点N |',
              '|---|---|---:|---:|---|---|---:|---:|']
    for r in sources['contrast']['support_rows']:
        ds = r['display']
        lines.append(f"| {r['setting']} | {r['diet']} | {r['people']} | {ds['score_delta']} | {ds['carb_low']}→{ds['carb_high']} | {ds['energy_low']}–{ds['energy_high']} | {r['cellN_min']} | {r['crossN_min']} |")
    lines += ['', '## 日期分离：24项主关联及同人配对差值', '',
              '单位为新全期参考Y的SD/前窗X参考SD。q24合并两样本的24项主规格；qD24来自单独24项次要差值家族。全期CI/q未展示，不补填。', '',
              '| 样本 | 前窗X | 后窗Y | N | 后窗β [95%CI] | q24 | 同人全期β | 后窗−全期 [95%CI] | qD24 |',
              '|---|---|---|---:|---|---:|---:|---|---:|']
    for r in d:
        v = r['display']
        lines.append(f"| {r['setting']} | {v['X']} | {v['Y']} | {v['people']} | {v['late_estimate']} [{v['late_ci_low']}, {v['late_ci_high']}] | {v['q24']} | {v['full_estimate']} | {v['delta_estimate']} [{v['delta_ci_low']}, {v['delta_ci_high']}] | {v['qD24']} |")
    lines += ['', '`0(display)`为数值/显示限制，不是概率恰为0。所有行的原文件、行号、run及截图位置见evidence_rows.json。', '']
    (out / '完整数值表.md').write_text('\n'.join(lines))
    audit = dict(date='2026-09-29', data_kind='REAL_SCREENSHOT_PRECISION',
                 new_models=0, FDR_recomputed=False, generated_unique_figures=1, reused_unique_figures=2,
                 row_counts={'A_background_and_full': len(a), 'B_transformed': len(b), 'timecourse': len(t), 'date_split': len(d)},
                 all_seven_scores_retained=True, main_figure_1_marks=len(used),
                 date_primary_FDR_supported=24, variability_differences_FDR_supported=18,
                 all_paired_differences_FDR_supported=20,
                 original_source_files_unchanged=True,
                 visual_QA='PENDING', manuscript_QA='PENDING')
    write_json(out / 'QA.json', audit)
    prov = dict(created='2026-09-29', inputs={f: sha(base / f) for f in SRC.values()},
                script_sha256=sha(Path(__file__)), main_figure_1_rows=used,
                reused_artifacts=copies,
                notes=['No new estimates or FDR calculation.', 'Copied figures retain their historical internal numbering; manuscript labels are in captions.',
                       'Server full precision and raw data unavailable; this is a verified local synthesis of screenshot precision results.'])
    write_json(out / 'artifact_provenance.json', prov)
    print(json.dumps(audit, ensure_ascii=False))

if __name__ == '__main__':
    main()
