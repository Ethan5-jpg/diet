"""Validate the execution document, not scientific data or statistical results."""
import hashlib
import json
import pathlib
import re
from html.parser import HTMLParser
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parent
MD = ROOT / 'reports/HPP_Team_Execution_Checklist_2026-10-26.md'
HTML = MD.with_suffix('.html')

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.duplicates, self.scripts = set(), [], []
        self.current_script = None
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            if a['id'] in self.ids:
                self.duplicates.append(a['id'])
            self.ids.add(a['id'])
        if tag == 'script':
            self.current_script = ''
            if a.get('src'):
                self.scripts.append({'external': a['src']})
    def handle_data(self, data):
        if self.current_script is not None:
            self.current_script += data
    def handle_endtag(self, tag):
        if tag == 'script' and self.current_script is not None:
            self.scripts.append({'inline': self.current_script})
            self.current_script = None

def main():
    text = MD.read_text()
    idx = json.loads((RUN / 'task_index.json').read_text())
    hb = json.loads((RUN / 'html_build.json').read_text())
    original = json.loads((RUN / 'input_manifest.json').read_text())
    tasks = idx['tasks']
    ids = [t['id'] for t in tasks]
    missing_fields = {}
    for t in tasks:
        keys = ''.join(t['fields'])
        required = ('主责', '复核', '前置', '输入', '动作', '交付', '验收', '结局', '状态')
        missing = [k for k in required if k not in keys]
        if missing:
            missing_fields[t['id']] = missing
    links = re.findall(r'\]\((?:<([^>]+)>|([^\s)]+))\)', text)
    broken_links = []
    for a, b in links:
        target = unquote(a or b)
        if target.startswith(('http:', 'https:', '#', 'mailto:')):
            continue
        target = target.split('#')[0]
        p = pathlib.Path(target)
        p = p if p.is_absolute() else MD.parent / p
        if target and not p.exists():
            broken_links.append(target)
    previous = (ROOT / 'reports/HPP_Project_TODO_v3_2026-10-23.md').read_text()
    old_t = sorted(set(re.findall(r'\bT\d{2}\b', previous)))
    new_t = sorted(set(re.findall(r'\bT\d{2}\b', text)))
    page = Page()
    page.feed(HTML.read_text())
    inline = '\n'.join(s['inline'] for s in page.scripts if 'inline' in s)
    (RUN / 'reading_page_inline.js').write_text(inline)
    unchanged = {}
    documented_appends = {}
    append_steps = [
        ('audits/w02_source_inventory_2026-10-26/background_append_manifest.json', '2026-10-26 W02实际执行更新'),
        ('audits/w03_identity_join_2026-10-26/background_append_manifest.json', '2026-10-26 W03实际核查'),
        ('audits/w05_nutrient_semantics_2026-10-26/background_append_manifest.json', '2026-10-26 W05实际核查'),
        ('audits/w03_followup_2026-10-27/background_append_manifest.json', '2026-10-27 W03肝表补核与用户规则澄清'),
        ('audits/w04_mapping_review_2026-10-27/background_append_manifest.json', '2026-10-27 W04映射与有效日核查'),
        ('audits/w06_food_cleaning_2026-10-28/background_append_manifest.json', '2026-10-28 W06食物事件与记录日清洗'),
        ('audits/w07_w08_diet_views_2026-10-28/background_append_manifest.json', '2026-10-28 W07与W08饮食特征视图'),
        ('audits/w09_cgm_point_qc_2026-10-28/background_append_manifest.json', '2026-10-28 W09点级QC与W08共同日补齐'),
        ('audits/w10_outcome_qc_2026-10-28/background_append_manifest.json', '2026-10-28 W10结局字段与基本QC'),
        ('audits/w11_w12_covariates_samples_2026-10-28/background_append_manifest.json', '2026-10-29 W11协变量与W12候选样本')]
    chains = []
    for filename, marker in append_steps:
        p = ROOT/filename
        records = {x['path']: x for x in json.loads(p.read_text())['records']} if p.exists() else {}
        chains.append((records, marker))
    for source in original['sources']:
        filename = source['path']
        if filename in ('reports/HPP_CURRENT_PLAN.md', 'HPP_PROJECT_BACKGROUND.md'):
            continue
        current = (ROOT/filename).read_bytes()
        preserved = hashlib.sha256(current).hexdigest() == source['sha256']
        if not preserved and filename in ('HPP_SERVER_BACKGROUND.md', 'HPP_DATA_BACKGROUND.md'):
            steps = [(records[filename], marker) for records, marker in chains if filename in records]
            ok = bool(steps)
            prev_hash = source['sha256']; prev_bytes = source['bytes']
            for i, (record, marker) in enumerate(steps):
                snapshot = ROOT/record['snapshot']
                end = steps[i+1][0]['bytes'] if i+1 < len(steps) else len(current)
                ok = (ok and record.get('append_only') is True
                      and record['sha256'] == prev_hash and record['bytes'] == prev_bytes
                      and snapshot.is_file() and snapshot.read_bytes() == current[:record['bytes']]
                      and hashlib.sha256(current[:record['bytes']]).hexdigest() == record['sha256']
                      and end > record['bytes']
                      and marker.encode() in current[record['bytes']:end]
                      and hashlib.sha256(current[:end]).hexdigest() == record['after_sha256'])
                prev_hash = record['after_sha256']; prev_bytes = end
            preserved = ok
            documented_appends[filename] = preserved
        unchanged[filename] = preserved
    checks = {
        'date': '2026-10-26', 'task_count': len(tasks), 'contiguous_task_ids': ids == [f'W{i:02}' for i in range(1, len(ids)+1)],
        'missing_task_fields': missing_fields, 'unknown_task_references': sorted(set(re.findall(r'\bW\d{2}\b', text)) - set(ids)),
        'prior_T_task_count': len(old_t), 'prior_T_tasks_missing_crosswalk': sorted(set(old_t)-set(new_t)),
        'unknown_T_task_ids': sorted(set(new_t)-set(old_t)),
        'broken_local_links': broken_links, 'html_unique_ids': not page.duplicates, 'html_duplicate_ids': page.duplicates,
        'all_tasks_in_html': all(x in page.ids for x in ids), 'html_source_hash_matches': hb['source_sha256'] == hashlib.sha256(MD.read_bytes()).hexdigest(),
        'html_no_external_script': not any('external' in s for s in page.scripts),
        'core_sources_preserved_or_documented_append': unchanged,
        'dated_background_append_verified': documented_appends, 'review_log_exists': (RUN/'review.md').exists(),
        'scientific_execution_performed': False, 'browser_visual_qa_performed': False,
        'scope': 'Document structure, protected scientific sources and dated background-append preservation; W02 metadata execution evidence is distinct from scientific data QC.'}
    progress_file = RUN/'workpackage_progress.json'
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        checked = re.findall(r'^- \[[xX]\] \*\*(W\d{2})', text, re.M)
        expected_checked = sorted(k for k, v in progress['tasks'].items() if v.get('checked'))
        required_checked = ['W02', 'W03'] + (['W04'] if progress['tasks'].get('W04', {}).get('checked') else []) + (['W05'] if progress['tasks'].get('W05', {}).get('checked') else [])
        required_checked += ['W06'] if progress['tasks'].get('W06', {}).get('checked') else []
        required_checked += ['W07'] if progress['tasks'].get('W07', {}).get('checked') else []
        required_checked += ['W08'] if progress['tasks'].get('W08', {}).get('checked') else []
        required_checked += ['W09'] if progress['tasks'].get('W09', {}).get('checked') else []
        required_checked += ['W10'] if progress['tasks'].get('W10', {}).get('checked') else []
        required_checked += ['W11'] if progress['tasks'].get('W11', {}).get('checked') else []
        required_checked += ['W12'] if progress['tasks'].get('W12', {}).get('checked') else []
        checks['execution_progress_consistent'] = (checked == expected_checked and checked == required_checked
                                                    and not progress['tasks']['W01']['checked']
                                                    and all(t['fields']['状态'].startswith('2026-10-26元信息') if t['id']=='W02' else True for t in tasks)
                                                    and (ROOT/progress['evidence']).is_file())
        if 'W04' in checked:
            w04_path = ROOT/'outputs/w04_diet_definition_20261026/delivery_validation.json'
            w04 = json.loads(w04_path.read_text())
            checks['W04_definition_evidence_present'] = (w04['passed'] and not w04['participant_features_computed']
                and not w04['formal_protocol_frozen']
                and all((ROOT/'config/w04_diet_features_20261026'/name).is_file()
                        for name in ('diet_feature_spec_draft.yaml','prior_work_overlap.md','decision_log.md')))
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W04_definition_evidence_present']
        if 'W05' in checked:
            w05 = json.loads((ROOT/'audits/w05_nutrient_semantics_2026-10-26/delivery_validation.json').read_text())
            checks['W05_semantics_delivery_verified'] = (w05['passed'] and w05['main_actual_exit'] == 0
                and w05['independent_source_examples'] == 12 and w05['event_rows_scanned'] == 9125981
                and not w05['formal_features_computed'] and not w05['formal_protocol_frozen'])
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W05_semantics_delivery_verified']
        if 'W06' in checked:
            w06 = json.loads((ROOT/'audits/w06_food_cleaning_2026-10-28/delivery_validation.json').read_text())
            checks['W06_cleaning_delivery_verified'] = (w06['passed']
                and w06['actual_exit_codes'] == {'01': 0, '02': 0, '03': 0}
                and w06['processed_rows'] == 2640959 and w06['clean_rows'] == 2420902
                and w06['day_rows'] == 164625 and w06['source_examples'] == 12
                and w06['mirrors'] == 5 and not w06['outcomes_read']
                and (ROOT/'audits/w06_food_cleaning_2026-10-28/03_independent_and_exit_codes.png').is_file())
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W06_cleaning_delivery_verified']
        if 'W07' in checked:
            w78 = json.loads((ROOT/'audits/w07_w08_diet_views_2026-10-28/delivery_validation.json').read_text())
            checks['W07_and_W08_food_only_delivery_verified'] = (w78['passed']
                and w78['build_actual_exit'] == 0 and w78['independent_verify_actual_exit'] == 0
                and w78['crosswalk_rows'] == 7770 and w78['daily_rows'] == 149912
                and w78['window_rows'] == 12085 and w78['source_days_verified'] == 3
                and w78['admitted_event_rows_reread'] == 2420902 and w78['mirrors_verified'] == 7
                and w78['W08_food_only_complete'] and not w78['W08_real_CGM_common_complete']
                and not w78['outcomes_read'])
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W07_and_W08_food_only_delivery_verified']
        if 'W08' in checked or 'W09' in checked:
            w98 = json.loads((ROOT/'audits/w09_cgm_point_qc_2026-10-28/delivery_validation.json').read_text())
            checks['W09_full_and_W08_common_delivery_verified'] = (w98['passed']
                and w98['W09_verified'] and w98['W08_real_common_verified']
                and w98['fragments'] == w98['independent_fragment_hashes'] == 9497
                and all(v == 0 for v in w98['actual_exit_codes'].values())
                and not w98['raw_individual_exported'] and not w98['primary_protocol_frozen']
                and progress['tasks']['W08']['checked'] and progress['tasks']['W09']['checked'])
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W09_full_and_W08_common_delivery_verified']
        if 'W10' in checked:
            w10 = json.loads((ROOT/'audits/w10_outcome_qc_2026-10-28/delivery_validation.json').read_text())
            checks['W10_fields_basic_QC_delivery_verified'] = (w10['passed']
                and w10['independent_groups_recomputed'] == w10['registry_rows'] == 84962
                and w10['measurement_QC_rows'] == 192434
                and w10['static_outcomes'] == 9 and w10['liver_outcomes'] == 6
                and w10['binary_mirrors'] == 3 and w10['build_hashes_verified'] == 8
                and all(v == 0 for v in w10['actual_exit_codes'].values())
                and w10['initial_failed_exit_retained'] == 1
                and w10['per_outcome_eligibility'] and w10['CGM_recipe_registered']
                and not w10['CGM_person_values_materialized']
                and not w10['manufacturer_QC_claimed']
                and not w10['raw_individual_exported'] and not w10['primary_protocol_frozen']
                and (ROOT/w10['evidence']).is_file())
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W10_fields_basic_QC_delivery_verified']
        if 'W11' in checked or 'W12' in checked:
            w112 = json.loads((ROOT/'audits/w11_w12_covariates_samples_2026-10-28/delivery_validation.json').read_text())
            checks['W11_W12_covariates_candidates_delivery_verified'] = (w112['passed']
                and w112['covariate_rows'] == 20407 and w112['candidate_rows'] == 20547
                and w112['artifact_hashes_verified'] == 13 and w112['prior_revisions_verified'] == 4
                and w112['binary_mirrors'] == 4 and w112['full_lag_coverage_verified']
                and w112['union_not_complete_case']
                and len(w112['actual_exit_codes']) == 12
                and all(v == 0 for v in w112['actual_exit_codes'].values())
                and not w112['W13_technical_QC_done'] and w112['final_samples_selected'] == 0
                and not w112['raw_individual_exported'] and not w112['primary_protocol_frozen']
                and not w112['team_human_review_claimed']
                and all((ROOT/f).is_file() for f in w112['evidence'])
                and progress['tasks']['W11']['checked'] and progress['tasks']['W12']['checked'])
            checks['execution_progress_consistent'] = checks['execution_progress_consistent'] and checks['W11_W12_covariates_candidates_delivery_verified']
    assignment_file = RUN / 'team_assignment.json'
    if assignment_file.exists():
        assignment = json.loads(assignment_file.read_text())
        by_id = {t['id']: t for t in tasks}
        checks['approved_workstream_assignment'] = assignment['status'] == 'user_approved_scientific_workstream_assignment'
        checks['cgm_qc_owned_by_user'] = by_id['W09']['fields']['主责'].startswith('甲')
        checks['daily_model_owned_by_user'] = by_id['W24']['fields']['主责'].startswith('甲')
        checks['human_level_models_assigned_to_both_lines'] = all(all(x in by_id[k]['fields']['主责'] for x in ('甲', '乙')) for k in ('W19', 'W20', 'W23'))
        checks['conditional_meal_extension_documented'] = all(x in text for x in ('逐餐／分钟级曲线模型', '条件性扩展', 'App记录时间不能直接当进食时间'))
        checks['workstream_assignment_consistent'] = all(checks[k] for k in ('approved_workstream_assignment', 'cgm_qc_owned_by_user', 'daily_model_owned_by_user', 'human_level_models_assigned_to_both_lines', 'conditional_meal_extension_documented'))
    checks['passed'] = all((checks['contiguous_task_ids'], not missing_fields, not checks['unknown_task_references'],
                            not checks['prior_T_tasks_missing_crosswalk'], not checks['unknown_T_task_ids'], not broken_links,
                            checks['html_unique_ids'], checks['all_tasks_in_html'], checks['html_source_hash_matches'],
                            checks['html_no_external_script'], all(unchanged.values()), checks['review_log_exists'], checks.get('workstream_assignment_consistent', True), checks.get('execution_progress_consistent', True)))
    (RUN/'validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks['passed']:
        raise SystemExit('Document validation failed')

if __name__ == '__main__':
    main()
