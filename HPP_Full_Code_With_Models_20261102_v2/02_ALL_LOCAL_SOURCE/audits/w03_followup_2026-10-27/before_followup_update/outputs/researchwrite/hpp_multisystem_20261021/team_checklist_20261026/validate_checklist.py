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
        ('audits/w05_nutrient_semantics_2026-10-26/background_append_manifest.json', '2026-10-26 W05实际核查')]
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
