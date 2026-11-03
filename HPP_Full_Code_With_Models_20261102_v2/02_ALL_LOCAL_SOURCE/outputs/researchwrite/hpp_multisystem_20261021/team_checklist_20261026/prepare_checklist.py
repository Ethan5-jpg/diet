"""Build a reviewed Markdown execution companion and a read-only task index.

Run only after the parent has reviewed the Astra draft and review notes.
No TRE access, participant data, or scientific configuration mutation.
"""
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parent
DRAFT = ROOT / 'reports/Astra_Team_Execution_Checklist_Draft_2026-10-26.md'
FINAL = ROOT / 'reports/HPP_Team_Execution_Checklist_2026-10-26.md'

def task_cards(text):
    matches = list(re.finditer(r'^###\s+(W\d{2})[\s：:]+(.+)$', text, re.M))
    cards = []
    for m in matches:
        rest = text[m.end():]
        nxt = re.search(r'^#{1,3}\s+', rest, re.M)
        body = rest[:nxt.start()] if nxt else rest
        fields = {}
        for line in body.splitlines():
            clean = line.strip().lstrip('- ').replace('**', '')
            field = re.match(r'([^：:]{1,24})[：:]\s*(.*)', clean)
            if field:
                fields[field.group(1)] = field.group(2)
        cards.append({'id': m.group(1), 'title': m.group(2).strip(), 'fields': fields, 'source_text': body.strip()})
    return cards

def main():
    review = RUN / 'review.md'
    if not review.exists():
        raise SystemExit('Parent review.md must exist before delivery build')
    text = DRAFT.read_text()
    edits_path = RUN / 'review_edits.json'
    if edits_path.exists():
        for edit in json.loads(edits_path.read_text()).get('edits', []):
            if text.count(edit['old']) != 1:
                raise SystemExit('Review replacement must match exactly once: ' + edit.get('reason', ''))
            text = text.replace(edit['old'], edit['new'], 1)
    assignment_path = RUN / 'team_assignment.json'
    assignment = json.loads(assignment_path.read_text()) if assignment_path.exists() else None
    if assignment:
        text = re.sub(r'^建议分工：.+$',
                      '已确定分工：**成员甲（你）负责血糖—肝脏线及日级CGM，成员乙（同事）负责总脂—VAT线。**两线各自建队列、发现、确认及交付；公共底座、角色规则、全局封存与结果校正共同维护。此为用户确认后的Codex分工更新，不冒称新一轮Astra评审。团队工作分配已确定，参与者发现／确认名单仍待实际生成冻结。',
                      text, count=1, flags=re.M)
        for edit in assignment.get('text_replacements', []):
            if text.count(edit['old']) != 1:
                raise SystemExit('Assignment replacement must match exactly once: ' + edit.get('reason', ''))
            text = text.replace(edit['old'], edit['new'], 1)
        for task_id, replacements in assignment['task_overrides'].items():
            pattern = r'(^###\s+' + task_id + r'[\s：:]+[^\n]+\n)([\s\S]*?)(?=^#{1,3}\s+|\Z)'
            def replace_card(match):
                body = match.group(2)
                for field, value in replacements.items():
                    p = r'^' + re.escape(field) + '：.*$'
                    if len(re.findall(p, body, flags=re.M)) != 1:
                        raise SystemExit('Assignment field missing/duplicated: ' + task_id + '/' + field)
                    body = re.sub(p, lambda _: field + '：' + value, body, count=1, flags=re.M)
                return match.group(1) + body
            text, count = re.subn(pattern, replace_card, text, count=1, flags=re.M)
            if count != 1:
                raise SystemExit('Assignment task missing: ' + task_id)
    progress_path = RUN / 'workpackage_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'tasks': {}}
    for task_id, record in progress['tasks'].items():
        pattern = r'(^###\s+' + task_id + r'[\s：:]+[^\n]+\n)([\s\S]*?)(?=^#{1,3}\s+|\Z)'
        def apply_progress(match):
            body, count = re.subn(r'^状态：.*$', lambda _: '状态：' + record['status_text'], match.group(2), count=1, flags=re.M)
            if count != 1:
                raise SystemExit('Progress status field missing: ' + task_id)
            if record.get('action_override'):
                body, count = re.subn(r'^动作：.*$', lambda _: '动作：' + record['action_override'], body, count=1, flags=re.M)
                if count != 1:
                    raise SystemExit('Progress action field missing: ' + task_id)
            return match.group(1) + body
        text, count = re.subn(pattern, apply_progress, text, count=1, flags=re.M)
        if count != 1:
            raise SystemExit('Progress task missing: ' + task_id)
    if progress.get('verified_storage'):
        text = text.replace('本次只整理任务，没有访问 TRE、重算人数、开展新 QC、拟合模型或运行模拟。',
                            '初版仅整理任务；2026-10-26后续已访问TRE完成W02元信息盘点及W03身份／日期投影核查。未读取表型／丰度数值、生成最终合格N、训练模型或运行模拟。', 1)
        text = text.replace('所有新脚本名和产物名均为**拟写／拟产物**。',
                            '尚未执行工作包的脚本名和产物名为**拟写／拟产物**；已执行工作包的实际代码与产物见最新进度及总表。', 1)
        text = text.replace('项目根候选 `P=', '项目根 `P=', 1).replace('运行根候选 `L=', '运行根 `L=', 1)
        text = text.replace('P/L 尚不能当成已建成；L 也在 TRE 内，不在 Mac。W01 现场验证后才使用。',
                            'P/L 已于2026-10-26通过W02实际创建、写入及读回核验；L 在 TRE 内，不在 Mac。W01完整重跑已由用户免除，沿用同一服务器。', 1)
    text = re.sub(r'^# .+$', '# HPP 两人团队执行任务清单', text, count=1, flags=re.M)
    assignment_note = '工作线分配已由用户确认，参与者角色未冻结' if assignment else '责任分配为建议'
    provenance = ('\n> 2026-10-26。Astra（gpt-6-astra，xhigh／极高）起草，Codex逐条复核。'
                  '这是V3及饮食候选评估的执行配套；不是新数据结果或已冻结的新统计协议。'
                  + assignment_note + '，所有新工作包须实际运行并复核后方可勾选。\n')
    text = text.replace('\n', '\n' + provenance, 1)
    cards = task_cards(text)
    ids = [c['id'] for c in cards]
    if len(ids) != len(set(ids)) or ids != [f'W{i:02}' for i in range(1, len(ids) + 1)] or not (28 <= len(ids) <= 32):
        raise SystemExit('Unexpected task IDs/count: ' + repr(ids))
    # Put immediately actionable material before the long technical cards.
    sections = re.split(r'(?=^##\s+)', text, flags=re.M)
    intro, sections = sections[0], sections[1:]
    front = [s for s in sections if re.match(r'##\s+[56]\.\s+', s)]
    rest = [s for s in sections if s not in front]
    checklist = ['## 工作包勾选总表', '',
                 '成员甲＝你、成员乙＝同事；姓名可补。完成一项后，写清跑了什么代码、代码放在哪里、Result放在哪里，再勾选。无需跑代码的任务，说明做了什么并填写文档位置即可。', '',
                 '- 成员甲姓名：________；成员乙姓名：________。', '']
    for c in cards:
        owner = c['fields'].get('主责', '待填').rstrip('。')
        record = progress['tasks'].get(c['id'], {})
        checked = 'x' if record.get('checked') else ' '
        status = '；' + record['short_status'] if record else ''
        checklist.append(f"- [{checked}] **{c['id']} {c['title']}**（主责：{owner}；姓名：____{status}）\n"
                         f"  - 跑了什么代码：{record.get('what_ran', '____')}\n"
                         f"  - 代码位置：{record.get('code_location', '____')}\n"
                         f"  - Result位置：{record.get('result_location', '____')}")
    overview = (RUN / 'team_assignment_overview.md').read_text() + '\n\n' if assignment else ''
    latest = (RUN / 'execution_update.md').read_text() + '\n\n' if progress['tasks'] else ''
    text = intro + latest + overview + ''.join(front) + '\n'.join(checklist) + '\n\n' + ''.join(rest)
    section_number = 0
    def renumber(match):
        nonlocal section_number
        content = re.sub(r'^\d+\.\s*', '', match.group(1))
        if content.startswith('附录'):
            return '## ' + content
        section_number += 1
        return f'## {section_number}. {content}'
    text = re.sub(r'^##\s+(.+)$', renumber, text, flags=re.M)
    text = re.sub(r'^(主责|复核|前置|状态|输入|动作|交付|验收／失败出口|结局／学习)：',
                  r'\n**\1：** ', text, flags=re.M)
    cards = task_cards(text)
    FINAL.write_text(text)
    execution_status = ('diet_definition_draft_complete_empirical_validation_pending' if progress['tasks'].get('W04', {}).get('checked')
                        else 'key_date_contract_complete_with_downstream_gates' if progress['tasks'].get('W03', {}).get('checked')
                        else 'metadata_inventory_complete_scientific_analysis_pending' if progress['tasks'] else 'planning_only')
    index = {'date': '2026-10-26', 'status': execution_status, 'author': 'Astra gpt-6-astra xhigh',
             'reviewer': 'Codex', 'document': str(FINAL.relative_to(ROOT)),
             'workstream_assignment': assignment,
             'workpackage_progress': progress,
             'dependency_note': 'Text dependencies include module-specific conditions; this index is not an executable DAG or a run-status database.',
             'tasks': cards}
    (RUN / 'task_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))
    checks = {'task_count': len(cards), 'unique_contiguous_ids': True, 'checked_new_workboxes': len(re.findall(r'^\s*-\s*\[[xX]\]\s*\*\*W\d{2}', text, re.M)),
              'fields_per_task': {c['id']: list(c['fields']) for c in cards},
              'draft_sha256': hashlib.sha256(DRAFT.read_bytes()).hexdigest(),
              'final_sha256': hashlib.sha256(FINAL.read_bytes()).hexdigest()}
    (RUN / 'build_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps({k: checks[k] for k in ('task_count', 'unique_contiguous_ids', 'checked_new_workboxes')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
