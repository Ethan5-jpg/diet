from pathlib import Path
import re,json,hashlib,html
D=Path(__file__).resolve().parent;W=D.parent.parent;P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';p=W/'reports/HPP_Team_Execution_Checklist_2026-10-26.md';s=p.read_text();start=s.index('- [x] **W15 ');end=s.index('- [ ] **W16 ',start);block=s[start:end]
items={'跑了什么代码':'01预登记、02构建顺序／账本、03独立BFS61检查、04发布、05正式读回82检查；实际exit全0','代码位置':f'`{P}/scripts/w15_split_access_20261029/`；本地功能见[MEMO](../audits/w15_split_access_2026-10-29/MEMO.md)','Result位置':f'`{P}/data_qc/w15_split_access_20261029/`；正式`freeze/`成员与账本、`config/access_policy.yaml`；[W15报告](HPP_W15_Split_Access_Ledger_2026-10-29.html)'}
for k,v in items.items():block=block.replace(k+'：____',k+'：'+v)
p.write_text(s[:start]+block+s[end:])
h=p.with_suffix('.html');s=h.read_text();s=s.replace('<aside data-w15-current="true"','<div data-w15-current="true"',1);s=s.replace('</p></aside><aside style="padding:20px;background:#e6f2ec','</p></div><aside style="padding:20px;background:#e6f2ec',1)
# Collapse the previous W14 notice so it no longer claims to be the latest status.
old=re.search(r'<aside style="padding:20px;background:#e6f2ec.*?</aside>',s,re.S)
assert old
text=old.group(0).replace('2026-10-29最新：W14技术交付已完成，下一步W15。','历史：W14技术交付时的状态；当前见W15。').replace('冲突处以本提示及当前Markdown/进度JSON为准','冲突处以W15提示及当前Markdown/进度JSON为准')
text=re.sub(r'^<aside[^>]*>','<details style="margin:16px"><summary>历史W14交付提示</summary><div>',text).replace('</aside>','</div></details>');s=s[:old.start()]+text+s[old.end():]
start=s.index('<li><p><input disabled="" type="checkbox"> <strong>W15 ');end=s.index('<li><p><input disabled="" type="checkbox"> <strong>W16 ',start);block=s[start:end].replace('type="checkbox"','type="checkbox" checked=""',1).replace('（主责：乙；姓名：____）','（Codex代执行；代码验收通过，人工复核待进行；最终角色W18）')
for k,v in items.items():block=block.replace(k+'：____',k+'：'+html.escape(v))
s=s[:start]+block+s[end:];start=s.index('<details class="task" id="W15"');end=s.index('<details class="task" id="W16"',start);block=s[start:end];assert '<p><strong>状态：</strong> 待执行。</p>' in block
status=json.loads((W/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json').read_text())['tasks']['W15']['status_text'];block=block.replace('<p><strong>状态：</strong> 待执行。</p>','<p><strong>状态：</strong> '+html.escape(status)+' <a href="HPP_W15_Split_Access_Ledger_2026-10-29.html">完整W15报告</a></p><p>本次可选设计核心未启用；后续需要时先固定全单元前缀及回执再开放。保护为流程／API层，无文件系统权限隔离。</p>');h.write_text(s[:start]+block+s[end:])
m=D/'local_delivery_manifest.json';j=json.loads(m.read_text());j['updated_file_sha256']={p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in j['updated_files']};m.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print('CHECKLIST_W15_MD_HTML_UPDATED')
