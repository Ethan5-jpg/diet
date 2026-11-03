"""Read authorized workspace files to hydrate iCloud placeholders; no external writes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import re,subprocess
W=Path(__file__).resolve().parents[2]
paths=[p for p in (W/'reports').glob('HPP_W*.md') if re.search(r'HPP_W(07_W08|09|10|11_W12|13|14|15)_',p.name)]
paths.append(W/'reports/HPP_Team_Execution_Checklist_2026-10-26.md')
for d in (W/'audits').iterdir():
 if d.is_dir() and any(d.name.startswith(x) for x in ['w07_w08','w09_','w10_','w11_w12','w13_','w14_','w15_']):
  paths += [p for p in d.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md'] and (p.name[:1].isdigit() or p.name.startswith('source_manifest') or p.name in ['MEMO.md','observed_aggregate.json','local_delivery_validation.json','delivery_validation.json','local_source_manifest.json','method_sources.json'])]
paths += list((W/'config').glob('w13*/*.json'))+list((W/'config').glob('w14*/*.json'))+list((W/'config').glob('w15*/*.json'))
paths += [W/'config'/n for n in ['w03_followup_20261027/liver_six_mean_spec.json','w03_followup_20261027/liver_six_phenotype_scope.md','w03_followup_20261027/visit_and_sampling_policy.json','w04_followup_20261027/historical_contact_ledger_draft.json','w04_diet_features_20261026/prior_work_overlap.md']]
paths.append(W/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json')
for p in paths:subprocess.run(['/usr/bin/brctl','download',str(p)],check=True,stdout=subprocess.DEVNULL)
print('DOWNLOADS_REQUESTED',len(paths),flush=True)
def read(p):return str(p.relative_to(W)),len(p.read_bytes())
with ThreadPoolExecutor(max_workers=12) as pool:
 for f in as_completed([pool.submit(read,p) for p in paths]):print('HYDRATED',*f.result(),flush=True)
