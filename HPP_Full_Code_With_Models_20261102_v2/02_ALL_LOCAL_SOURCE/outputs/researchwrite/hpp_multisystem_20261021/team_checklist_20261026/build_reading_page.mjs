// Offline HTML reading companion, generated from the reviewed Markdown only.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { marked } from '/Users/haochenxue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';

const run = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(run, '../../../..');
const source = path.join(root, 'reports/HPP_Team_Execution_Checklist_2026-10-26.md');
const output = path.join(root, 'reports/HPP_Team_Execution_Checklist_2026-10-26.html');
const md = await fs.readFile(source, 'utf8');
const escape = s => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const headings = [...md.matchAll(/^##\s+(.+)$/gm)];
const usedIds = new Map();
const headingIds = [];
let body = marked.parse(md, { gfm: true });
body = body.replace(/<h2>([\s\S]*?)<\/h2>/g, (_, content) => {
  const id = `section-${headingIds.length + 1}`;
  headingIds.push(id);
  return `<h2 id="${id}">${content}</h2>`;
});
body = body.replace(/<h3>(W\d{2})[\s：:]+([\s\S]*?)<\/h3>([\s\S]*?)(?=<h[123](?:\s|>)|$)/g, (_, id, title, content) => {
  usedIds.set(id, true);
  return `<details class="task" id="${id}" ${Number(id.slice(1)) <= 6 ? 'open' : ''}><summary><span class="task-id">${id}</span> ${title}</summary><div class="task-body">${content}</div></details>`;
});
if (usedIds.size < 28 || headings.length !== headingIds.length) throw new Error('Task or section conversion mismatch');
const toc = headings.map((h, i) => `<a href="#${headingIds[i]}">${escape(h[1])}</a>`).join('\n');
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HPP 期刊优先 · 两人团队任务清单</title>
<style>
:root{color-scheme:light;--ink:#192b39;--muted:#526776;--line:#dce4e9;--accent:#185b85;--paper:#fff;--bg:#f3f6f8}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}header{background:#183446;color:white;padding:22px max(24px,calc((100vw - 1360px)/2));}header strong{font-size:21px}header p{margin:5px 0 0;color:#d6e4ec;font-size:14px}
.layout{display:grid;grid-template-columns:245px minmax(0,1fr);gap:28px;max-width:1360px;margin:26px auto;padding:0 24px}aside{position:sticky;top:18px;align-self:start;max-height:94vh;overflow:auto;font-size:14px}aside nav a{display:block;padding:7px 10px;text-decoration:none;border-left:2px solid var(--line)}aside nav a:hover{background:#e7eff4;border-color:var(--accent)}
main{min-width:0;background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:32px 40px}h1{font-size:29px;line-height:1.4;margin:0 0 22px}h2{font-size:23px;margin:38px 0 16px;padding-top:10px;border-top:1px solid var(--line);scroll-margin-top:18px}h3{font-size:19px}p{margin:12px 0}blockquote{margin:18px 0;padding:10px 18px;border-left:4px solid #6c96b0;background:#f1f6f9;color:#354f60}blockquote p{margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;display:block;overflow-x:auto}th{background:#eaf0f4;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top;min-width:90px}code{font:13px/1.6 ui-monospace,SFMono-Regular,monospace;background:#edf2f5;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{background:#edf2f5;padding:16px;overflow:auto}pre code{padding:0}li{margin:6px 0}input[type=checkbox]{margin-right:7px}
.tools{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 20px}button,input[type=search]{font:inherit;font-size:14px;border:1px solid #b9cbd7;border-radius:5px;background:white;padding:7px 12px;color:var(--ink)}button{cursor:pointer}button:hover{background:#edf3f7}input[type=search]{flex:1;min-width:210px}.hint{font-size:13px;color:var(--muted)}.task{border:1px solid var(--line);border-radius:7px;margin:12px 0;scroll-margin-top:18px}.task summary{cursor:pointer;padding:13px 16px;font-weight:650;background:#f6f9fb;line-height:1.6}.task-id{display:inline-block;color:var(--accent);font:700 14px ui-monospace,monospace;margin-right:7px}.task-body{padding:4px 20px 16px}.task[hidden]{display:none}footer{padding:20px 0;color:var(--muted);font-size:13px;border-top:1px solid var(--line);margin-top:32px}
@media(max-width:950px){.layout{display:block;margin:16px auto;padding:0 12px}aside{position:static;max-height:none;margin-bottom:16px}aside nav{display:flex;flex-wrap:wrap;gap:4px}aside nav a{border:1px solid var(--line);border-radius:4px}main{padding:22px 18px}h1{font-size:25px}}
@media print{body{background:white;font-size:11pt}header,aside,.tools,.hint{display:none}.layout{display:block;margin:0;padding:0}main{border:0;padding:0}.task{break-inside:avoid}details.task>summary{list-style:none}details.task>.task-body{display:block!important}h2{break-after:avoid}table{font-size:9pt;display:table}a{color:inherit;text-decoration:none}}
</style></head><body><header><strong>HPP · 期刊优先</strong><p>两人协作执行清单 · ${usedIds.size}个工作包 · 2026-10-29用户修订 · 原Astra起草 / Codex更新</p></header>
<div class="layout"><aside><nav>${toc}</nav><p class="hint"><a href="${path.basename(source)}">编辑用 Markdown 原文</a></p></aside><main>
<div class="tools"><input type="search" id="search" placeholder="搜索工作包：饮食、CGM、主责、冻结……" aria-label="搜索任务"><button id="expand">全部展开</button><button id="collapse">全部收起</button><button id="print">打印</button></div>
<p class="hint">本页为只读任务说明；勾选及负责人请在 Markdown 原文统一维护。展开、搜索操作不会改变任务验收状态。</p>
${body}<footer>当前科学设计仍为V3；本清单为执行配套。会议与Agent延期。真实人数、分组名单、暴露套餐和科学验收须后续记录。</footer></main></div>
<script>
const cards=[...document.querySelectorAll('details.task')];
document.getElementById('expand').addEventListener('click',()=>cards.forEach(x=>x.open=true));
document.getElementById('collapse').addEventListener('click',()=>cards.forEach(x=>x.open=false));
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.trim().toLocaleLowerCase();cards.forEach(x=>{const match=!q||x.textContent.toLocaleLowerCase().includes(q);x.hidden=!match;if(q&&match)x.open=true;});});
window.addEventListener('hashchange',()=>{const x=document.getElementById(location.hash.slice(1));if(x&&x.matches('details'))x.open=true;});
document.getElementById('print').addEventListener('click',()=>{cards.forEach(x=>{x.hidden=false;x.open=true;});window.print();});
</script></body></html>`;
await fs.writeFile(output, html);
await fs.writeFile(path.join(run,'html_build.json'), JSON.stringify({date:'2026-10-29',output:path.relative(root,output),source:path.relative(root,source),source_sha256:crypto.createHash('sha256').update(md).digest('hex'),task_count:usedIds.size,sections:headings.length,external_runtime_dependencies:false,browser_visual_qa_performed:false,reason:'Local file URL policy prevents automated browser preview; no alternate URL workaround used.'},null,2)+'\n');
console.log(JSON.stringify({output:path.relative(root,output),tasks:usedIds.size,sections:headings.length}));
