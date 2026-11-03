/** Build the complete local project review reader; no research data are read. */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {marked} from '/Users/haochenxue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';

const base = path.dirname(fileURLToPath(import.meta.url));
const documents = [
  ['integrated', '当前整合方案 V2', 'HPP_Integrated_Multisystem_Plan_2026-10-21.md'],
  ['cohorts', '各领域准入 V2', 'HPP_Cohort_Specific_Amendment_v2_2026-10-21.md'],
  ['roadmap', '当前执行路径 V2', 'HPP_Project_Roadmap_CohortSpecific_v2_2026-10-21.md'],
  ['todo', '任务清单与启动顺序', 'HPP_Project_TODO_2026-10-21.md'],
  ['changes', '整合与审查记录', 'HPP_Plan_Integration_Change_Log_2026-10-21.md'],
  ['sources', '一手来源核对', 'HPP_Methods_Source_Check_2026-10-21.md'],
  ['archive-assessment', '历史：Astra 独立评估', 'Astra_Integrated_Plan_Assessment_2026-10-21.md'],
  ['archive-roadmap', '历史：Astra 原38项路线', 'Astra_Detailed_Project_Roadmap_2026-10-21.md'],
];
const out = 'HPP_项目方案与详细工作路径_Astra_2026-10-21.html';
const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const manifest = [];
const sections = [];
let headingCount = 0;
const articles = documents.map(([id, label, filename]) => {
  const text = fs.readFileSync(path.join(base, filename), 'utf8');
  manifest.push({id, filename, bytes: Buffer.byteLength(text), sha256: createHash('sha256').update(text).digest('hex')});
  const math = [];
  let prepared = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, expression) => {
    const token = `HPPMATHTOKEN${math.length}END`;
    math.push(`<div class="math-display">\\[${esc(expression.trim())}\\]</div>`);
    return `\n\n${token}\n\n`;
  });
  prepared = prepared.replace(/\\\[([\s\S]*?)\\\]/g, (_, expression) => {
    const token = `HPPMATHTOKEN${math.length}END`;
    math.push(`<div class="math-display">\\[${esc(expression.trim())}\\]</div>`);
    return `\n\n${token}\n\n`;
  });
  prepared = prepared.replace(/(?<![\\$])\$([^$\n]+)\$(?!\$)/g, (_, expression) => {
    const token = `HPPMATHTOKEN${math.length}END`;
    math.push(`\\(${esc(expression)}\\)`);
    return token;
  });
  let body = marked.parse(prepared, {gfm: true});
  body = body.replace(/<p>HPPMATHTOKEN(\d+)END<\/p>/g, (_, i) => math[Number(i)]);
  body = body.replace(/HPPMATHTOKEN(\d+)END/g, (_, i) => math[Number(i)]);
  body = body.replace(/<h([123])>([\s\S]*?)<\/h\1>/g, (_, level, title) => {
    const anchor = `${id}-heading-${++headingCount}`;
    if (Number(level) <= 2) sections.push({id: anchor, label: title.replace(/<[^>]+>/g, ''), level: Number(level)});
    return `<h${level} id="${anchor}">${title}</h${level}>`;
  });
  body = body.replace(/<table>/g,'<div class="table-wrap"><table>').replaceAll('</table>', '</table></div>');
  const history = id.startsWith('archive-') ? '<blockquote><strong>历史原文：</strong>其Common4全局准入、共同日统一饮食、根检验门控，以及菌群固定时间／日期唯一筛选，均以当前V2为准。菌群主标准为同人同访视＋QC，时间敏感性可选。当前执行请先读“各领域准入V2”和“当前执行路径V2”。V2由Codex按用户要求修订，并非新的Astra评估。</blockquote>' : '';
  return `<article id="${id}" data-document="${esc(filename)}"><div class="document-meta">${esc(label)} · <a href="${encodeURI(filename)}">Markdown 原文件</a></div>${history}${body}</article>`;
}).join('\n');

const nav = documents.map(([id, label]) => `<a href="#${id}">${esc(label)}</a>`).join('');
const toc = sections.map(s => `<a class="level-${s.level}" href="#${s.id}">${esc(s.label)}</a>`).join('');
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HPP 项目方案与详细工作路径 · Astra · 2026-10-21</title>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root{--bg:#f5f5f0;--paper:#fff;--ink:#203334;--muted:#627170;--line:#dce5df;--accent:#17665b}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}a{color:var(--accent);overflow-wrap:anywhere}header{padding:22px max(24px,calc((100vw - 1480px)/2));background:#173f39;color:#fff}header p{margin:8px 0 0;color:#d9e8e0;font-size:14px}header strong{font-size:24px}.top-nav{position:sticky;top:0;z-index:3;display:flex;gap:8px;flex-wrap:wrap;padding:10px 24px;background:#f5f5f0f5;border-bottom:1px solid var(--line)}.top-nav a,.top-nav button{padding:6px 10px;border:1px solid var(--line);border-radius:5px;text-decoration:none;font-size:13px;background:#fff;color:var(--accent)}button{cursor:pointer}.layout{max-width:1480px;margin:auto;padding:26px;display:grid;grid-template-columns:255px minmax(0,1fr);gap:28px}aside{align-self:start;position:sticky;top:85px;max-height:calc(100vh - 105px);overflow:auto;padding-right:8px}aside a{display:block;font-size:12px;line-height:1.65;padding:5px 7px;text-decoration:none;color:var(--muted)}aside .level-1{color:var(--accent);font-weight:650;margin-top:9px}main{min-width:0}article{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:30px 36px;margin-bottom:30px;scroll-margin-top:85px}.document-meta{font-size:12px;color:var(--muted);padding-bottom:12px;border-bottom:1px solid var(--line)}h1{font-size:28px;line-height:1.5}h2{font-size:22px;line-height:1.6;margin-top:36px;color:var(--accent)}h3{font-size:18px;margin-top:25px}p,li{overflow-wrap:anywhere}li{margin:7px 0}ul,ol{padding-left:25px}.table-wrap{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;min-width:580px;font-size:13px;line-height:1.7}th,td{text-align:left;vertical-align:top;border:1px solid var(--line);padding:10px 12px}th{background:#eef5ef}tr:nth-child(even) td{background:#fbfcfa}code{font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef2ee;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{overflow-x:auto;background:#f1f4f0;border:1px solid var(--line);padding:15px;border-radius:5px;font-size:12px;line-height:1.75}pre code{background:none;padding:0}blockquote{margin:18px 0;padding:8px 18px;border-left:3px solid var(--accent);background:#eef5ef}input[type=checkbox]{width:16px;height:16px;vertical-align:middle;accent-color:var(--accent)}footer{padding:20px;text-align:center;color:var(--muted);font-size:12px}.math-display{overflow-x:auto;padding:12px;background:#f6f8f4;border-radius:5px}.task-list-item{list-style:none}hr{border:0;border-top:1px solid var(--line);margin:30px 0}@media(max-width:900px){.layout{display:block;padding:14px}aside{display:none}article{padding:22px}header strong{font-size:20px}}@media print{header{background:#fff;color:#000}header p{color:#555}.top-nav,aside,button{display:none}.layout{display:block;padding:0}article{border:0;border-radius:0;padding:0;margin:0 0 35px}h1,h2,h3{break-after:avoid}tr,pre{break-inside:avoid}.table-wrap{overflow:visible}table{min-width:0;font-size:10px}a{color:#000}body{font-size:11px;background:#fff}}
</style></head><body>
<header><strong>HPP 项目方案与详细工作路径 · V2</strong><p>2026-10-21 · 按用户要求改为各结局独立准入／分析，跨域比较才取对应交集 · 保留Astra历史评估，未启动新数据分析</p></header>
<nav class="top-nav">${nav}<button onclick="window.print()">打印 / 保存 PDF</button></nav>
<section aria-label="工作路径概览" style="max-width:1428px;margin:24px auto 0;padding:18px 22px;background:#eaf2eb;border:1px solid #dce5df;border-radius:8px;font-size:14px;line-height:2"><strong>V2人群规则</strong>　各域自己的合格人群 → 各自发现／确认；两域比较 → 对应交集＋共享评分；四域联合 → Common4扩展，不门控其他域。<br><strong>菌群准入</strong>　同人同访视＋身份／QC明确；不以固定天数、同日、前采或日期唯一筛人。已知时间作描述，敏感性可选。<br><span style="color:#627170">Body／liver／BMI不要求CGM或无关结局；饮食—CGM继续真实日期对齐。全项目按人／家庭统一划分。各模块人数待重算，3070不是统一N。</span></section>
<div class="layout"><aside aria-label="目录">${toc}</aside><main>${articles}</main></div>
<footer>八份文档全文收录，当前V2与历史Astra原文明确分开；原始及个体数据留在 TRE。勾选仅保存到本浏览器，正式状态以运行记录为准。</footer>
<script>
document.querySelectorAll('input[type=checkbox]').forEach((el,i)=>{el.disabled=false;const key='hpp-plan-20261021-v2-task-'+i;try{const old=localStorage.getItem(key);if(old!==null)el.checked=old==='1';}catch{}el.addEventListener('change',()=>{try{localStorage.setItem(key,el.checked?'1':'0');}catch{}});});
</script>
</body></html>`;
fs.writeFileSync(path.join(base, out), html);
fs.writeFileSync(path.join(base, 'HPP_Project_Review_Manifest_2026-10-21.json'), JSON.stringify({generated_at: new Date().toISOString(), output: out, heading_count: headingCount, html_bytes: Buffer.byteLength(html), html_sha256: createHash('sha256').update(html).digest('hex'), documents: manifest}, null, 2)+'\n');
console.log(JSON.stringify({output: path.join(base,out), documents: manifest.length, headings: headingCount, bytes: Buffer.byteLength(html)}));
