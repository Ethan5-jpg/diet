import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {marked} from '/Users/haochenxue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';

const base=path.dirname(fileURLToPath(import.meta.url));
const source='Claude_独立方法评估_中文完整修订版_2026-10-15.md';
const text=fs.readFileSync(path.join(base,source),'utf8');
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const math=[];
let prepared=text.replace(/\$\$([\s\S]*?)\$\$/g,(_,x)=>{
  const k=math.length;math.push(`<div class="equation">\\[${esc(x.trim())}\\]</div>`);return `\n\nMATHPLACEHOLDER${k}END\n\n`;
});
prepared=prepared.replace(/\$([^$\n]+)\$/g,(_,x)=>{
  const k=math.length;math.push(`\\(${esc(x)}\\)`);return `MATHPLACEHOLDER${k}END`;
});
let body=marked.parse(prepared,{gfm:true});
body=body.replace(/<p>MATHPLACEHOLDER(\d+)END<\/p>/g,(_,k)=>math[Number(k)]);
body=body.replace(/MATHPLACEHOLDER(\d+)END/g,(_,k)=>math[Number(k)]);
body=body.replace(/<table>/g,'<div class="table-wrap"><table>').replace(/<\/table>/g,'</table></div>');
let i=0;const nav=[];
body=body.replace(/<h([23])>(.*?)<\/h\1>/g,(_,n,title)=>{
  const id=`section-${++i}`;nav.push(`<a class="level-${n}" href="#${id}">${title}</a>`);return `<h${n} id="${id}">${title}</h${n}>`;
});
const html=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Claude独立方法评估｜中文完整修订版</title>
<style>
:root{--ink:#203334;--muted:#5b6b6b;--line:#dce5df;--green:#16645b}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}body{margin:0;background:#f5f4ef;color:var(--ink);font:16px/1.9 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}header{padding:14px 24px;border-bottom:1px solid var(--line);background:#f5f4ef;display:flex;gap:15px;align-items:center;flex-wrap:wrap}header a,button{font-size:13px}a{color:var(--green);overflow-wrap:anywhere}button{margin-left:auto;padding:7px 12px;border:1px solid var(--line);background:#fff;border-radius:6px;color:var(--ink);cursor:pointer}.layout{max-width:1450px;margin:auto;padding:30px;display:grid;grid-template-columns:245px minmax(0,1fr);gap:30px}aside{position:sticky;top:20px;align-self:start;max-height:94vh;overflow:auto}aside a{display:block;text-decoration:none;color:var(--muted);padding:5px 7px;font-size:12px;line-height:1.7}aside .level-3{padding-left:20px}main{min-width:0;padding:32px 40px;background:#fff;border:1px solid var(--line);border-radius:12px}h1{font-size:29px;line-height:1.5;margin-top:0}h2{font-size:23px;color:var(--green);margin-top:44px;padding-top:22px;border-top:1px solid var(--line)}h3{font-size:19px;margin-top:32px}p,li{overflow-wrap:anywhere}li{margin:9px 0}blockquote{margin:20px 0;padding:12px 20px;background:#eef5f0;border-left:3px solid var(--green);font-size:14px}code{font:13px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f0f4f0;padding:2px 4px;overflow-wrap:anywhere}.table-wrap{max-width:100%;overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.8}th,td{text-align:left;vertical-align:top;padding:10px 13px;border-bottom:1px solid var(--line);min-width:95px}th{background:#edf4ef}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcf9}.equation{overflow-x:auto;margin:20px 0;padding:12px;background:#f7faf7}hr{border:0;border-top:1px solid var(--line);margin:30px 0}footer{margin-top:45px;font-size:12px;color:var(--muted);word-break:break-all}.notice{font-size:12px;color:var(--muted)}@media(max-width:850px){.layout{display:block;padding:12px}aside{position:static;max-height:none;margin-bottom:18px}aside nav{columns:2}main{padding:22px 18px}h1{font-size:24px}}@media(max-width:480px){aside nav{columns:1}}@media print{aside,header{display:none}.layout{display:block;padding:0}body{background:#fff;font-size:10pt}main{border:0;padding:0}h2,h3{break-after:avoid}.table-wrap,.equation{overflow:visible}table{font-size:8pt}}
</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'local'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
</head><body><header><b>Claude 独立方法评估 · 中文完整修订版</b><a href="${esc(source)}">中文Markdown</a><a href="Claude_Independent_Methods_Assessment_EN_2026-10-14.md">英文源稿</a><button onclick="window.print()">打印／保存PDF</button></header><div class="layout"><aside><details open><summary>全文目录</summary><nav>${nav.join('')}</nav></details></aside><main>${body}<footer><p>英文源稿SHA256：${createHash('sha256').update(fs.readFileSync(path.join(base,'Claude_Independent_Methods_Assessment_EN_2026-10-14.md'))).digest('hex')}</p><p class="notice">公式使用MathJax在线排版；离线时数学表达式仍保留，完整公式源码见中文Markdown。原文观点与待核实事项按原评估保留。</p></footer></main></div></body></html>`;
const out=path.join(base,'Claude_独立方法评估_中文完整修订版_2026-10-15.html');
fs.writeFileSync(out,html,'utf8');
console.log(JSON.stringify({file:out,headings:i,mathExpressions:math.length,bytes:Buffer.byteLength(html)}));
