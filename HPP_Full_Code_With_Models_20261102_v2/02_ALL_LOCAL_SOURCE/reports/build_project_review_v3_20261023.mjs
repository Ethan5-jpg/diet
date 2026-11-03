/** Build the offline V3 reader from design documents, without reading study data. */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {marked} from '/Users/haochenxue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';

const base=path.dirname(fileURLToPath(import.meta.url));
const documents=[
  ['plan','整合方案','HPP_Integrated_Multisystem_Plan_v3_2026-10-23.md'],
  ['journal','期刊路线','HPP_Journal_Route_v3_2026-10-23.md'],
  ['conference','会议路线','HPP_Conference_Route_v3_2026-10-23.md'],
  ['roadmap','完整工作路径','HPP_Project_Roadmap_v3_2026-10-23.md'],
  ['todo','TODO与人数','HPP_Project_TODO_v3_2026-10-23.md'],
  ['adoption','建议落实表','HPP_Astra_Recommendations_Implementation_v3_2026-10-23.md'],
];
const registryName='HPP_Cohort_Analysis_Register_v3_2026-10-23.json';
const output='HPP_最新版方案_双路线与工作路径_v3_2026-10-23.html';
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const hash=b=>createHash('sha256').update(b).digest('hex');
const manifest=[];const headings=[];let headingCount=0;
const fileAnchors=new Map(documents.map(([id,,file])=>[file,id]));fileAnchors.set(registryName,'register');
const articles=documents.map(([id,label,file])=>{
  const bytes=fs.readFileSync(path.join(base,file));const source=bytes.toString('utf8');
  manifest.push({id,filename:file,bytes:bytes.length,sha256:hash(bytes)});
  const math=[];
  const prepared=source.replace(/\\\[([\s\S]*?)\\\]/g,(_,expr)=>{
    const token=`HPPV3FORMULA${math.length}END`;
    math.push(`<pre class="equation">${esc(expr.trim())}</pre>`);return `\n\n${token}\n\n`;
  });
  let body=marked.parse(prepared,{gfm:true});
  body=body.replace(/<p>HPPV3FORMULA(\d+)END<\/p>/g,(_,i)=>math[Number(i)]);
  body=body.replace(/HPPV3FORMULA(\d+)END/g,(_,i)=>math[Number(i)]);
  body=body.replace(/<h([123])>([\s\S]*?)<\/h\1>/g,(_,level,title)=>{
    const anchor=`${id}-heading-${++headingCount}`;
    if(Number(level)<=2)headings.push({anchor,title:title.replace(/<[^>]+>/g,''),level:Number(level)});
    return `<h${level} id="${anchor}">${title}</h${level}>`;
  });
  body=body.replace(/href="([^"]+)"/g,(whole,link)=>{
    const key=decodeURI(link.split('#')[0]);
    return fileAnchors.has(key)?`href="#${fileAnchors.get(key)}"`:whole;
  });
  body=body.replaceAll('<table>','<div class="table-wrap"><table>').replaceAll('</table>','</table></div>');
  return `<article id="${id}" data-document="${esc(file)}"><div class="meta">${esc(label)} · <a href="${encodeURI(file)}">Markdown原文件</a></div>${body}</article>`;
}).join('\n');
const raw=fs.readFileSync(path.join(base,registryName));const registry=JSON.parse(raw);
manifest.push({id:'register',filename:registryName,bytes:raw.length,sha256:hash(raw)});
const table=(headers,rows)=>`<div class="table-wrap"><table><thead><tr>${headers.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${esc(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const registerArticle=`<article id="register" data-document="${registryName}"><div class="meta">机器登记 · <a href="${registryName}">JSON原文件</a></div><h1>分析队列与假设登记</h1><p>所有新队列N、实际评分数与秩均待验收。下表是设计配置，未运行数据筛选。</p>${table(['模块','基础来源／交集','饮食视图','m','评分来源','最终N'],registry.modules.map(m=>[m.analysis_id,m.candidate_parent_intersection?.join(' ∩ ')||m.base_eligibility_id,m.diet_view,m.output_count,m.score_scope,m.final_n??'待算']))}<h2>当前正式检验清单（待W18/W22冻结）</h2>${table(['假设','模块','零假设','限制数规则'],registry.multiplicity.hypotheses.map(h=>[h.hypothesis_id,h.analysis_id,h.null_hypothesis,h.restriction_count_rule]))}<p>六肝表型及实际暴露家族尚待重订，旧15块不可直接执行；点态95%分量区间描述，不作额外正式方向／阈值／等效确认。所有统计关口尚未验收。</p><details><summary>查看完整配置</summary><pre>${esc(JSON.stringify(registry,null,2))}</pre></details></article>`;
const nav=[...documents.map(([id,label])=>[id,label]),['register','分析登记']].map(([id,label])=>`<a href="#${id}">${esc(label)}</a>`).join('');
const toc=headings.map(h=>`<a class="level-${h.level}" href="#${h.anchor}">${esc(h.title)}</a>`).join('');
const html=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP最新版方案V3 · 双路线与工作路径 · 2026-10-23</title><style>
:root{--bg:#f4f5f0;--paper:#fff;--ink:#203733;--muted:#65736d;--line:#dbe3dc;--green:#196354}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}a{color:var(--green);overflow-wrap:anywhere}header{padding:30px max(25px,calc((100vw - 1450px)/2));background:#184b40;color:#fff}header h1{margin:0;font-size:28px}header p{margin:7px 0 0;color:#dceae1;font-size:14px}.nav{position:sticky;top:0;z-index:3;display:flex;flex-wrap:wrap;gap:7px;padding:10px 22px;background:#f4f5f0f5;border-bottom:1px solid var(--line)}.nav a,.nav button{border:1px solid var(--line);padding:6px 11px;border-radius:5px;background:#fff;color:var(--green);text-decoration:none;font:inherit;font-size:13px;line-height:1.5;cursor:pointer}.summary{max-width:1398px;margin:25px auto 0;padding:18px 24px;background:#e7f0e9;border:1px solid var(--line);border-radius:8px}.summary p{margin:5px 0}.layout{max-width:1450px;margin:auto;padding:25px;display:grid;grid-template-columns:245px minmax(0,1fr);gap:25px}aside{position:sticky;top:90px;align-self:start;max-height:calc(100vh - 110px);overflow:auto}aside a{display:block;padding:5px 8px;font-size:12px;line-height:1.6;text-decoration:none;color:var(--muted)}aside .level-1{font-weight:650;color:var(--green);margin-top:12px}main{min-width:0}article{background:var(--paper);padding:28px 35px;border:1px solid var(--line);border-radius:9px;margin-bottom:28px;scroll-margin-top:85px}.meta{font-size:12px;color:var(--muted);padding-bottom:12px;border-bottom:1px solid var(--line)}h1{font-size:27px;line-height:1.5}h2{font-size:21px;line-height:1.6;color:var(--green);margin-top:32px}h3{font-size:18px}p,li{overflow-wrap:anywhere}li{margin:6px 0}.table-wrap{overflow:auto;margin:18px 0}table{width:100%;min-width:600px;border-collapse:collapse;font-size:13px;line-height:1.7}th,td{text-align:left;vertical-align:top;border:1px solid var(--line);padding:10px 12px}th{background:#eef4ed}tr:nth-child(even) td{background:#fafcf9}code{font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#edf2ed;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{background:#f0f4ee;border:1px solid var(--line);padding:15px;overflow:auto;border-radius:5px;font:12px/1.8 ui-monospace,SFMono-Regular,Menlo,monospace}pre code{padding:0;background:none}.equation{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{padding:9px 17px;margin:18px 0;background:#edf4ed;border-left:3px solid var(--green)}input[type=checkbox]{width:15px;height:15px;vertical-align:middle}.task-list-item{list-style:none}details{margin:20px 0}summary{cursor:pointer}footer{text-align:center;padding:20px;color:var(--muted);font-size:12px}@media(max-width:900px){.layout{display:block;padding:12px}aside{display:none}article{padding:20px}.summary{margin:15px 12px 0}header h1{font-size:23px}}@media print{body{background:#fff;font-size:11px}header{background:#fff;color:#000}header p{color:#444}.nav,aside{display:none}.layout{display:block;padding:0}article{border:0;padding:0;margin-bottom:35px}table{min-width:0;font-size:10px}tr,pre{break-inside:avoid}h1,h2,h3{break-after:avoid}.table-wrap{overflow:visible}a{color:#000}}
</style></head><body><header><h1>HPP研究方案V3</h1><p>2026-10-29用户修订 · 六肝表型正式确定 · CGM两层 · 跨领域比较暂缓 · 统计家族待冻结</p></header><nav class="nav">${nav}<button type="button" onclick="window.print()">打印／保存PDF</button></nav><section class="summary"><p><strong>期刊：</strong>四域各自最大合格队列；六肝表型分别分析；跨领域直接比较暂缓。人级CGM访视汇总不要求饮食共同日，日级序列仍按实际日期对齐。</p><p><strong>菌群：</strong>same visit＋身份／QC可信；无固定天数、前采、同日或日期唯一硬筛。日期unknown保留，访视归属须核实。</p><p><strong>会议：</strong>方法M先科学零假设／先例／最小原型；主动Agent A另列信息与评价协议。最终会议稿类型未指定。</p><p><strong>开工：</strong>环境与来源 → 字段／样本QC → 各模块基础人群与N → 角色／合成精度 → 发现评分 → 一次确认。W13已完成矩阵QC、主样本锁定与独立读回，服务器已同步用户修订；下一包W14。基线O8291/G8690/双部位7228，非最终分析N。</p></section><div class="layout"><aside aria-label="全文目录">${toc}<a href="#register">分析队列与待冻结假设</a></aside><main>${articles}${registerArticle}</main></div><footer>六份当前文档全文＋完整分析配置；当前入口为HPP_CURRENT_PLAN.md。无新个体数据、模型或模拟结果。</footer></body></html>`;
fs.writeFileSync(path.join(base,output),html);
fs.writeFileSync(path.join(base,'HPP_Project_Review_Manifest_v3_2026-10-23.json'),JSON.stringify({version:registry.version,output,html_bytes:Buffer.byteLength(html),html_sha256:hash(html),documents:manifest,article_count:7,heading_count:headingCount,no_external_runtime_or_stylesheet_required:true},null,2)+'\n');
console.log(JSON.stringify({output:path.join(base,output),articles:7,source_documents:manifest.length,bytes:Buffer.byteLength(html)}));
