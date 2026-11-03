"""Render a bounded prior-art comparison for the symmetric joint question."""
from pathlib import Path
from html import escape
import hashlib
import json
import re

BASE=Path(__file__).resolve().parent
RECORD=BASE/"joint_oral_gut_novelty_search_2026-10-14.json"
ASTRA=BASE/"Astra_joint_oral_gut_novelty_assessment_2026-10-14.txt"
FULL="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html"
OUT=BASE/"HPP_联合口腔-肠道_母题新颖性核查_2026-10-14.html"
record=json.loads(RECORD.read_text(encoding="utf-8"))
astra=ASTRA.read_text(encoding="utf-8")
css=re.search(r"<style>(.*?)</style>",(BASE/FULL).read_text(encoding="utf-8"),re.S).group(1)
css+="""
.sidebar summary{display:none}.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;line-height:1.95}
.idea{border:1px solid var(--line);border-radius:10px;padding:20px;margin:20px 0;background:#f6f9f2}.idea h3{margin-top:0}
@media(max-width:760px){.sidebar summary{display:list-item;cursor:pointer;font-weight:650;color:var(--green)}.sidebar-title{display:none}.sidebar nav{padding-top:12px}.astra-original{font-size:13px}}
@media print{.sidebar{display:none!important}.astra-original{font-size:10pt}}
"""
sections=[]
def section(key,title,body,label="研究判断＋原始先例"):
 i=len(sections)+1
 sections.append((key,title,f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{i:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'

section("answer","结论：有接近先例，联合母题仍值得推进", """
<p class="lead">目前核查到的原始文献已经包含双部位饮食／代谢研究、肠道精准营养、大规模HPP饮食—菌群及口腔—代谢研究。<strong>尚未核实一篇把数千人配对的口腔与肠道对称联合特征，作为识别饮食相关动态反应模式的主分析。</strong></p>
<div class="idea"><h3>用户确定的母题</h3><p><strong>口腔与肠道菌群的联合特征，是否标记一个人的饮食相关代谢反应模式？</strong></p><p>两个部位同时作为变量，不预设主从。固定某一部位检查另一部位的信息属于解释性子比较。</p></div>
<p>这意味着有一个值得检验、可与前作区分的研究方向；并不意味着已经证明首次。新意必须由反应定义、联合信息结构及其在留出参与者中的保留性支撑。</p>
<div class="note accent">“未核实完全同型论文”描述本轮检索的边界。它不能被写成“没人做过”。本轮也未证明数据组合是最大、最全或最好。</div>
""")

rows=[]
for study in record["studies"]:
 rows.append((f'{study["authors"]} · {study["journal"]} {study["year"]}',study["sample"],study["design"],study["overlap"],study["difference"]))
section("matrix","八项最需要对照的研究／协议",table(["研究", "人数口径", "设计与资源", "与母题重叠", "已核查的主要区别"],rows)+"""
<p>人数按各研究的实际口径列出；模态总数不是完整配对数。MIVAS III是协议，不计作已完成的大样本结果。不同研究的规模不能替代标准餐、随机干预、重复菌群或独立验证的设计信息。</p>
""")

section("closest","最接近的双部位先例，具体重叠到哪里", """
<p><a href="https://www.nature.com/articles/s41467-023-41042-x">Shoer等，NC2023</a>是关键参照。主文Model analysis使用肠道组成预测血清代谢物变化，PPT个性化餐后预测也描述使用肠道特征；双部位主要结果围绕干预变化、中介及菌株动态。其附件未穷尽核查，不能断言没有任何联合模型。</p>
<p><a href="https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.00454/full">Tang等，Frontiers2019</a>已有饮食、口腔、肠道和代谢组的联合资源及全局关系分析；主文口腔全局关系未获支持，后续转向肠道。它不是个人CGM饮食反应模式的主要研究。采样子集交集需要按原始图表，不能把136当全链N。</p>
<p>因此，不能声称首次把饮食、口腔、肠道与代谢放在一起。当前要辨认的区别，是<strong>双部位作为对称特征，能否表征人的饮食相关反应，并区分共同、特异与联合信息。</strong></p>
""")

section("hpp","同队列前作：规模与资源必须对照", """
<p><a href="https://www.nature.com/articles/s41591-026-04312-x">HPP Nature Medicine 2026</a>已用10068名参与者的日志与宏基因组分析饮食—肠道关系及探索性饮食模拟。其已核查的摘要与扩展图不能等同完整正文／附件审查，但足以说明HPP规模和个性化营养资源已有正式前作。</p>
<p><a href="https://www.nature.com/articles/s41467-026-72748-3">HPP口腔NC2026</a>也已有大规模口腔—代谢表型图谱。当前研究应在这些基础上提出联合反应问题；同队列、重叠参与者与既有探索需披露。</p>
<p>当前项目已整理的肠道—饮食—CGM主基础为4139人，CGM汇总7493人；口腔资源由用户确认。新的四模态配对N尚未核实。该交集及模态时点决定本研究可分析的联合模式。</p>
""")

section("contribution","把区别写成可检验的科学贡献",table(["贡献层", "实际需要的结果", "不能替代它的结果"],[
 ("饮食相关反应模式", "预先限定的食物／营养／日级饮食对比对应不同血糖关联结构，区分平均水平与餐前状态。", "仅重新预测谁总体血糖高，或按整体CGM分组后命名饮食反应型。"),
 ("双部位对称信息", "同配对样本、同饮食支持域中比较无菌群、口腔、肠道、联合模型；控制容量及资料可用性。", "两张MWAS热图，或把某部位样本更多造成的成绩差叫生理分工。"),
 ("联合对应的模式保留", "所有特征与模型选择在训练参与者内；整人留出并利用既有重复记录评价反应差异。", "训练集中的联合差异，或仅依靠已有参与者／模板的平均预测误差降低。"),
 ("共同与互补的解释", "量化两个部位的信息重叠与互补；若提出非加性，另作直接有限检验。", "强求三重交互显著，把加性互补视为失败，或把同名菌叫迁移。"),
])+"""
<p>该框架允许联合信息主要是冗余、加性互补或有条件的非加性，也允许有精度地发现没有有意义增量。结果决定结论，不预设联合一定胜出。</p>
<p>反应模式的估计对象应在正式分析前明确。多餐／多日信息可用于估计饮食相关差异，但单次菌群测量不能证明菌群动态变化、暴露前预测或换食物的因果获益。</p>
""")

section("best","怎样评价“最全、最好”",table(["维度", "当前可以说什么"],[
 ("人群规模", "若四模态交集保持数千人，这是显著资源优势；尚不能据此证明全球最大。"),
 ("模态广度", "已有研究还测了代谢组、细胞因子或菌株动态；当前确认的分析表不能支持最全面。"),
 ("反应刻画", "HPP的真实饮食日志与重复CGM有价值；标准餐的可比性和自由生活的生态真实性分别服务不同问题。"),
 ("时序与因果深度", "随机饮食与重复组学有其独立优势；现有观察资料不能以人数弥补随机化。"),
 ("科学贡献", "能否界定双部位如何共同标记饮食相关异质性，需靠具体结果及其精度。"),
])+"""
<p>较准确的研究定位是<strong>“人群规模、配对口腔—肠道、饮食相关反应表征”</strong>。这个定位已经足以提出有辨识度的问题，无需先获得最大或最好标签。</p>
""")

section("astra","Astra独立新颖性裁决全文",'<div class="astra-original" id="astra-source">'+escape(astra)+'</div>',"gpt-6-astra · 原文")

links=''.join(f'<li><a href="{escape(s["url"],quote=True)}">{escape(s["authors"]+" · "+s["journal"]+" "+str(s["year"]))}</a>：'+escape(s['title'])+'<br><small>DOI '+escape(s['doi'])+'；核查深度：'+escape(s['evidence_depth'])+'</small></li>'for s in record['studies'])
section("sources","原始来源、检索记录与边界",'<ul>'+links+'</ul>'+"""
<p>截至2026-10-14完成23条定向检索词组合，并核查最接近原始论文的摘要、主文或扩展图。学术MCP未挂载，本轮使用可检索PubMed记录和官方期刊页面，不声称完成Scopus／CrossRef穷尽检索；摘要、主文及附录的核查深度在上方逐项列明。</p>
<div class="file-links"><a href="joint_oral_gut_novelty_search_2026-10-14.json">检索词／逐篇核查JSON</a><a href="Astra_joint_oral_gut_novelty_assessment_2026-10-14.txt">Astra裁决TXT</a><a href="HPP_大队列与口腔-肠道_研究定位_Astra_2026-10-14.html">双部位研究定位（已改为对称母题）</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">已有结果报告</a></div>
<p>前六节为主线程综合文献核查与判断，第七节保存Astra原文。本轮未进入服务器拟合模型；四模态交集和反应模式可识别性仍待实际数据核实。</p>
"""+'<p>Astra原文SHA256</p><pre>'+hashlib.sha256(ASTRA.read_bytes()).hexdigest()+'</pre>',"可复查来源记录")

nav=''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>'for i,(key,title,_)in enumerate(sections,1))
page="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 联合口腔—肠道母题新颖性核查 · 2026-10-14</title><style>"""+css+"""</style></head><body id="top"><header class="topbar"><div class="brand">HPP · 联合母题新颖性<small>PRIOR ART / SYMMETRIC ORAL AND GUT FEATURES</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header><div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 先例、区别与资源定位</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>"""+nav+"""</nav></details><div class="sidebar-footer"><b>2026年10月14日</b>23条定向检索组合<br>八项原始研究／协议<br>具体新意尚待结果支撑</div></aside><main><div class="hero"><div class="eyebrow">HPP / JOINT MICROBIOME QUESTION</div><h1>联合菌群与饮食反应<br>已有研究做到哪一步</h1><p>对称比较口腔与肠道，区分已有资源组合与可论证的科学贡献；完整保留Astra裁决及检索边界。</p><div class="hero-meta"><span>最接近先例逐项比较</span><span>最新HPP同队列前作</span><span>规模／广度／因果深度分开</span><span>2026-10-14</span></div></div>
"""+''.join(body for _,_,body in sections)+"""<footer class="footer">联合母题新颖性核查 · 已核查证据与未知范围分别标注 · 2026-10-14</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a><script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){d.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>"""
OUT.write_text(page,encoding="utf-8")
print(OUT)
print(f"sections={len(sections)} bytes={OUT.stat().st_size}")
