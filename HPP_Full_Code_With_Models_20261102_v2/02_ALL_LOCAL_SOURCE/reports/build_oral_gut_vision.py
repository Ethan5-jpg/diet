"""Render Astra's large paired-cohort and oral/gut scientific vision."""
from pathlib import Path
from html import escape
import hashlib
import re

BASE=Path(__file__).resolve().parent
SOURCE=BASE/"Astra_oral_gut_large_cohort_vision_2026-10-14.txt"
FULL="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html"
OUT=BASE/"HPP_大队列与口腔-肠道_研究定位_Astra_2026-10-14.html"
source=SOURCE.read_text(encoding="utf-8")
css=re.search(r"<style>(.*?)</style>",(BASE/FULL).read_text(encoding="utf-8"),re.S).group(1)
css+="""
.sidebar summary{display:none}.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;line-height:1.95}
.idea{border:1px solid var(--line);border-radius:10px;padding:20px;margin:20px 0;background:#f6f9f2}.idea h3{margin-top:0}
@media(max-width:760px){.sidebar summary{display:list-item;cursor:pointer;font-weight:650;color:var(--green)}.sidebar-title{display:none}.sidebar nav{padding-top:12px}.astra-original{font-size:13px}}
@media print{.sidebar{display:none!important}.astra-original{font-size:10pt}}
"""
sections=[]
def section(key,title,body,label="综合研究定位 · 尚待数据检验"):
 i=len(sections)+1
 sections.append((key,title,f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{i:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

section("judgment","大队列与口腔数据，确实改变研究定位", """
<p class="lead"><strong>口腔菌群可以成为主科学问题的一部分。</strong>数千人的同人配对数据，有机会让我们辨认单看肠道会混在一起的人群，研究口腔与肠道的联合特征如何对应饮食相关代谢易感性。</p>
<div class="idea"><h3>按用户最新选择固定的母题</h3><p><strong>口腔与肠道菌群的联合特征，是否标记一个人的饮食相关代谢反应模式？</strong></p><p>两个部位对称作为变量。固定肠道比较口腔、固定口腔比较肠道，均是子比较；总问题不预设主从。它可以回答两个部位的共同、特异与联合信息，每种结论需要自己的直接证据。</p></div>
<p>建议用现有动态证据最清楚的NOVA4—日Mean关系作为主要检验入口，将AMED／地中海饮食问题作为限定的背景或次要问题。具体心血管表型若能可靠匹配，再承担这类易感性的临床关联层。</p>
""")

section("scale","规模优势如何转成可回答的问题",table(["已知资源", "它支持什么", "人数口径"],[
 ("现有CGM汇总7493人", "群体覆盖与已有动态表型基础。", "CGM人数，不等于双菌群配对人数。"),
 ("饮食—肠道—CGM主基础4139人", "相较小队列，更有机会估计有足够支持的群落组合与异质性。", "单条模型仍按字段完整性确定N。"),
 ("餐次菌群分析3631人", "利用已有多餐记录区分背景状态与相对反应。", "独立菌群信息单位是参与者，餐次不能替代人数。"),
 ("用户确认口腔菌群数据可用", "同人比较两个生态系统，检验口腔对肠道未捕捉的信息。", "配对ID、各模态时点与全链交集待核实。"),
])+"""
<p>数千人规模的科研价值，是使较少见但有足够支持的双部位组合可比较，并能做按人留出的内部评估。分组、交互及联合结构的精度取决于配对N、暴露变化和状态分布，不能根据总N直接承诺功效。</p>
""", "冻结项目事实＋用户新确认资源")

section("references","PREDICT与地中海饮食研究，各学什么",table(["参照论文", "已经建立的基础", "本项目可以借鉴的研究逻辑"],[
 ("PREDICT 2020，1002人UK研究＋100人US验证", "人的特征、餐食和情境对应代谢反应；已展示个人食物排序。", "把个体特征放进一个明确反应问题，认真评价泛化。双部位联合状态是我们需要实际检验的区别。"),
 ("PREDICT 2021，1098人深宏基因组研究", "习惯饮食、肠道组成与空腹／餐后代谢表型的深度连接。", "用深表型分辨共同背景与反应成分，避免只以相关条目数组织故事。"),
 ("地中海饮食2021，307名男性", "饮食与心代谢指标的关联随菌群组成而不同，使用长期饮食与重复菌群资料。", "学习以效应修饰回答‘哪些人关联更强’；HPP规模与双部位是独立优势，不能替代其时序资料。"),
])+"""
<p>原始来源：<a href="https://www.nature.com/articles/s41591-020-0934-0">PREDICT 2020</a>；<a href="https://www.nature.com/articles/s41591-020-01183-8">PREDICT 2021</a>；<a href="https://pubmed.ncbi.nlm.nih.gov/33574608/">地中海饮食2021</a>。PREDICT两篇论文的人数与研究重点不同，不能混用。</p>
<p>我们的联合问题应明确到：在相近饮食和肠道表征下，口腔信息是否对应不同的饮食—日Mean斜率；这种结构能否在留出参与者中保留。仅增加特征数量不会自动回答它。</p>
""")

section("prior-art","已有口腔成果与双部位先例", """
<p>HPP已有<a href="https://www.nature.com/articles/s41467-026-72748-3">Nature Communications 2026口腔研究</a>，公开报告9431名参与者的颊拭子宏基因组与44项代谢表型关联，第一作者为Haochen Xue。论文及其代码是当前研究需要明确区别的既有基础；不能再次将口腔—CGM关联图谱本身作为主要创新。</p>
<p><a href="https://www.nature.com/articles/s41467-023-41042-x">Nature Communications 2023</a>也已在200名糖尿病前期参与者的饮食干预中，同时研究口腔、肠道、代谢物和细胞因子。它的口腔材料是龈下菌斑，与HPP公开口腔研究的颊拭子不同。双部位饮食组合已有先例，采样生态位差别也需要在比较中保留。</p>
<p>本次提出的贡献是大规模同人配对中，<strong>口腔是否揭示肠道相近者之间的饮食相关易感性差异</strong>，以及联合结构是否超出各自的信息。这一精确问题的新颖性仍需进一步检索，不称首次。</p>
<p>公开HPP论文报告了功能分析，证明相关项目曾有功能资源；本任务可访问的配对功能表、采样日期和处理版本尚未核实。因此组成分析可作为首批方案，功能解释以实际可用数据为条件。</p>
""", "原始论文先例核查")

section("comparison","四个决定论文的比较与关键图",table(["关键比较", "方法对应的对象", "值得关注的结果"],[
 ("① 同饮食背景下的双部位特征", "在训练参与者中，分别学习口腔与肠道对同一主要饮食维度的连续表征O、G；投射到留出参与者，展示二维支持。", "相近实际饮食与G下，O仍有可靠差异。它不是两个部位应长相同菌，也不是随时间发生变化。"),
 ("② O/G平面的饮食关联斜率", "既有人日模型中，将个人内NOVA4偏离与O、G的乘积共同纳入；需要时增加限定的联合非加性项。按人聚类。", "在相近G下，O对应不同NOVA4—日Mean斜率；联合非加性需超出两个单部位项，不能由O−G显著自动推出。"),
 ("③ 差异属于背景还是餐反应", "同一批配对人员、同样餐和调整下，直接比较口腔／肠道对Pre、Net和日Mean的关联。", "可区别共同基础状态与特定反应信息。部位分工需直接差异，不能根据一项显著另一项不显著推断。"),
 ("④ 结构能否保留", "整人留出，比较gut-only、oral-only、双部位加性与联合模型。所有变换、选择和调参限于训练资料。", "评价斜率／成分差及区间；预测误差降低是辅助证据。同队列评估不称外部验证。"),
])+"""
<p><strong>核心裁决图：</strong>横纵轴为口腔与肠道的饮食相关表征，颜色为个人内NOVA4—日Mean条件斜率，覆盖实际人数密度及不确定性。只在有共同支持的范围解释，四象限可作展示，不能用任意健康／不健康标签选择结果。</p>
<p>同一饮食维度的分数需报告校准、可靠性与量尺；两部位的测量质量不同可能造成表面分离。一次菌群测量支持同窗组成特征，不能叫生态动态响应。</p>
""")

section("outcomes","什么结果会形成实质贡献",table(["结果形态", "对应结论"],[
 ("口腔在给定肠道与饮食后，仍标记饮食斜率差异", "两部位提供互补的易感性信息；单看肠道遗漏了一部分人群差异。"),
 ("联合结构超出两个单部位调节项，且在留出者中保留", "支持有限支持域内的非加性联合标记。需要效应量及精度，仍不等同生物协同机制。"),
 ("两部位主要对应共同背景Mean／Pre", "可有精度地界定它们主要是共同宿主状态标记；不能称特异餐后机制。"),
 ("仅多列相关菌、略降预测误差或同名菌同时检出", "不足以证明联合易感性、功能分工或口腔到肠道迁移。"),
])+"""
<p>现有T068／T096逐餐交互未获支持，原四菌Pre调整Net／Peak也未获支持。新母题不能预定成功；它使用双部位联合表征提出新的直接比较，并保留既有阴性。</p>
<p>吸烟、口腔健康、用药和采样间隔是重要竞争解释。具备字段时在冻结的比较中处理；缺失则保留限制。只有菌名相同、相关方向相似，不能建立口→肠→表型的迁移链。</p>
<p>心血管临床联系应围绕已有具体表型，按定义、时点与交集纳入。血糖结果提供代谢背景；它不能替代疾病事件，也不能被直接称为心血管饮食获益。</p>
""")

section("recommendation","推荐的论文定位", """
<div class="idea"><h3>工作母题</h3><p><strong>口腔与肠道菌群的联合特征，是否标记一个人的饮食相关代谢反应模式？</strong></p></div>
<p>首批研究集中在一个饮食维度、一个日级主结局、少数双部位联合特征。表型结构比较服务于解释这一问题；AMED与具体心血管表型承担有限延伸。原中介与单菌交互作为背景，研究主角变为同人两个生态系统的互补或联合关系。</p>
<p>启动前只需确认配对ID／人数、两个部位采样日期与分辨率、对应饮食与CGM窗口，以及可用口腔健康信息。该前置步骤决定联合问题能否识别，不是再扩增敏感性目录。数据存在已经确认。</p>
<p>队列规模使这样的比较更有机会得到足够精度。最终贡献是明确单看肠道何时足够、何时会遗漏饮食相关异质性；是否存在这样的遗漏由数据决定。</p>
""")

section("astra","Astra完整原文",'<div class="astra-original" id="astra-source">'+escape(source)+'</div>',"gpt-6-astra · 原文")
section("sources","来源与构想状态", """
<p>2026-10-14，按用户要求沿用gpt-6-astra，将用户新确认的口腔资源纳入研究定位。前七节为主线程综合建议与原始论文核查，第八节保留顾问原文；原文反映最初条件比较建议，总母题已按用户后续选择改为双部位对称联合问题。本轮没有进入服务器拟合新模型或核实配对交集。</p>
<div class="file-links"><a href="Astra_oral_gut_large_cohort_vision_2026-10-14.txt">Astra完整意见TXT</a><a href="Astra_review_packet_2026-10-13.json">既有冻结事实包</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">完整结果报告</a><a href="HPP_研究母题重构_Astra_2026-10-14.html">前一母题与新颖性核查</a></div>
<p>本轮属于限定先例核查，不是穷尽新颖性审查。公开论文的模态与人数不自动等同本任务可分析矩阵；研究建议也不代表期刊录用判断。</p>
"""+'<p>Astra原文SHA256</p><pre>'+hashlib.sha256(SOURCE.read_bytes()).hexdigest()+'</pre>',"来源记录")

nav=''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>'for i,(key,title,_)in enumerate(sections,1))
page="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 大队列与口腔—肠道研究定位 · Astra · 2026-10-14</title><style>"""+css+"""</style></head><body id="top"><header class="topbar"><div class="brand">HPP · 双部位研究定位<small>ASTRA / PAIRED ORAL AND GUT ECOLOGY</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header><div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 规模、双部位与科学问题</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>"""+nav+"""</nav></details><div class="sidebar-footer"><b>2026年10月14日</b>用户确认口腔资源<br>Astra明确推荐<br>配对研究尚待检验</div></aside><main><div class="hero"><div class="eyebrow">HPP / LARGE PAIRED-COHORT VISION</div><h1>数千人的双部位菌群<br>可以提出什么新问题</h1><p>把口腔与肠道放到同一个人的饮食背景中，检验联合生态能否揭示单部位遗漏的代谢易感性。</p><div class="hero-meta"><span>Astra独立判断</span><span>现有数据限定</span><span>PREDICT／地中海饮食对照</span><span>2026-10-14</span></div></div>
"""+''.join(body for _,_,body in sections)+"""<footer class="footer">大队列与口腔—肠道研究定位 · 已有资源／先例／拟议结论分别呈现 · 2026-10-14</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a><script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){d.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>"""
OUT.write_text(page,encoding="utf-8")
print(OUT)
print(f"sections={len(sections)} bytes={OUT.stat().st_size}")
