"""Render a scientific-question redesign with complete Astra opinions."""
from pathlib import Path
from html import escape
import hashlib
import re

BASE=Path(__file__).resolve().parent
FULL="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html"
ORIGINAL=BASE/"Astra_research_vision_2026-10-14.txt"
ADDENDUM=BASE/"Astra_research_vision_novelty_addendum_2026-10-14.txt"
OUT=BASE/"HPP_研究母题重构_Astra_2026-10-14.html"
original=ORIGINAL.read_text(encoding="utf-8")
addendum=ADDENDUM.read_text(encoding="utf-8")
css=re.search(r"<style>(.*?)</style>",(BASE/FULL).read_text(encoding="utf-8"),re.S).group(1)
css+="""
.sidebar summary{display:none}.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;line-height:1.95}
.idea{border:1px solid var(--line);border-radius:10px;padding:20px;margin:20px 0;background:#f6f9f2}.idea h3{margin-top:0}
@media(max-width:760px){.sidebar summary{display:list-item;cursor:pointer;font-weight:650;color:var(--green)}.sidebar-title{display:none}.sidebar nav{padding-top:12px}.astra-original{font-size:13px}}
@media print{.sidebar{display:none!important}.astra-original{font-size:10pt}.idea{break-inside:avoid}}
"""
sections=[]
def section(key,title,body,label="综合研究构想 · 待检验"):
 i=len(sections)+1
 sections.append((key,title,f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{i:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

section("vision","待细化的方向：个性化食物反应什么时候成立", """
<div class="note accent"><strong>新颖性核查更新（2026-10-14）：</strong>Cell Metabolism 2017面包随机交叉试验已利用干预前菌群预测某个人吃哪种面包血糖反应更低。下面的宽泛方向与已有研究直接重叠，不能作为创新点；其更具体的稳定性／状态／信息问题也尚未证实新颖。原Astra意见保留为构想记录。</div>
<p class="lead">将论文中心从“饮食相关菌的中介路径”转向：<strong>个人对食物的相对反应，能否形成可重复的模式；菌群能否提供关于这种模式的额外信息？</strong></p>
<p>这个问题把研究对象变为<strong>人、食物和当天状态的关系</strong>。同样的食物有人反应高、有人反应低，只是入口；我们真正要辨认的是整体水平、稳定的食物差异与日间变化。</p>
<div class="idea"><h3>待细化工作问题</h3><p><strong>自由生活中个体食物反应的可重复性，以及菌群在食物排序中的信息价值。</strong></p><p>这可作为研究方向定位，尚不是已确立的原创问题。菌群增量可能存在，也可能很小；主结论不能提前写成“菌群决定饮食反应”。</p></div>
<p>Astra推荐“个人食物排序”为主线，“稳定特征还是当天状态”为核心检验。先例核查进一步收紧了新意：个人排序与重复餐不稳定均已有研究，可能的贡献应落在<strong>哪些对比中稳定排序可辨认、状态解释了什么，以及菌群是否真的改变排序判断</strong>。</p>
""")

section("example","用一个例子说明：三种现象对应三个结论",table(["示意场景", "甲的反应", "乙的反应", "可以提出的结论"],[
 ("整体高低不同", "A比B反应低；两者都较高", "A比B反应低；两者都较低", "两人总体反应强弱不同；现象本身没有改变A/B的相对排序。"),
 ("个人排序不同", "A比B反应低", "B比A反应低", "若在重复记录中保留，支持相对食物反应存在个体差异。"),
 ("同人当天状态不同", "同样A在不同日期反应差异大", "另一人的重复模式也需估计", "需分清稳定模式、已测状态和剩余记录／测量差异。"),
])+"""
<div class="note">本表是概念示例，不是HPP结果。这里的“排序”只针对预先选定的血糖反应及可比份量，不能称某食物总体更健康。共同排序下的反应幅度差异仍可能与份量等其他个性化问题有关。</div>
<p>关键比较是同一人的A−B反应差，再比较不同人的该差值，并用另一批已有重复记录检验它能否保留。可比餐量、记录时刻和背景状态必须进入定义；日志匹配餐不自动等同标准化配餐。</p>
""")

section("directions","三条有区别的研究方向",table(["方向", "科学问题", "关键结果", "取舍"],[
 ("① 个体食物模式", "整体血糖强弱之外，个人食物对比能否可重复，并由菌群标记？", "留出记录中的A−B差值与交叉，菌群对相对反应的增量。", "推荐主线；需要跨人共享食物及同人重复支持。"),
 ("② 稳定特征与当天状态", "个人反应模式有多少稳定信息，多少与当时状态有关？", "独立重复记录中的模式一致性，以及状态信息改变了多少判断。", "与①整合为核心科学检验；不单独扩张成大量状态模型。"),
 ("③ 营养标签之外的信息", "营养相近的餐，食物身份是否仍提供跨人的反应信息？", "营养匹配对比在留出参与者中保留，是否再被菌群特征标记。", "备选；需要可靠食物编码，不能将剩余差异命名为加工机制。"),
])+"""
<p>①关注“谁对什么食物不同”，②关注“这种模式在什么时候稳定”，③关注“食物记录包含什么营养标签之外的信息”。三者不是给同一模型换名字；当前建议只以①为主，②作为其核心检验。</p>
""")

section("design","方法围绕关键比较组织", """
<h3>第一层：建立人—共同食物—重复记录矩阵</h3>
<p>以食物ID、组成和克数建立可核查模板，识别共同餐量范围。只在实际有重叠的记录中比较，不用模型给没有共同食物的人生成貌似完整的排行榜。模板几乎个人专属，或多人仅吃过一个可比模板时，应判断主问题无法在当前资料识别。</p>
<h3>第二层：分开整体反应与人×食物差异</h3>
<p>用人和食物模板的交叉层级结构，估计人的整体反应、食物平均反应以及人的食物差异；重复记录提供差异能否保留的证据。餐前水平与趋势是竞争解释，分别呈现总观察差异及给定已测状态后的差异，不能仅靠调Pre宣称差值问题已经解决。</p>
<h3>第三层：检验菌群究竟提供了什么信息</h3>
<p>将379物种组成压缩或正则化为有限特征，比较三个模型的角色：食物／营养与已测状态；再加菌群主效应；再加菌群与食物特征的相互作用。增加菌群后整体血糖水平预测改善，与A−B差值或排序改善是两个结果。</p>
<p>所有食物对选择、菌群特征提取和调参均在训练参与者内进行。整个人留出评估相对反应；不能用测试人的餐后结果拟合其个人效应，也不能把测试全期Mean作“基线”输入。若设计需要个人校准记录，应另列为有校准场景，明确其资料和后续评估窗口。</p>
<p>以参与者等权的差值误差、排序表现及区间评价，与不含菌群的强基线比较。一次菌群测量支持同窗组成信息，未证明稳定菌群特质或暴露前预测。</p>
<div class="note accent"><strong>分析设计可以更有判别力，数据设计仍是观察性的。</strong> 同人／同食物比较能减少部分混杂；吃什么仍受行为与情境影响，模型不能把记录变为随机换食实验。这里估计的是观察支持域内的相对反应及信息价值。</div>
""")

section("story","一篇论文的证据故事应该是什么",table(["顺序", "这一图／结果回答的问题", "成立形态"],[
 ("1 · 共同食物矩阵", "数据能否把不同人通过共同摄入连接起来？", "可靠模板、可比份量与重复信息足以支持核心食物对。"),
 ("2 · 决定性的反应对比", "个人差异是否超出整体高低？", "独立记录中人×食物差值保留，排序差异有精度和实际量级。"),
 ("3 · 稳定性与状态", "这种差异可否重复，当天状态解释了什么？", "能区分稳定信息、已测状态和无法解释的误差；不把剩余量全称生理状态。"),
 ("4 · 菌群信息", "菌群对相对反应是否提供新信息？", "留出参与者中增加差值／排序信息，或有精度地界定其信息很小。"),
 ("5 · 条件性心血管联系", "反应模式是否对应已有真实心血管表型？", "指标和时点可匹配时，评价与整体血糖水平之外的关联；仍属表型联系。"),
])+"""
<p>心血管部分保留你的原始研究意图，具体参与者指标与时点尚需确认。若只有一次血压／血脂／血管表型，它可回答反应模式的心血管关联，不能表示同一餐后的心血管反应，也不能将血糖排名直接称为心血管食物处方。它应服务于主问题的临床解释。</p>
<p>原871条中介路径作为发现背景或补充目录；七评分、全部扩展单菌交互不再共同争夺主线。T068／T096保留为既有线索，主假说不要求其中某株必然显著。</p>
""")

section("evidence","当前证据对新母题的支持与挑战",table(["已有结果", "对构想的含义"],[
 ("3040人、6281对日志匹配重复餐；Net绝对差中位数6.012 mg/dL", "有重复记录入口，但这个数字没有区分状态、真实餐差异和测量误差。"),
 ("T068／T096逐餐NOVA4交互q约0.779／0.910；原四菌Pre调整Net／Peak均0/4", "直接挑战简单单菌餐后敏感性叙事；不保证全群落与具体食物模式存在。"),
 ("Pre／trend增量使MAE下降0.744 mg/dL", "仅已知参与者／已见模板、绝对餐后水平场景；不能当新人的相对反应成绩。"),
 ("整体／日级饮食×菌交互、双菌B/W及N0/N1已完成", "提供背景，不是新母题的新意，也不能替代共同食物排序的直接检验。"),
 ("单次菌群CLR-Z、既有食物日志", "适合关联／信息分析；菌群时间与食物真实性仍限制预测和生物机制解释。"),
])+"""
<p>若共同食物不足，或重复资料不能辨认稳定差值，排序母题应停止。若菌群只对应整体水平，论文应如实界定其信息范围。宽区间的阴性不能证明个体模式不存在。</p>
""", "已有发现 · 构想尚未检验")

section("novelty","先例已经做了什么，潜在贡献在哪",table(["原始研究", "已建立的先例", "本项目仍需回答"],[
 ("Zeevi等，Cell 2015", "同餐反应异质性、综合个人特征的预测与独立验证／饮食干预。", "重复‘不同人反应不同’不足以形成新结论。"),
 ("Korem等，Cell Metabolism 2017", "面包随机交叉试验；干预前菌群用于预测每个人哪种面包反应更低。", "‘菌群帮助选择较低反应食物’也已有直接先例，与宽泛母题高度重叠。"),
 ("Berry等，Nature Medicine 2020", "同餐代谢异质性；Extended Data Fig.3已展示个人餐排序。", "排序存在本身不是空白；要界定稳定信息、状态与菌群在相对反应中的作用。"),
 ("Nature Medicine 2021，饮食—菌群—心代谢研究", "地中海饮食与心代谢风险指标的关联随菌群组成而异。", "将饮食×菌群交互接到心血管相关表型，也不能自动构成新意。"),
 ("Hengist等，AJCN 2025", "控制喂养下研究重复餐CGM反应的个体内高变异。", "记录中不稳定不是新发现；需要判定自由生活条件下哪些比较仍可辨认。"),
])+"""
<p>原始来源：<a href="https://pubmed.ncbi.nlm.nih.gov/26590418/">Cell 2015</a>；<a href="https://pubmed.ncbi.nlm.nih.gov/28591632/">Cell Metabolism 2017，面包试验</a>；<a href="https://www.nature.com/articles/s41591-020-0934-0">Nature Medicine 2020及扩展图</a>；<a href="https://www.nature.com/articles/s41591-020-01223-3">Nature Medicine 2021，饮食—菌群—心代谢</a>；<a href="https://pubmed.ncbi.nlm.nih.gov/39755436/">AJCN 2025，DOI 10.1016/j.ajcnut.2024.10.007</a>。</p>
<p>本轮是针对性先例核查，学术检索MCP未挂载，使用PubMed／官方期刊来源；并非系统新颖性审查。提出“菌群是否增加稳定排序信息”不表示该问题从未有人研究。</p>
<p>潜在贡献应改变读者对<strong>何时有理由个性化选食物、何时应先理解状态、菌群信息在哪里有用</strong>的理解。仅画更多个人榜单、增加模型复杂度或略提高预测成绩，不能自动形成这一贡献。</p>
<p>当前判断：宽泛母题已有直接先例；具体HPP贡献尚未建立。接下来应围绕这些原始论文的实际比较与关键图识别可回答的区别，不能由本次未找到完全相同标题便宣布空白。</p>
""", "原始先例核查＋研究判断")

section("astra","Astra完整原文：三个母题与推荐",'<div class="astra-original" id="astra-original">'+escape(original)+'</div>',"gpt-6-astra · 原文")
section("astra-novelty","Astra完整补充：更严格的新意边界",'<div class="astra-original" id="astra-addendum">'+escape(addendum)+'</div>',"gpt-6-astra · 原文")
section("sources","来源与构想状态", """
<p>2026-10-14依据用户要求调用gpt-6-astra，重构母题与证据故事。前七节为主线程综合判断及先例核查，后两节保留顾问原文。原文推荐题目经新意补充后收敛为“可重复性、状态与菌群的信息价值”。</p>
<p>当前未运行新服务器模型，尚未确认共同模板矩阵足以支持上述估计；没有新增采样、随访或干预。这个构想可被现有数据支持、修正或否定，不承诺任何期刊录用。</p>
<div class="file-links"><a href="Astra_research_vision_2026-10-14.txt">Astra三个母题原文TXT</a><a href="Astra_research_vision_novelty_addendum_2026-10-14.txt">Astra新意补充TXT</a><a href="Astra_review_packet_2026-10-13.json">既有固定事实包</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">完整结果报告</a></div>
"""+'<p>源文件SHA256</p><pre>'+hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()+'\n'+hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()+'</pre>',"来源与状态")

nav=''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>'for i,(key,title,_)in enumerate(sections,1))
page="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 研究母题重构 · Astra · 2026-10-14</title><style>"""+css+"""</style></head><body id="top">
<header class="topbar"><div class="brand">HPP · 研究母题重构<small>ASTRA / SCIENTIFIC VISION</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header><div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 科学问题与证据故事</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>"""+nav+"""</nav></details><div class="sidebar-footer"><b>2026年10月14日</b>现有数据上的母题重构<br>三条方向＋明确推荐<br>构想尚待数据检验</div></aside><main><div class="hero"><div class="eyebrow">HPP / RESEARCH QUESTION REDESIGN</div><h1>从相关菌名单<br>走向个人食物反应规律</h1><p>围绕人、食物和状态重新组织研究；辨认可重复的相对反应，并检验菌群的信息价值。</p><div class="hero-meta"><span>Astra研究构想</span><span>现有数据限定</span><span>先例核查与新意边界</span><span>2026-10-14</span></div></div>
"""+''.join(body for _,_,body in sections)+"""<footer class="footer">研究母题重构 · 示例／已有发现／拟议研究分别标注 · 2026-10-14</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a><script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){d.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>"""
OUT.write_text(page,encoding="utf-8")
print(OUT)
print(f"sections={len(sections)} bytes={OUT.stat().st_size}")
