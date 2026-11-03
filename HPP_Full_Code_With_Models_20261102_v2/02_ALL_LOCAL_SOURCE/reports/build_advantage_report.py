# -*- coding: utf-8 -*-
"""Render strategic comparisons; no participant data or new model outputs."""
from pathlib import Path
from html import escape
import hashlib
import re

BASE = Path(__file__).resolve().parent
ASTRA = BASE / "Astra_advantage_beyond_data_2026-10-14.txt"
OUT = BASE / "HPP_研究优势与科学推进_Astra_2026-10-14.html"
astra = ASTRA.read_text(encoding="utf-8")
css = re.search(r"<style>(.*?)</style>", (BASE / "HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html").read_text(encoding="utf-8"), re.S).group(1)
css += """
.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:14px;line-height:1.95}
.idea{background:#f6f9f2;border:1px solid var(--line);padding:18px 21px;border-radius:9px;margin:18px 0}.idea h3{margin-top:0}
@media(max-width:760px){.astra-original{font-size:13px}}
"""
sections = []


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def section(key, title, body, label="主线程综合文献与Astra建议"):
    number = len(sections) + 1
    sections.append((key, title, f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{number:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))


section("position", "优势在资源组合，贡献在新的解释", """
<p class="lead">最值得建立的优势，是在同一批人中，用口腔与肠道两种生态测量理解饮食相关的动态代谢差异。数千人、多餐、多日让这个问题有机会被细致地检验；科学贡献则取决于能否发现单部位测量看不清的反应结构。</p>
<div class="idea"><h3>已经确定的研究母题</h3><p><strong>口腔与肠道菌群的联合特征，是否标记一个人的饮食相关代谢反应模式？</strong></p><p>口腔和肠道对称进入研究，不预设任何一方是主体或补充；仅使用现有数据。</p></div>
<p>“更大的队列”相对部分前作成立，但要用新的配对交集人数。“更多指标”尚不等于更全面的代谢健康。20项相关CGM指标不代表20种独立生理领域，标准餐、随机干预和血液组学也各有当前方案没有替代的价值。</p>
<div class="note accent">这是研究设计建议，不是新增分析结果。本报告保留Astra全文；前七节是主线程综合，功能解释与器官表型锚定为主线程的延伸建议。</div>
""")

section("comparison", "和最相关的前作逐项比较", table(["前作", "它已经建立的优势", "我们可能提供的区别"], [
    ("NC 2023：200名完成者", "六个月随机饮食试验；双部位微生物、1095项血清代谢物和76项细胞因子；观察干预前后变化。", "如果配对交集维持数千人，可覆盖更多生态组合及饮食相关异质性。它的干预与分子深度不能由更大观察样本替代。"),
    ("PREDICT 2020/2021", "标准餐与自由生活反应、深度肠道测量；2020研究有1002名UK参与者和另100名US验证者。", "对称检验口腔与肠道的共享、各自保留及联合信息；自由生活重复记录服务于明确的反应问题。个人食物反应及排序已有先例。"),
    ("地中海饮食 NatMed 2021：307名男性", "长期饮食、重复肠道和血液指标，已有菌群修饰饮食保护性关联的问题。", "扩大为双部位联合反应结构，并研究具体饮食对比的适用范围；加一项交互本身不是足够区别。"),
    ("HPP饮食 NatMed 2026：10068人", "大规模饮食—肠道关联与预测、持续关系及探索性模拟。", "从diet→gut预测，转向双部位联合特征如何表征饮食相关代谢异质性。不能以HPP规模本身作为新贡献。"),
    ("HPP口腔 NC 2026：9431人", "大规模口腔与44项多系统代谢表型图谱，含CGM、肝脏超声与DXA。", "把静态关联图谱推进到饮食相关的联合反应问题；新的配对数据和器官表型可用性仍需核实。"),
]) + """
<p>原始来源：<a href="https://www.nature.com/articles/s41467-023-41042-x">NC 2023</a>；<a href="https://www.nature.com/articles/s41591-020-0934-0">PREDICT 2020</a>；<a href="https://www.nature.com/articles/s41591-020-01183-8">PREDICT 2021</a>；<a href="https://pubmed.ncbi.nlm.nih.gov/33574608/">地中海饮食 2021</a>；<a href="https://www.nature.com/articles/s41591-026-04312-x">HPP饮食 2026</a>；<a href="https://www.nature.com/articles/s41467-026-72748-3">HPP口腔 2026</a>。</p>
<p>各研究人数对应不同纳入及测量口径，不能作为完全配对人数直接相除。双部位采样位置也不同：NC2023为龈下菌斑，HPP口腔前作为颊拭子。已有检索未穷尽所有附录，不据此声称首次或前作完全没有联合分析。</p>
""")

section("resources", "哪些数据优势已经成立，哪些仍是潜力", table(["维度", "当前事实", "可转化成的研究能力"], [
    ("人数", "已整理肠道—饮食—CGM主共同基础4139人，CGM汇总7493人；用户确认口腔资源存在。口腔＋肠道＋饮食＋CGM交集N未核实。", "配对人群若足够大，可估计更细的联合模式及不确定性，避免只研究最常见生态组合。"),
    ("动态表型", "真实饮食日志、多餐及多日CGM；日级与餐次已有分析基础。", "区分总体血糖水平与饮食相关的反应差异；检验估计出的模式能否在其他记录中保留。"),
    ("双部位", "同一HPP平台已有口腔和肠道资源，新的具体配对时点仍待核实。", "共同与特异信息可以在同一人、同一饮食范围内对称比较，而不是跨队列拼接。"),
    ("多系统健康", "公开HPP口腔前作含肝脏超声、DXA等。当前饮食项目尚未正式接好全部器官／心血管结局。", "若现有合格配对表可用，可问动态反应模式是否伴随更广的代谢负担。不能把静态器官表型称为餐后反应。"),
    ("功能层级", "公开HPP前作已有基因／通路资源，当前双部位功能矩阵的版本、交集和时间待核实。", "有条件把物种现象推进到有限、明确的微生物功能假说。DNA通路丰度表示功能潜能，不能直接当代谢活动。"),
]))

section("science", "三件可以改变论文内容的事", """
<h3>一，定义“饮食反应差异”究竟是什么</h3>
<p>主要问题可以限定为：在相近的总体血糖水平下，饮食暴露与日均血糖、日内波动的关联幅度或形状是否仍不同？这会把一个风险高低的标记推进成饮食相关的反应表征。开始时选一个主要饮食对比和少数有明确区别的领域，餐后反应承担独立的次要问题。</p>
<p><strong>示例仅是假设：</strong>两个人平均血糖相近，其中一人的日内波动与NOVA4比例呈较强正关联，另一人的关联较弱。如果联合菌群特征能标记这种差异，并且在其他记录中仍有支持，得到的就是有意义的反应结构。并不要求出现方向相反，更不能预先宣称一定有这种结构。</p>
<p>方法可采用带部分汇聚的层级反应模型：分开个人间饮食差异与个人内每日变化，处理餐前或当时的状态，估计可识别的曲线差异。现有个人内／个人间分析已做；新增内容应是双部位联合反应表征及其可重复程度。每个人的记录不足时，不强行给每人一个可靠斜率或食物排序。</p>
<p>Mean和GMI高度重叠；Pre、Net、Abs的恒等关系不能制造三种独立生理机制。若所谓模式只是同一血糖水平的重复投影，应收窄主张。</p>
<h3>二，解释“为什么两个部位都值得测”</h3>
<p>同一配对人群中，对称比较无菌群、口腔、肠道与联合表征，进一步区分共享信息和各自保留的信息。要回答它们是重复标记同一种宿主状态，还是分别帮助识别某些已定义的反应成分。</p>
<p>“两部位特征一致／不一致”可以成为解释性问题：口腔与肠道不同组合，是否对应不同饮食关联结构？组合应优先保留连续特征，避免从同一结局挑菌、分组，再宣布存在反应类型。不预设口腔负责炎症、肠道负责发酵，也不把同名菌当作口腔向肠道迁移。</p>
<p>多视图表征或信息分解可作为工具，选择和变换仅在训练参与者中完成，并以整人留出评估。同容量比较及不确定性使“联合更好”可以被认真检验。加性互补也有价值，不要求三重交互或某种复杂网络一定显著。</p>
<h3>三，找到联合信息何时会改变饮食判断</h3>
<p>选一个有共同摄入支持、解释清楚的食物或营养对比，问联合特征是否改变其关联幅度、方向或适用人群。也可能发现另一个对比主要由食物本身解释，双部位信息几乎无帮助。两种结果共同界定精准营养的适用范围。</p>
<p>这里的科学回报是一个有范围的原则：哪些饮食相关差异需要人的微生物生态信息，哪些不需要。仅降低一点平均预测误差、却不改变任何实质解释，不能完成这个目标。饮食对比应先确定，不能从大量食物中事后挑出漂亮交叉。</p>
<div class="note">自由生活数据估计的是观察条件下的饮食关联差异。反事实饮食处方、换食物的因果获益、或菌群导致反应差异，均不能由这个设计直接推出。整人留出是本队列内部验证。</div>
""")

section("biology", "进一步建立生物学含义：两条现有资源路线", """
<p>以下是主线程对Astra建议的延伸，是否执行取决于现有资源的实际配对情况。</p>
<h3>功能是否比物种名称更能组织反应模式？</h3>
<p>若两部位宏基因组功能矩阵可用，可以问：物种组成不同、但某些饮食相关功能潜能相近的人，是否有更相近的反应结构？先围绕选定饮食对比确定少量功能模块，再检查两部位共享或特异的功能信息是否解释物种层面的现象。</p>
<p>这使“发现了一批菌”推进到“哪类微生物能力与这种饮食相关差异一致”。功能差异必须实际测到；宏基因组只能提供潜能层面的证据，不是代谢通量或机制证明。</p>
<h3>动态模式有没有更广的代谢健康含义？</h3>
<p>如果现有肝脂肪、内脏脂肪或合格血脂／血压数据能配对，选少数器官或心代谢指标问：联合标记的饮食反应模式，是否对应不同的多系统代谢负担？器官表型的价值在于给动态模式一个健康含义，而不是把所有指标塞进相关热图。</p>
<p>这些分析仍为观察关联。当前家族心血管病史代理不能作为参与者心血管病结局；未经定义和核实的疾病事件也不能纳入论文主张。</p>
""")

section("results", "已有结果怎样参与新主线", """
<p>已有统计中介和单菌关联可提供候选线索。它们尚未证明联合口腔—肠道反应模式，也不能据路径数建立多条独立机制。</p>
<p>T096的个人内日级NOVA4—血糖关联值得作为后续问题的线索；NOVA4与T068／T096的餐次交互尚未获得支持。已有相似记录餐次也存在可见差异，不能把日志匹配直接当标准化同餐或稳定个人反应的证据。</p>
<p>因此，新的主分析应从明确的饮食对比、反应定义和对称联合表征开始。既有候选用于公开说明探索依据，不能先认定它们代表两个不同生理系统，再寻找支持。</p>
<div class="file-links"><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">已有结果与阴性结果</a><a href="HPP_联合口腔-肠道_母题新颖性核查_2026-10-14.html">已完成的先例核查</a></div>
""")

section("story", "一篇主论文可围绕四个证据环节", table(["拟议图／证据环节", "需要回答的事", "若没有支持，怎样收窄结论"], [
    ("1. 饮食相关反应结构", "哪些少数反应成分超出总体水平，能在其他记录中保留？", "将结论限定为总体代谢状态标记，不称稳定个人饮食反应。"),
    ("2. 双部位信息组织", "共享与各自保留的信息，是否对应不同反应成分？", "诚实报告冗余或有限增量，不强求生态分工。"),
    ("3. 一个具体饮食对比", "联合信息在哪种条件下改变关联判断，又在哪种条件下不改变？", "限定适用饮食范围，不推出广泛个性化处方。"),
    ("4. 功能或健康锚点", "现有功能潜能或器官表型能否赋予上述模式更清楚的含义？", "资源或结果不足时，保留为探索，不制造机制链。"),
]) + """
<p><strong>拟议定位：</strong>人口尺度的双部位微生物生态与饮食相关代谢异质性。这个定位要由结果支撑：它说明为什么值得联合测量、看到了什么差异、以及这种差异在什么范围内有用。</p>
<p>篇幅应集中于这些证据；不要求所有饮食评分、全部CGM指标和全部中介路径进入主文。方法的复杂度应服务估计问题，不能作为期刊层级的替代。</p>
""")

section("astra", "Astra原文与来源边界", '<div class="astra-original" id="astra-source">' + escape(astra) + '</div><div class="file-links"><a href="Astra_advantage_beyond_data_2026-10-14.txt">Astra原文TXT</a><a href="joint_oral_gut_novelty_search_2026-10-14.json">逐篇文献与检索记录</a></div><p>2026-10-14。比较依据为已核查原始研究与项目冻结事实，正文、摘要、扩展图或附录核查深度见检索记录。本轮未拟合新模型，配对交集、功能矩阵和心血管结局仍待实际数据核实。</p><p>Astra原文 SHA256</p><pre>' + hashlib.sha256(ASTRA.read_bytes()).hexdigest() + '</pre>', "gpt-6-astra · 原文完整保留")

nav = ''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>' for i, (key, title, _) in enumerate(sections, 1))
page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 研究优势与科学推进 · 2026-10-14</title><style>''' + css + '''</style></head><body id="top"><header class="topbar"><div class="brand">HPP · 研究优势与科学推进<small>PAIRED ORAL–GUT / DIET / DYNAMIC METABOLISM</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header><div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 比较、问题与证据</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>''' + nav + '''</nav></details><div class="sidebar-footer"><b>2026年10月14日</b>仅用现有数据的研究建议<br>同人双部位，对称比较<br>未运行新模型</div></aside><main><div class="hero"><div class="eyebrow">HPP / SCIENTIFIC CONTRIBUTION</div><h1>数据优势怎样成为<br>有辨识度的科学贡献</h1><p>对照关键前作，区分实际资源与待证实能力；把联合菌群母题收敛为可检验的反应结构、信息组织与饮食适用范围。</p><div class="hero-meta"><span>Astra建议全文</span><span>前作逐项对照</span><span>功能／健康含义延伸</span><span>2026-10-14</span></div></div>''' + ''.join(body for _, _, body in sections) + '''<footer class="footer">研究策略建议 · 实际结果、假设示例与资源条件分别标明 · 2026-10-14</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a><script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){d.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>'''
OUT.write_text(page, encoding="utf-8")
print(OUT)
print(f"sections={len(sections)} bytes={OUT.stat().st_size}")
