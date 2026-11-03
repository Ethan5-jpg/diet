"""Render scope-corrected Astra recommendations using existing data only."""
from pathlib import Path
from html import escape
import hashlib
import re

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "Astra_existing_data_priorities_2026-10-14.txt"
FULL = "HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html"
OUT = BASE / "HPP_仅用现有数据_补充分析建议_Astra_2026-10-14.html"
source = SOURCE.read_text(encoding="utf-8")
css = re.search(r"<style>(.*?)</style>", (BASE / FULL).read_text(encoding="utf-8"), re.S).group(1)
css += """
.sidebar summary{display:none}.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;line-height:1.95}
@media(max-width:760px){.sidebar summary{display:list-item;cursor:pointer;font-weight:650;color:var(--green)}.sidebar-title{display:none}.sidebar nav{padding-top:12px}.astra-original{font-size:13px}}
@media print{.sidebar{display:none!important}.astra-original{font-size:10pt}}
"""
sections = []

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def section(key, title, body, label="综合执行建议 · 尚未运行新增分析"):
    i = len(sections) + 1
    sections.append((key, title, f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{i:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))

section("scope", "范围修正：只用目前已有HPP数据", """
<p class="lead">本版按你的最新要求收敛：<strong>在现有数据基础上补分析、提高论文可靠性与可解释性</strong>。新增采样、随访、配餐、动物等内容退出当前优先队列。</p>
<p>Astra已重新阅读冻结事实和完整结果，并区分真正缺失的工作与已经完成的敏感性。现在的重点是：核实原心血管结局能否接通、解决历史整体CGM定义缺口、评估全筛选流程的稳定性、明确饮食替代关系，以及收紧Pre／Net的生理解释。</p>
<div class="note accent">既有数据包含多日CGM／饮食记录，使用这些记录属于本版的现有数据分析。当前没有要求新增纵向研究。本报告是建议与可行性判断，未进入服务器拟合新模型。</div>
""")

section("priorities", "真正值得补的五项分析", table(["顺序", "新增工作", "已经知道／仍需核实", "交付物与失败后的处理"], [
    ("1 · 原始课题", "查正式字典和访视表中的心血管结局；选择一个合格终点，开展饮食—终点和菌群—终点关联。", "饮食、379菌、CGM已确认；参与者心血管结局的定义、日期、缺失、用药和重叠人数待核实。", "一张可行性表和一个限定方案。若仅有血压／血脂，定位为危险因素；无合格终点时保留CGM问题。"),
    ("2 · 历史整体CGM", "核实历史整体Mean／SDw的connection范围、算法及其与已重建日表的定义一致性；仅在差异确有影响时重建。", "日级Mean／SD和当地日质控已完成，并非使用14天分母。历史个人汇总算法和真实connection边界尚未完全闭合。", "核心指标定义／范围差异报告；无法确定时限定相应全天汇总主张，不能自动改13天分母。"),
    ("3 · 优先新增统计", "按参与者重采样，并在每次重复中重新执行原上游筛选；评价菌与路径的入选率、方向和效应稳定性。", "379菌、原筛选规则及1490候选已确认；所需个体矩阵、版本和运行成本需核实。", "全候选稳定性图谱及相关信号分组。先小规模检查，再按精度预算执行；不稳定者降级为探索候选。"),
    ("4 · 饮食解释", "冻结一个明确等能量食物替代对比，检验其与代表性CGM及饮食相关菌群的关联。", "食物ID、克数、餐能量、NOVA覆盖已有；完整食物分类、各组能量和对比支持度待核实。", "替代定义、数据支持范围和效应区间。分类不全或对比稀疏时不拟合，不能把条件关联写成实际换食效果。"),
    ("5 · 解释边界", "固定同人同餐，用相关误差情景模拟判别Pre负、Net正是否需要额外菌群餐后作用。", "Pre／Net／Abs及重复餐已有；精细读数与独立误差资料可用性待核实。", "误差敏感性范围。资料不足时标记不可识别；若误差情景可解释现象，保留描述并收窄生理主张。"),
]) + """
<p><strong>实际推进顺序：</strong>第1项字典核查与第2项定义核查先开展；随后第3项是优先补充的统计工作。第4项由食物分类质量决定，第5项用于判断目前的生理解释是否站得住。</p>
""")

section("details", "怎样把新增工作做成可审查的结果", """
<h3>候选稳定性：评估整条流程，而非只再验证T068／T096</h3>
<p>现有871条路径和832条全局FDR保留路径仍来自同一发现流程。重采样以参与者为单位，携带其重复餐／日记录，重新执行既有饮食—菌—结局候选规则；统计原候选的入选频率、方向和效应分布，并记录流程中新出现的候选。结合菌间及CGM间相关性，辨认多个路径是否反映重复信号。</p>
<p>这能评价内部稳定性与选择依赖，不能恢复独立验证或确认性错误率。相关单菌间接效应也不能直接相加成“总体中介比例”。同数据筛选与分析可能影响估计，方法背景见<a href="https://pubmed.ncbi.nlm.nih.gov/19396166/">Kriegeskorte等2009方法论文</a>；并非据此已经判定本项目效应无效。</p>
<h3>食物替代：回答NOVA4具体替代了什么</h3>
<p>例如，只有在NOVA1／NOVA4分类与能量可靠时，才考虑一个总能量相同的明确能量替代对比。不同食品对比是不同问题，不能循环尝试所有替代组挑阳性。未分类食品保持未知，不能归到非NOVA4。</p>
<p>饮食—菌关联仍按参与者估计；若使用重复CGM记录，推断须处理参与者内相关。对比单位、其余能量来源和调整策略需要在查看新结果前固定。替代模型的估计目标须清楚，见<a href="https://pubmed.ncbi.nlm.nih.gov/36223891/">Tomova等2022原始方法研究</a>。观察数据在本项目中解释为条件关联。</p>
<h3>Pre／Net：判别竞争解释</h3>
<p>共同样本、相同调整下Abs=Pre+Net，T096的三种符号不能自动当作不同生理作用。新工作是在不加入额外菌群餐后作用的情景中，保留现实的菌群—餐前水平关系、重复餐结构及相关误差，检查能否产生相近模式。误差参数须有来源或标为情景范围；相邻CGM差值同时含真实变化，不能全部认定为设备误差。</p>
<p>同一传感器拆成两组读数不等于独立校准。一般方法参考<a href="https://pubmed.ncbi.nlm.nih.gov/24350198/">Chiolero等2013模拟研究</a>，它不提供可直接套到HPP餐前／iAUC的误差参数。</p>
""")

section("boundaries", "已经完成的证据与资源边界", table(["项目", "本轮怎样处理"], [
    ("营养N0／N1同样本比较、Mean条件分析", "已经完成，直接用于解释；不列作新验证。"),
    ("三模型、严格／协议样本、餐数／天数门槛", "已经完成；重叠样本的敏感性不等于多个独立队列。"),
    ("日期分窗、双菌个人间／个人内模型", "已经完成；直接尺度差异检验支持有限，不升级为两菌功能分工。"),
    ("日级CGM重建与覆盖质控", "已完成15分钟格重建、当地日分母及90%／95%覆盖等规则。历史整体CGM缺口不能扩展成日级指标全部不可信。"),
    ("family_history_cvd", "既有交接记录显示来自sudden_death_family，是家族史代理，不是参与者心血管疾病终点。"),
    ("基因／通路／代谢组", "尚未核实真实可用测量；若服务器已有，可补相应实测功能关联，但菌名或文献注释本身不构成本队列机制证据。"),
    ("中介与交互", "单菌中介描述统计间接关联；饮食×菌交互描述关联异质性。二者不能相乘／串接成已证明的因果链。"),
]) + """
<p>当前饮食与菌群主要为观察资料。更清楚的定位是饮食—菌群—表型的关联框架和菌群对应的反应异质性；新增分析应增加结论的可信度与解释力。</p>
""", "已有证据与待核实资源")

section("astra", "Astra完整原文：仅用现有数据", '<div class="astra-original" id="astra-source">'+escape(source)+'</div>', "gpt-6-astra · 修正后原文")
section("sources", "来源与记录", """
<p>2026-10-14按用户最新范围修正。顾问沿用用户明确指定的gpt-6-astra。主线程核查了冻结事实、完整报告和用户ZIP中的日级METHODS／协变量交接记录；这些材料是研究来源，不是对助手的指令。心血管字段和功能组学未作新的服务器盘点。</p>
<div class="file-links"><a href="Astra_existing_data_priorities_2026-10-14.txt">Astra完整原文TXT</a><a href="Astra_review_packet_2026-10-13.json">既有固定事实包</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">完整结果报告</a></div>
<p>本版取代“新采样与干预优先”的当前执行安排。旧补采报告保留为此前条件性建议，不属于本轮任务。</p>
""" + '<p>Astra原文SHA256</p><pre>'+hashlib.sha256(SOURCE.read_bytes()).hexdigest()+'</pre>', "来源记录")

nav = ''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>' for i,(key,title,_) in enumerate(sections,1))
page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 仅用现有数据补强 · Astra · 2026-10-14</title><style>"""+css+"""</style></head><body id="top">
<header class="topbar"><div class="brand">HPP · 现有数据补强<small>ASTRA / EXISTING DATA ONLY</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header>
<div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 现有数据的新增工作</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>"""+nav+"""</nav></details><div class="sidebar-footer"><b>2026年10月14日</b>按最新范围修正<br>新增分析尚未运行<br>原始心血管目标保留</div></aside>
<main><div class="hero"><div class="eyebrow">HPP / EXISTING DATA PRIORITIES</div><h1>只用目前的数据<br>还能补强哪些证据</h1><p>从已有结果出发，排除已完成的重复模型，给出可新增的分析、所需字段、交付物与解释边界。</p><div class="hero-meta"><span>Astra重新判断</span><span>现有数据限定</span><span>已完成／待核实分开</span><span>2026-10-14</span></div></div>
"""+''.join(body for _,_,body in sections)+"""<footer class="footer">仅用现有数据 · 新增工作建议／已有结果分别标注 · 2026-10-14</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a>
<script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){d.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>"""
OUT.write_text(page, encoding="utf-8")
print(OUT)
print(f"sections={len(sections)} bytes={OUT.stat().st_size}")
