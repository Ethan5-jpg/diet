"""Render the frozen Astra advice with a separately labelled editorial summary."""
from pathlib import Path
from html import escape
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / "Astra_Nature_subjournal_strategy_2026-10-13.txt"
FULL_REPORT = "HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html"
OUTPUT = BASE / "HPP_Nature子刊研究提升建议_Astra_2026-10-13.html"
source = ORIGINAL.read_text(encoding="utf-8")
original_sha = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
css = re.search(r"<style>(.*?)</style>", (BASE / FULL_REPORT).read_text(encoding="utf-8"), re.S).group(1)
css += """
.sidebar summary{display:none}.priority{font-weight:700;color:var(--green)}
.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;line-height:1.95;font-size:14px;margin:0}
.decision{border:1px solid var(--line);border-radius:8px;padding:18px 20px;margin:18px 0;background:#f8faf6}
.decision h3{margin-top:0}.section-top .source-tag{font-size:11px}
@media(max-width:760px){.sidebar summary{display:list-item;cursor:pointer;color:var(--green);font-weight:650;font-size:13px}.sidebar-title{display:none}.sidebar nav{padding-top:12px}.astra-original{font-size:13px}}
@media print{.astra-original{font-size:10pt}.sidebar{display:none!important}}
"""

sections = []
def section(key, title, body, label):
    sections.append((key, title, '<section class="report-section" id="' + key +
                     '"><div class="section-top"><span class="section-number">' +
                     f'{len(sections)+1:02d}' + '</span><span class="source-tag">' +
                     label + '</span></div><h2>' + title + '</h2>' + body + '</section>'))

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+escape(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'

section("judgment", "最重要的研究判断", """
<p class="lead">继续做有价值。希望竞争高选择性的 Nature 系列期刊，当前最重要的提升是<strong>明确一个新的、可被反驳的结论，并获得离开发现过程的新证据</strong>。现有871条中介路径和109种菌提供候选图谱；路径数本身不能代替独立验证、生物学解释或临床含义。</p>
<p>Astra优先的CGM问题是：<strong>NOVA4与血糖水平及波动的关联，哪些能在同一个人的日间变化中观察到；菌群是否标记这种差异，并提供超出餐前状态和营养组成的信息？</strong> 这是拟进一步检验的问题，尚未确认为机制或临床应用。</p>
<p>当前优先候选是T096的个人内NOVA4—日Mean交互；T068的个人间SD交互作为不同估计目标的第二候选。T096给定Mean后的SD证据较弱。该优先级是在查看已有结果后提出，不能回写成原先预设主假设。</p>
<div class="note accent"><strong>保留你的原始心血管目标。</strong> 首批先核查真实心血管终点、访视／事件时间、匹配人数与信息量，再决定心血管能否成为主线。CGM可以提供代谢背景，但不能替代心血管疾病终点。当前没有证据证明两种菌各自“负责一个时间尺度”。</div>
<p>已经完成的N0/N1同样本营养比较、Mean条件模型、直接尺度差值及日期分离衔接，应直接用于解释。新增工作应解决尚未关闭的缺口，避免重复已完成模型。</p>
""", "综合建议 · 基于Astra与现有结果")

section("priorities", "我建议先做的五项工作", table(["优先级", "工作及所回答的问题", "最小交付物／决定标准"], [
("1 · 真实资源审计","心血管结局、重复访视、功能基因／通路、代谢组和样本库存，哪些真正可用？","字段、定义、访视／时间、与饮食菌群重叠N、事件数或有效范围、缺失／药物、访问条件。未确认不预设存在。"),
("1 · 关闭核心测量缺口","14天代理窗口尾缺、实际设备观察期、Mean／日SD能否按真实元数据复算？","核查核心量和少量关键样本，不先重算全部20指标。不能为多纳入样本自动改13天分母；若核心量不可复现，收窄或暂停相应主张。"),
("2 · 冻结科学问题","当前论文主结局是明确心血管表型，还是动态血糖异质性？","一项主暴露、少数主结局和候选、估计目标、完整多重性规则。方向、单位和失败标准从现在冻结，记录探索历史。"),
("3 · 有终点的脆弱性检查","T096水平／升幅模式是否受Pre误差支配？菌交互是否由零值、批次或少数人支配？","针对性基线误差、组成、批次及影响诊断。报告量级和区间；失败则降级物种／生理主张，不自动换下一株显著菌。"),
("4 · 真正新数据验证","关键结论能否离开已经看过的发现队列？","首选外部或真正未使用的新批次；固定定义、目标与检验家族。若新资料精确排除有意义效应，停止原主张；宽区间则保持未定。"),
("5 · 候选稳定后再升级","需要饮食实验或实测功能链，才能回答哪一个尚未解决的问题？","随机交叉饮食研究匹配日级问题；功能测量匹配物种／菌株、底物和产物。饮食随机化仍不自动使菌群成为因果因素。"),
]) + """
<p>目前无法给出可靠的固定样本量或完成周期。新研究需要基于参与者层面、聚类、暴露变化、失访及保守收缩后的效应进行功效／精度规划；十万餐次不能当十万独立菌群样本。</p>
<div class="note">三模型、严格人群、餐数门槛、旧队列重新拆分和同队列日期分窗均不能替代真正独立验证。对已看过数据进行完整流程重采样可评价选择稳定性及乐观程度，但不能恢复独立性。</div>
""", "具体执行优先级 · 尚未实施")

section("route", "CGM与心血管，怎样决定主线", """
<div class="decision"><h3>有可靠的新发心血管事件和时间</h3><p>先核查饮食／菌群先于事件、事件数、随访、删失和匹配样本。如果信息量允许，心血管可成为原项目主线，CGM作为预先界定的代谢补充。不能自动串成饮食→菌→CGM→事件的因果中介。</p></div>
<div class="decision"><h3>有可靠血管／心脏结构功能表型</h3><p>围绕一个生物学明确的主表型建立问题。当前CGM候选菌可以进入候选检验，但不能代表心血管特异菌的完整发现；新的心血管探索应另列层次。</p></div>
<div class="decision"><h3>仅有横断面血压／血脂，或结局质量不足</h3><p>可研究心代谢危险因素，但仅换结局未必增加新意。若时间、质量或重叠不足，当前稿件可以收敛为动态CGM研究，心血管保留为后续项目；这属于研究决策，不把CGM改名为心血管终点。</p></div>
<p>下一批交付物应是<strong>数据资产与心血管时间表、核心窗口／表型审计、先例对照、冻结问题与验证计划</strong>。这些材料足以决定继续投入哪里。</p>
""", "原始目标的分流决策")

section("journals", "期刊范围与需要的证据", table(["候选期刊", "官方范围", "本项目的条件性定位判断"], [
("Nature Communications","要求对各领域专家有意义的重要推进。","作为高水平目标，需要明确新结论、可靠测量和扎实验证；目前不据关联条目数判断达到门槛。"),
("Nature Cardiovascular Research","涵盖基础、转化、临床及公共卫生心血管研究。","只有真实心血管结局成为中心问题时才适合考虑；单靠CGM间接关联不足以确认方向匹配。"),
("Nature Metabolism","重视代谢与稳态过程的理解，包含生理、临床、营养及组学研究。","若形成清楚的代谢生理或干预／功能解释，可作为进一步升级的目标；不把动物实验说成所有稿件通用硬条件。"),
("Communications Medicine","选择性临床、转化和公共卫生期刊。","若重点是有临床意义、可复现的人群或验证结果，可作为候选；没有临床价值证据时仍需补强。"),
]) + """
<p>官方范围来源：<a href="https://www.nature.com/ncomms/ncomms/aims">Nature Communications</a>、<a href="https://www.nature.com/natcardiovascres/aims">Nature Cardiovascular Research</a>、<a href="https://www.nature.com/natmetab/aims">Nature Metabolism</a>、<a href="https://www.nature.com/commsmed/aims">Communications Medicine</a>。定位是依据范围与当前证据作出的建议，不是录用预测。</p>
<h3>先例意味着什么</h3>
<p>饮食—菌群—代谢标志物的综合关联已有<a href="https://www.nature.com/articles/s41591-020-01183-8">Nature Medicine 2021的1098人PREDICT研究</a>。<a href="https://www.nature.com/articles/s41586-025-09854-7">Nature 2026卷期的最新大规模研究</a>纳入超过34000名参与者，另有超过7800份公共样本与746名饮食干预参与者；该文在线发表为2025-12-10。据此，我的判断是当前项目更应在动态血糖问题和新证据上形成区别，而非泛化健康菌名单或关联数量。</p>
<p>这是有限、针对性的先例核查，不是完成了系统新颖性审查。公共测序数据不必然包含可匹配的CGM、饮食和临床表型；例如PREDICT非测序数据需要访问申请，不能预设可立即完成外部复制。</p>
""", "官方范围＋综合判断")

# Render the complete frozen adviser text without rewriting its wording.
chunks = re.split(r"\n(?=[一二三四五六七八九十]+、)", source)
intro = chunks[0].strip()
section("astra-scope", "Astra原文 · 评估范围", '<div class="astra-original">'+escape(intro)+'</div>', "Astra独立意见 · 原文")
for i, chunk in enumerate(chunks[1:], 1):
    title, _, body = chunk.partition("\n")
    section("astra-"+str(i), "Astra原文 · "+escape(title), '<div class="astra-original">'+escape(body.strip())+'</div>', "Astra独立意见 · 原文")

section("provenance", "原文、固定事实与来源记录", """
<p>建议日期2026-10-13。本轮按用户明确要求使用gpt-6-astra提供一份独立策略建议，不把它呈现为三个互盲审稿人或期刊编辑意见。未运行新服务器模型，未修改原结果报告。</p>
<div class="file-links"><a href="Astra_Nature_subjournal_strategy_2026-10-13.txt">Astra完整原文TXT</a><a href="Astra_review_packet_2026-10-13.json">固定事实包</a><a href="Nature_subjournal_literature_check_2026-10-13.json">文献／期刊核查记录</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">原完整结果报告</a></div>
<p>前四节为主线程综合建议，后续标为“Astra原文”的部分保持顾问文本。现有结果、未来建议和期刊范围分层列出，未来工作没有写成已完成发现。</p>
<p>冻结Astra原文SHA256</p><pre>""" + original_sha + """</pre>
""", "来源与边界")

nav = ''.join('<a href="#'+key+'"><span>'+f'{i:02d}'+'</span>'+title+'</a>' for i,(key,title,_) in enumerate(sections,1))
page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HPP · Nature子刊研究提升建议 · Astra · 2026-10-13</title><style>""" + css + """</style></head><body id="top">
<header class="topbar"><div class="brand">HPP · 研究提升建议<small>ASTRA / RESEARCH STRATEGY</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header>
<div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 研究决策与完整原文</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>""" + nav + """</nav></details><div class="sidebar-footer"><b>2026年10月13日</b>Astra独立建议＋综合执行优先级<br>未来方案尚未实施<br>保留原始心血管研究目标</div></aside>
<main><div class="hero"><div class="eyebrow">HPP / NEXT RESEARCH DECISIONS</div><h1>面向 Nature 子刊<br>下一步该补哪条证据</h1><p>聚焦可检验的问题、测量可靠性和真正的新数据验证。完整保留Astra独立建议，并区分现有发现、未来计划与期刊定位判断。</p><div class="hero-meta"><span>Astra独立评估</span><span>固定已有证据</span><span>期刊官方范围核查</span><span>建议日期 2026-10-13</span></div></div>
""" + ''.join(html for _,_,html in sections) + """<footer class="footer">研究提升建议 · 独立顾问意见与综合建议分层呈现 · 2026-10-13</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a>
<script>(function(){var details=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){details.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>"""

assert len({k for k,_,_ in sections}) == len(sections)
OUTPUT.write_text(page, encoding="utf-8")
print(OUTPUT)
print("sections", len(sections), "bytes", OUTPUT.stat().st_size, "astra_sha256", original_sha)
