# -*- coding: utf-8 -*-
"""Build the methods proposal from literature checks and preserved Astra advice."""
from pathlib import Path
from html import escape
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
ASTRA = BASE / "Astra_method_architecture_2026-10-14.txt"
OUT = BASE / "HPP_方法优势与分析架构_Astra_2026-10-14.html"
astra = ASTRA.read_text(encoding="utf-8")
record = json.loads((BASE / "method_comparison_sources_2026-10-14.json").read_text(encoding="utf-8"))
css = re.search(r"<style>(.*?)</style>", (BASE / "HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html").read_text(encoding="utf-8"), re.S).group(1)
css += """
.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:14px;line-height:1.95}
.formula{background:#f1f6f2;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin:18px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;overflow-wrap:anywhere}
.idea{background:#f6f9f2;border:1px solid var(--line);border-radius:9px;padding:18px 21px;margin:18px 0}.idea h3{margin-top:0}
@media(max-width:760px){.astra-original{font-size:13px}.formula{font-size:12px}}
"""
sections = []


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def section(key, title, body, label="主线程综合核查与Astra建议"):
    n = len(sections) + 1
    sections.append((key, title, f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{n:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))


section("answer", "首选：围绕饮食条件对比建立双部位反应模型", """
<p class="lead">方法目标是估计饮食摄入变化对应的代谢变化，以及哪些可重复差异能由口腔与肠道组成共同标记。数据支持时，进一步联合学习跨饮食比较的反应结构。优势应在相同任务上优于强基线，并形成清楚的新结论。</p>
<div class="idea"><h3>一个主架构，三个相互配合的部分</h3><p><strong>在重复记录上直接估计反应差异 → 用双部位信息组织跨饮食／结局的响应系数 → 在完全留出参与者上验证饮食相关变化。</strong></p><p>既有统计中介结果可以提供线索。新主分析不先筛一批“健康菌”，再靠中介路径连接故事。</p></div>
<p>单暴露、双结局的随机斜率模型是可行性入口，本质仍是交互模型。真正值得测试的推进层，是多个饮食对比或有支持的非线性暴露范围之间，是否存在能被双部位组成联合标记的共同响应结构。</p>
<div class="note accent">本报告没有运行新模型。前七节为主线程综合，第八节完整保留Astra建议。单项统计技术已有前作；能否构成方法优势须由实际比较建立。</div>
""")

section("predecessors", "前作的方法不能简化成“相关加中介”", table(["研究", "已核查的方法", "对我们方法定位的约束"], [
    ("PREDICT 2020/2021", "标准／重复餐，变异分解，机器学习预测；肠道物种和功能特征；有针对不同对象的外部验证。", "个体异质性、食物排序和个性化预测已有先例；需要证明双部位信息针对饮食反应差异的具体作用。"),
    ("AMED Nature Medicine 2021", "重复采样的线性混合模型；饮食、菌群及乘积项。", "重新批量扫描菌×饮食，不能独立承担方法贡献。"),
    ("NC 2023", "随机饮食干预；双部位组学变化、中介及菌株分析；预训练肠道模型预测代谢物变化。", "不能把观察模型当作对随机化识别的替代，也不能把两部位都纳入视为首次。"),
    ("Toumi与Salathé 2026预印本", "混合效应＋XGBoost；个人营养斜率和低维结构；重复性、标准餐迁移和补充整人留出。", "随机斜率、去除截距、低秩反应轴和整人留出本身均已有。"),
    ("Sugimoto等 HPP／10K 2026", "CGM因子分析与曲线重构；跨数据泛化。", "重新把总体CGM降成几个轴不足以构成新意。"),
    ("AEGIS正式论文 2026", "多层功能主成分、function-on-scalar regression及个人功能随机效应。", "整条曲线及FDA已有先例；若采用，要回答双部位特有的响应问题。"),
]) + """
<p>原始来源：<a href="https://www.nature.com/articles/s41591-020-0934-0">PREDICT2020</a>；<a href="https://www.nature.com/articles/s41591-020-01183-8">PREDICT2021</a>；<a href="https://pubmed.ncbi.nlm.nih.gov/33574608/">AMED2021</a>；<a href="https://www.nature.com/articles/s41467-023-41042-x">NC2023</a>；<a href="https://www.medrxiv.org/content/10.64898/2026.09.14.26362991v1">Toumi与Salathé预印本</a>；<a href="https://www.nature.com/articles/s43856-026-01523-8">HPP／10K2026</a>；<a href="https://link.springer.com/article/10.1186/s12874-025-02748-2">AEGIS2026正式论文</a>。</p>
<p>最新预印本尚不按已同行评议证据处理，但其方法先例需要纳入。逐篇核查深度、版本关系及访问限制见来源记录。</p>
""")

section("estimand", "第一部分：在全部重复记录上估计饮食反应差异", """
<h3>首先确定估计的量</h3>
<p>对于同一背景条件，在实际观测支持的摄入范围中，比较某饮食暴露从d变成d+δ时的模型预期结局。O和G分别为口腔、肠道组成特征；c为已记录背景。</p>
<div class="formula">Δ<sub>k</sub>(o,g;c,d,δ) = m<sub>k</sub>(d+δ,o,g,c) − m<sub>k</sub>(d,o,g,c)</div>
<p>它表示<strong>条件关联对比</strong>。实际自由生活日对中的睡眠、其他食物或活动也可能不同；模型不把全部血糖差归于饮食，更不直接估计因果获益。</p>
<h3>一阶段估计为什么有价值</h3>
<p>直接联合拟合全部重复观测：总体饮食关联、个人的水平差异、菌群解释的响应差异，以及菌群没有解释的剩余个人差异。稀疏个人记录通过部分汇聚估计，保留估计噪声。</p>
<div class="formula">代谢结局 = 个人水平 + 总体饮食／背景关联 + 双部位标记的饮食差异 + 剩余个人反应差异 + 观测噪声</div>
<p>避免先分别拟合每人的不稳定斜率，再把这些斜率当作没有误差的结局去关联菌群。个人内和个人间饮食差异分开；年龄、BMI等与饮食的关系在各竞争模型中一致处理，避免将其异质性归给菌群。</p>
<p>初版可用日级NOVA4、日Mean与日内SD，后两者联合建模但不视为独立生理系统。选择源于现有探索，正式评价仍属内部验证。部分汇聚不能修复饮食漏报、CGM误差或单次菌群测量误差。</p>
<p>基础方法来源：<a href="https://www.jstatsoft.org/article/view/v067i01">混合效应模型</a>与<a href="https://pubmed.ncbi.nlm.nih.gov/31179172/">层级GAM</a>。现有B/W、Mean条件和Pre调整已做，不能重新列为新增方法。</p>
""")

section("response-structure", "第二部分：由反应学习共同方向，避免逐项穷举", """
<p>完整推进层需要至少两个科学上不同、实际摄入有支持的饮食对比，或者足以估计曲线形状的暴露范围。例如，在资源允许时考察食物加工相关差异与一个能量一致的营养替代对比，而不是把高度相关的多个健康饮食评分当作独立维度。</p>
<p>用少量平滑基函数描述饮食暴露，每人对应一个“饮食基函数×代谢结局”的响应系数向量。允许非线性，但不凭稀疏数据拟合任意复杂曲线。</p>
<div class="formula">B = (O A<sub>O</sub> + G A<sub>G</sub>) Vᵀ + 剩余个人系数</div>
<p>B的每一行对应一个参与者的响应系数；V表示跨饮食基函数和结局的少数共同方向；A<sub>O</sub>和A<sub>G</sub>把两部位特征对称映射到这些方向。它是拟议结构，不是本项目已经估计的矩阵。</p>
<p><strong>该层改变了信息共享方式：</strong>多种饮食比较及结局共同帮助估计响应；菌群特征因其对响应的贡献被保留，而非仅因丰度方差大进入模型。若真实存在共同结构，可能降低稀疏个人估计的不确定性，并发现哪些饮食对比沿相近或不同方向变化。</p>
<p>对应的科学问题是：多个饮食相关差异，是否受到同一个能跨参与者保留的联合标记结构约束？这超出把大量独立交互检验排列成热图，但共同结构必须实际成立。</p>
<p>采用低秩前，要与同等灵活、分别拟合各对比的模型比较；秩、基函数及惩罚在训练参与者内选择。低秩方向有旋转或符号不确定性，应比较对齐后的子空间与预测曲面，不能把未经对齐的单轴变化当作失败，也不能把同一统计方向命名为某生物机制。</p>
<p>方法构件来源：<a href="https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.734178">稀疏降秩回归</a>与<a href="https://pubmed.ncbi.nlm.nih.gov/31179172/">层级平滑模型</a>。这些构件已有；在本项目中的联合结构与获益尚待检验。</p>
""")

section("validation", "第三部分：检验对变化的预测，而不只检验水平", table(["环节", "具体设计", "能排除的替代解释"], [
    ("完全留出参与者", "外层按人划分，内层按人调参。测试者CGM不用于估计个人截距或随机斜率。", "模型只是记住了某个已见参与者的血糖水平。"),
    ("验证日对差异", "按事先定义的摄入差异和共同支持域选不重叠日对；不依据结局选对。预测Y(t)−Y(s)，个人截距抵消。", "整体误差下降主要来自谁的总体血糖更高。"),
    ("正确评价变化", "其他已记录背景在两日可能不同，均用同一模型处理；每人等权，按人计算不确定性，报告实际单位误差和校准。", "把全部日差叫作饮食效应，或把日对当独立人数。"),
    ("对称信息比较", "同人群、同划分比较背景、背景+O、背景+G、背景+O+G。分别检查给已有G加入O和给已有O加入G。", "联合增益源自不同纳入样本，或只能重复标记总体代谢状态。"),
    ("强基线竞争", "层级GAM、预限定交互的正则化模型；完整版本另与非线性独立对比模型竞争。输入、支持域及目标一致。", "把普通非线性、参数更多或验证任务改变误当方法优势。"),
]) + """
<p>首要比较可用留出参与者的变化预测误差、相应校准和不确定性。它仍在有噪声的观测差异上评价，并没有拿到每人的真实饮食斜率。预期最小有用增益需在正式模型比较前根据单位、噪声与用途确定。</p>
<p>两方向有可靠条件增益可称互补信息，不能称生物协同。先测试加性分块；没有证据时不加高维O×G乘积或强求三重交互。</p>
<p>单次菌群常晚于CGM，当前能检验的是回顾性标记能力。内部留出不能改称外部验证或前瞻部署；加入因果森林／双重机器学习也不能自动补上时序与未测混杂。</p>
""")

section("conclusions", "这种方法可能带来什么新的结论", table(["假设性结论，不是已有发现", "必须有的证据", "不能替代这些证据的结果"], [
    ("双部位组成可共同标记多个饮食对比的反应结构。", "响应方向或曲面跨训练划分保留；完全留出参与者上的变化校准；比独立对比模型有实质收益。", "几个菌×饮食显著；同一菌群向量算出的分数彼此一致。"),
    ("总体血糖相近的人仍有饮食相关差异，两个部位分别补充信息。", "水平与响应差异分开；对称条件增益；支持域内的新参与者观测差异与预测对应。", "绝对血糖MAE略降，或按训练集结局分组后得到差异。"),
    ("微生物信息主要标记日级模式，未能可靠改善逐餐反应；或只在某摄入范围可用。", "对应层级／范围的直接比较及足够精度；阴性结果能排除事先定义的有用增益。", "一个尺度显著另一个不显著，或把共同支持不足当生理边界。"),
]) + """
<p>例如，联合模型可能发现两种饮食对比对应相近的响应方向，也可能发现它们相互独立；曲线也可能在部分范围较陡、部分范围接近平台。这些都应由模型和留出观测支持，不能事先命名敏感型或保护型。</p>
<p>前作已覆盖多个构件，不能声称别人原则上得不到这些结论。当前可争取的区别，是把双部位信息放进明确的饮食反应估计与验证框架，建立尚未被前作清楚证明的结果。</p>
""")

section("execution", "实施顺序：入口、推进层与两个条件替代", table(["阶段", "要做的内容", "进入下一步的条件"], [
    ("第一轮：可行性入口", "核实四模态交集及时间；一个饮食对比、两个日级结局、四种信息输入和强基线；输出变化校准及双向信息增益。", "足够个人内摄入跨度，联合信息在留出变化上有可解释价值。"),
    ("第二轮：完整推进层", "增加少数非冗余饮食对比或有限非线性基；响应导向的分块低秩模型，比较独立对比模型。", "共享结构比独立估计有稳定收益；对齐后子空间与曲面可保留。"),
    ("替代A：餐后函数模型", "若问题转向时间形状，用功能型混合模型分析曲线，并考虑餐前状态和餐次重叠。", "分辨率、窗口质量及问题本身支持；相比标量目标确有额外信息。"),
    ("替代B：共享／特有组成分解", "JIVE／AJIVE等表征进入同一反应框架，与直接分块正则化比较。", "组成表征稳定且提高解释或预测；不把统计共有成分叫口肠迁移。"),
]) + """
<p>函数方法已有<a href="https://link.springer.com/article/10.1186/s12874-025-02748-2">AEGIS直接先例</a>和<a href="https://arxiv.org/abs/1207.5947">功能型加性混合模型</a>；共享分解可参考<a href="https://www.sciencedirect.com/science/article/pii/S0047259X1730204X">AJIVE</a>。两条替代路线不要求同时加入主分析。</p>
<p>停止条件应围绕问题：个人内饮食跨度不足时停止个人反应曲线叙事；只提升水平预测时收窄为代谢状态标记；单部位足够时报告不对称结果；复杂模型无稳定收益时采用简单模型。</p>
<div class="note">现有人日3305人／29008人日及营养完整1878人／13667人日均不是四模态交集。餐次菌交互尚未获得支持。上述资源事实影响实施顺序，但不是本报告的方法优势结论。</div>
""")

section("astra", "Astra方法建议全文与核查记录", '<div class="astra-original" id="astra-source">' + escape(astra) + '</div><div class="file-links"><a href="Astra_method_architecture_2026-10-14.txt">Astra全文TXT</a><a href="method_comparison_sources_2026-10-14.json">方法前作来源JSON</a><a href="HPP_研究优势与科学推进_Astra_2026-10-14.html">上一轮资源与科学推进报告</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">已有结果报告</a></div><p>所有建议为待执行设计，未新增统计结果。文献核查为定向方法比较，部分页面直接访问受限，原始主文可检索段落与核查深度逐条留存。不是方法首次性证明。</p><p>Astra原文 SHA256</p><pre>' + hashlib.sha256(ASTRA.read_bytes()).hexdigest() + '</pre>', "gpt-6-astra · 原文完整保留")

nav = ''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>' for i, (key, title, _) in enumerate(sections, 1))
head = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 方法优势与分析架构 · 2026-10-14</title><style>'''
body = '''</style></head><body id="top"><header class="topbar"><div class="brand">HPP · 方法优势与分析架构<small>DIET CONTRASTS / RESPONSE STRUCTURE / PAIRED MICROBIOMES</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header><div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 方法、估计对象与证据</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>'''
hero = '''</nav></details><div class="sidebar-footer"><b>2026年10月14日</b>Astra方法架构建议<br>强前作与强基线比较<br>仅用现有数据，未拟合新模型</div></aside><main><div class="hero"><div class="eyebrow">HPP / METHODS THAT SERVE THE QUESTION</div><h1>用方法发挥数据优势<br>从关联列表到饮食反应结构</h1><p>明确估计的量，以重复观测共同估计响应，并检验双部位信息能否帮助理解完全留出参与者的饮食相关差异。</p><div class="hero-meta"><span>一阶段层级估计</span><span>响应导向的低秩曲面</span><span>变化预测与校准</span><span>Astra全文</span></div></div>'''
tail = '''<footer class="footer">方法设计报告 · 新结论示例均为假设 · 2026-10-14</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a><script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function layout(e){d.open=!e.matches;}layout(mq);mq.addEventListener('change',layout);})();</script></body></html>'''
OUT.write_text(head + css + body + nav + hero + ''.join(x[2] for x in sections) + tail, encoding="utf-8")
print(OUT)
print(f"sections={len(sections)} bytes={OUT.stat().st_size}")
