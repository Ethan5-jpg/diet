"""Build a local decision report; preserve both Astra source texts verbatim."""
from pathlib import Path
from html import escape
import hashlib
import re

BASE = Path(__file__).resolve().parent
FULL = "HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html"
PLAN = BASE / "Astra_supplementary_experiments_2026-10-13.txt"
ADDENDUM = BASE / "Astra_axis_design_addendum_2026-10-13.txt"
OUTPUT = BASE / "HPP_补充实验决策_Astra_2026-10-13.html"
plan = PLAN.read_text(encoding="utf-8")
addendum = ADDENDUM.read_text(encoding="utf-8")
css = re.search(r"<style>(.*?)</style>", (BASE / FULL).read_text(encoding="utf-8"), re.S).group(1)
css += """
.sidebar summary{display:none}
.astra-original{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;line-height:1.95}
.stages{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}
.stage{background:#f5f8f1;border:1px solid var(--line);border-radius:10px;padding:20px;min-width:0}
.stage b{display:block;color:var(--green);font-size:18px;margin-bottom:12px}.stage p{font-size:14px;margin:0}
.decision{border:1px solid var(--line);padding:18px 20px;border-radius:8px;margin:18px 0}.decision h3{margin-top:0}
.tagline{font-size:13px;color:var(--green);font-weight:650}
@media(max-width:760px){.stages{grid-template-columns:1fr}.sidebar summary{display:list-item;cursor:pointer;font-weight:650;color:var(--green)}.sidebar-title{display:none}.sidebar nav{padding-top:12px}.astra-original{font-size:13px}}
@media print{.sidebar{display:none!important}.astra-original{font-size:10pt}.stages{break-inside:avoid}}
"""
sections = []

def section(key, title, body, label="综合决策 · 待实施建议"):
    number = len(sections) + 1
    sections.append((key, title, f'<section class="report-section" id="{key}"><div class="section-top"><span class="section-number">{number:02d}</span><span class="source-tag">{escape(label)}</span></div><h2>{escape(title)}</h2>{body}</section>'))

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+escape(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

section("choice", "明确首选：新样本前瞻验证，随后决定机制投入", """
<p class="lead">基于你已确认可以补采<strong>粪便、血液和CGM</strong>，Astra的首选是：<strong>补上观察前菌群、后续完整饮食／CGM，以及有针对性的重复测量</strong>。最有价值的新增证据，是时间顺序、离开发现过程的复现和可靠测量。</p>
<p>当前重点候选为<strong>T096（现有数据标注 Ruthenibacterium lactatiformans）</strong>，主检验为“观察前菌群是否标记同一个人NOVA4摄入变化与日平均血糖的关联差异”。T068（现有数据标注 Eubacterium rectale）的个人间血糖波动交互作为预先限定的次要候选。</p>
<div class="note accent"><strong>你的原目标仍是饮食—菌群—心血管／代谢表型。</strong> 基线菌群作为反应标记，与菌群变化作为潜在中间环节，是两种研究问题。下文分别给出采样顺序。当前已核实结果主要为CGM；要把心血管作为论文中心，必须预设并实测一个真实心血管终点。</div>
<p class="tagline">推荐现在筹备：前瞻补采＋核心测量复核。配餐试验、培养、动物和FMT均为条件性后续方案。</p>
""")

section("evidence", "为什么把资源集中到这些问题", table(["现有结果", "对补充实验的影响"], [
    ("T096的个人内NOVA4—日Mean交互，在统一双菌模型及营养完整样本的N0/N1中得到支持。", "优先验证日级平均血糖的反应标记；这是看过发现结果后作出的候选选择，新验证计划须从现在冻结。"),
    ("T096的SD交互在给定Mean后支持较弱；T068个人间SD在营养完整样本中支持有限。", "Mean作为优先宿主结局，SD作为有限次要结局；不据一项显著、另一项不显著宣称菌种功能分工。"),
    ("原日级结果中，粪便采样大多晚于饮食／CGM记录。", "新增研究把菌群采样放到观察前；若研究中间环节，再加早期变化样本及其后的宿主窗口。"),
    ("原逐餐NOVA4×T096／T068交互的q约0.91／0.78，未得到支持。", "重复标准餐优先用于测量校准和反应可靠性；不把单餐小试验当当前主要菌群交互验证。"),
    ("T096呈Pre负、Net正、Abs负；相同窗口和调整时Abs=Pre+Net。", "需要基线测量误差的判别；不能直接解释为菌有两种相反生理作用。数学关系本身也不证明全部为误差。"),
]) + f'<p>以上来自<a href="{FULL}#cross">统一尺度结果</a>、<a href="{FULL}#daily">日级结果</a>、<a href="{FULL}#meals">逐餐结果</a>和<a href="{FULL}#meal-taxa">Pre／Net／Abs分析</a>。本轮没有运行新模型。</p>', "依据 · 已有发现")

section("minimum", "最小补采套餐：钱先花在这里", table(["模块", "采什么／怎样安排", "解决什么问题", "优先级与限制"], [
    ("基线粪便", "至少一份早于后续饮食／CGM窗口的样本；固定物种定量与组成处理流程。记录采便日期、性状、近期排便、处理和保存。", "确认菌群在反应观察之前；建立可复现的候选测量。", "必需。若预算能再采一份，标记可靠性问题优先安排另一独立日期的观察前样本。"),
    ("多日CGM＋结构化饮食", "统一或可校准设备；真实佩戴时间与读数定义有效日。完整记录食物、实食量、进食时间、NOVA及营养；记录重要用药、活动和生活状态。", "验证个人内NOVA4变化与日Mean的关系及T096交互。", "必需。新参与者优先；原参与者新访视应另称时间外重复。观察长度由可靠性和暴露变化决定。"),
    ("针对性血液", "按方案记录生理状态和采血时间的实验室血浆葡萄糖；可加基线HbA1c作为长期背景。少量预算可集中到与CGM同步的参照测量子研究。", "判断血糖测量和长期背景，减少Pre／Net解释的不确定性。", "推荐但按资源设计。一次空腹血糖不能校准全部自由生活餐前读数；不默认全套代谢组。"),
    ("重复标准餐校准", "若平台允许，同人重复相同组成与时刻的标准餐；统一前餐、禁食与运动条件，采多时点参考血糖及CGM曲线。", "估计餐反应可靠性，判别共享基线和测量误差的影响。", "可选嵌套模块。需考虑血浆／组织间液时滞；当前不优先小样本高低菌组单餐交互。"),
    ("早期／期末粪便", "目标为菌群变化时安排早期重复采样，宿主主结局放到它之后；期末样本可补持续变化。", "观察饮食变化后菌群变化及其与后续宿主指标的衔接。", "轴问题的关键新增模块；不能把受饮食影响后的样本当观察前调节变量。"),
]) + """
<p>HbA1c反映此前约3个月的血糖背景，适合基线描述；短观察窗反应不宜以其为主要终点。来源：<a href="https://www.niddk.nih.gov/health-information/diagnostic-tests/A1C-test">NIDDK官方说明</a>。</p>
<p>重复粪便同时记录粪便性状、处理时间和近期排便，是成本较低但有价值的设计。肠道环境可伴随个体内菌群和代谢物变化，不能一概归为技术噪声。依据：<a href="https://www.nature.com/articles/s41564-024-01856-x">Nature Microbiology 2024原始研究</a>。</p>
<div class="note"><strong>预算紧时的顺序：</strong>观察前菌群＋后续饮食／CGM完整性＋参与者信息 → 重复基线菌群／血糖参照测量 → 为明确变化或功能假说采早期／期末样本。所需人数不能用餐次数代替。</div>
""")

section("sequence", "按原始“轴”目标，采样时间怎样排", """
<div class="decision"><h3>问题A：菌群能否标记谁对饮食更敏感？</h3><p><strong>观察前菌群 → 后续饮食与CGM。</strong> 主检验T096基线×同一个人的NOVA4日间偏离与日Mean。它不要求菌群发生改变，最贴近当前日级结果，适合作为首个冻结验证问题。</p></div>
<div class="decision"><h3>问题B：饮食改变菌群，菌群变化是否衔接后续表型？</h3><p>安排下列时间顺序。箭头仅表示采样／观察先后，未表示已经证实的因果关系。</p>
<div class="stages">
<div class="stage"><b>① 基线</b><p>饮食、粪便、CGM背景及选定血液／心血管指标。记录药物与生活状态。</p></div>
<div class="stage"><b>② 饮食变化＋早期粪便</b><p>记录或随机分配明确的饮食方案；较早重复采粪便，测相对基线的菌群变化，按假说采功能指标。</p></div>
<div class="stage"><b>③ 其后的宿主窗口</b><p>主要CGM／选定心血管表型窗口放在早期粪便之后。继续记录饮食、药物与生活状态。</p></div>
</div><p>CGM可连续记录，但主要后续结局窗口要预先定义。窗口间隔按菌群、血糖或心血管指标的反应速度及先导资料确定；无法直接套用固定天数。</p></div>
<p><strong>自由生活纵向观察</strong>可检验先后与变化关联；共同发生的体重、活动、药物变化仍可能混杂。<strong>随机饮食研究</strong>能更有力地判断所分配方案是否改变菌群和宿主指标；菌群变化未被随机化，因此“先变菌、后变血糖”仍不能单独证明因果中介。</p>
<p>如果独立新参与者招募受限，原参与者新时段可回答时间稳定性；应与新参与者验证分开报告。</p>
""", "Astra补充判断 · 两种问题的时间顺序")

section("upgrade", "进一步投入：饮食干预与靶向功能", table(["实验", "什么时候值得做", "关键设计与能支持的结论"], [
    ("多日随机控制饮食", "前瞻候选有足够精度，且有配餐／依从性平台及交互功效支持。", "优先考虑随机顺序交叉的明确高／低NOVA4配餐，周期Mean为主；固定供能、核查实食量，匹配关键营养与时间。设计要评估残留；持续菌群改变可能更适合平行试验。"),
    ("目标菌／功能的实测", "先确认宏基因组、样本库存及目标菌株身份、覆盖和可归属功能。", "可增加目标菌绝对定量／总菌量；功能按菌细胞或标记量归一化。DNA表示功能潜力；测代谢物浓度不能直接当生成通量。"),
    ("靶向离体发酵", "确认了实际菌株、底物与产物预测，且有厌氧培养平台。", "设无底物、无活菌及适当功能对照，控制pH／接种量，测底物消耗与产物生成。供体为生物学重复；完整菌群培养的净反应不能直接归于单菌。"),
    ("动物／FMT／单菌操作", "人体候选和功能链已明确，仍有必须通过因果操作回答的问题。", "当前投入顺序靠后。FMT回答整个菌群的可传递效应；单菌归因需对应添加／缺失／救援、定植及真实层级重复。"),
]) + """
<p>随机配餐须为<strong>饮食方案×基线菌群的交互</strong>规划功效。只能检验平均饮食效果的小试验，应将菌群分层列探索性。已发表饮食随机研究可作为实施先例，不能借用其人数或周期作为本项目的功效依据。</p>
<p>NOVA4是食物分类；两种配餐即使匹配主要营养，也可能改变形态、能量密度或其他因素。结论应指向实际分配的方案。提供菜单匹配不能替代实食量核查，见<a href="https://pubmed.ncbi.nlm.nih.gov/31105044/">Hall等2019随机喂养原始研究</a>。</p>
<p>功能投入先围绕一个可检验预测。当前没有依据默认同时购买TMAO、全套短链脂肪酸、胆汁酸和炎症面板。Astra在完整原文中提出的乳酸／底物利用路线只是候选培养线索，必须核实本队列菌株。</p>
""")

section("cardio", "心血管分支怎样保留", """
<p>若论文题目仍以心血管为中心，应在补采之前选定一个真实终点，并核实平台和可分析信息量。现在已经核实的主要证据来自CGM，不能把血糖关联直接称为心血管疾病获益。</p>
""" + table(["可考虑方向", "采用条件", "结论范围"], [
    ("已有真实临床事件", "查事件定义／审定、日期、既往病史、随访、药物、事件数以及饮食菌群时间。", "时间及信息量支持时开展事件研究；横断面表型不能称新发事件预测。"),
    ("新采血压表型", "明确饮食—血压问题，标准化重复或动态测量；处理钠、用药、测量时刻和合理反应窗口。", "血压是心血管表型／危险因素，不是疾病事件。"),
    ("血管功能／动脉僵硬", "已有可靠测量合作，假说和响应时间与指标匹配。", "预设一个主要表型及其质控；不同测量不能随意替换。"),
    ("血脂／ApoB", "中心问题确实涉及脂质代谢和相应饮食改变。", "按具体指标解释；不因一项改善泛称心血管获益。"),
]) + "<p>短期补充试验不宜追求心血管事件或结构性斑块逆转。可先核实原HPP的终点库存，再决定新采哪一项；目前尚未证明这些字段／平台可用。</p>")

section("decision-rules", "设计前要冻结什么，什么结果意味着停止", table(["必须提前确定", "实际要求"], [
    ("主问题／主终点", "候选T096、主暴露NOVA4、主宿主结局日Mean、个人内估计目标；T068及SD列有限次要假设。原始探索历史应如实保留。"),
    ("功效与精度", "基于参与者数、日间暴露变化、聚类、菌群可靠性、缺失及保守收缩后的交互模拟；同时考虑有实际意义的最小幅度。不给尚无依据的固定人数和周期。"),
    ("验证判据", "比较预设方向、量级和置信区间。精确排除有意义效应或明确相反方向时停止候选升级；宽区间保持未定。"),
    ("质量与测量", "核实实际CGM观察期、Mean／日SD及餐前／后窗口。针对Pre误差模拟或参考测量，避免按显著性选择窗口。"),
    ("新增证据边界", "已有N0/N1、Mean条件、日期分离和三模型敏感性已经完成；不重新包装为独立验证。培养、配餐、动物资源尚未确认。"),
]) + """
<p>Pre／Net的新增判别可先利用现有读数，但需确认时间元数据。互不重叠的餐前读数可减少直接复用，同一CGM的近时误差仍可能相关；需要独立参照或有边界的误差情景。一般方法依据见<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3854983/">Chiolero等2013原始方法学模拟</a>，它不代表已经证实本项目存在偏倚。</p>
<p>新增验证即使阴性仍有价值：它决定是否保留候选和解释。不要临时换下一株显著菌掩盖验证结果。</p>
""")

section("astra-plan", "Astra完整原文：补充实验的具体决策", '<p>以下保留顾问文字；原文中的 #锚点指原完整结果报告。上文为主线程整理，便于直接比较和落实。</p><div class="astra-original" id="astra-original-plan">'+escape(plan)+'</div>', "gpt-6-astra · 完整原文")
section("astra-addendum", "Astra补充原文：标记问题与潜在中间环节", '<div class="astra-original" id="astra-original-addendum">'+escape(addendum)+'</div>', "gpt-6-astra · 补充判断原文")

section("sources", "来源、文件与建议边界", """
<p>建议日期2026-10-13。用户已确认可以补采粪便、血液或CGM；其他平台尚未确认。本次依据已冻结结果，按用户要求由gpt-6-astra给出判断。本报告为实验决策建议，没有运行新服务器模型或开展人体／动物实验。</p>
<div class="file-links"><a href="Astra_supplementary_experiments_2026-10-13.txt">Astra补充实验原文TXT</a><a href="Astra_axis_design_addendum_2026-10-13.txt">采样顺序补充原文TXT</a><a href="Astra_review_packet_2026-10-13.json">固定事实包</a><a href="HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html">完整结果报告</a><a href="HPP_Nature子刊研究提升建议_Astra_2026-10-13.html">上一份研究提升建议</a></div>
<h3>针对性方法／设计依据</h3>
<ul>
<li><a href="https://pubmed.ncbi.nlm.nih.gov/31105044/">Hall等，Cell Metabolism 2019：随机喂养研究</a>。规范全文地址为<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7946062/">PMC7946062</a>；原顾问TXT里的地址缺少PMC前缀，本栏提供规范链接，原文字句未修改。</li>
<li><a href="https://www.nature.com/articles/s41564-024-01856-x">Nature Microbiology 2024：肠道环境、菌群和代谢变化</a>，支持采样元数据与重复测量的考虑。</li>
<li><a href="https://www.nature.com/articles/nature24460">Nature 2017：定量菌群与总菌量</a>，支持条件性绝对定量设计。</li>
<li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3854983/">Chiolero等2013：基线与变化的测量误差方法研究</a>，用于解释设计边界。</li>
<li><a href="https://www.niddk.nih.gov/health-information/diagnostic-tests/A1C-test">NIDDK：HbA1c的时间含义</a>。</li>
<li><a href="https://www.microbiologyresearch.org/content/journal/ijsem/10.1099/ijsem.0.001143">T096对应菌种原始研究</a>与<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4437465/">T068对应菌株原始实验</a>仅提供功能假说线索，不能代替本队列菌株／体内功能核实。</li>
</ul>
<p>目标是提高结论的可信度与解释力，不依据实验数量预测期刊录用。前7节为综合执行建议，后两节分别保存Astra原方案与后续补充判断。</p>
""" + '<p>源文件SHA256</p><pre>补充实验原文：'+hashlib.sha256(PLAN.read_bytes()).hexdigest()+'\n采样顺序补充：'+hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()+'</pre>', "来源记录")

nav = ''.join(f'<a href="#{key}"><span>{i:02d}</span>{escape(title)}</a>' for i, (key, title, _) in enumerate(sections, 1))
page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPP · 补充实验决策 · Astra · 2026-10-13</title><style>""" + css + """</style></head><body id="top">
<header class="topbar"><div class="brand">HPP · 补充实验决策<small>ASTRA / NEXT EXPERIMENTS</small></div><button class="print-btn" type="button" onclick="window.print()" style="margin-left:auto">打印 / 保存PDF</button></header>
<div class="layout"><aside class="sidebar"><p class="sidebar-title">CONTENTS / 可执行套餐与完整原文</p><details id="tocDetails" open><summary>报告目录 · 点击展开</summary><nav>""" + nav + """</nav></details><div class="sidebar-footer"><b>2026年10月13日</b>按已确认的新样本条件收敛<br>实验尚未实施<br>原始心血管目标保留</div></aside>
<main><div class="hero"><div class="eyebrow">HPP / SUPPLEMENTARY EXPERIMENT DECISIONS</div><h1>下一步补什么实验<br>让证据更有说服力</h1><p>基于已有结果和可补采粪便、血液、CGM的条件，给出最小套餐、升级门槛，以及贴近饮食—菌群—宿主轴的采样顺序。</p><div class="hero-meta"><span>Astra具体决策</span><span>最小补采套餐</span><span>完整原文保留</span><span>2026-10-13</span></div></div>
""" + ''.join(body for _, _, body in sections) + """<footer class="footer">补充实验决策 · 现有发现／拟议方案分别标注 · 2026-10-13</footer></main></div><a class="back-top" href="#top">返回顶部 ↑</a>
<script>(function(){var d=document.getElementById('tocDetails');var mq=window.matchMedia('(max-width:760px)');function setLayout(e){d.open=!e.matches;}setLayout(mq);mq.addEventListener('change',setLayout);})();</script></body></html>"""
assert len({key for key, _, _ in sections}) == len(sections)
OUTPUT.write_text(page, encoding="utf-8")
print(OUTPUT)
print(f"sections={len(sections)} bytes={OUTPUT.stat().st_size}")
