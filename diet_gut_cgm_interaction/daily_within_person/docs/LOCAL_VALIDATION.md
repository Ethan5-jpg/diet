# 本地构建与验证记录

以下全部是代码检查与**合成数据验证**，不包含真实服务器分析结论。

清理说明：2026-10-11按用户要求，旧版`.1/.2`合成目录及对应旧日志已删除，仅保留最新版`fix_empty_cgm_20261011_v3`完整验证输出。下方旧运行路径为历史记录，不代表文件仍存在；清理范围及留存manifest见[清理记录](CLEANUP_2026-10-11.md)。

## 最新修复验证：2026-10-11.3

服务器13:30截图报告`Cannot cast DatetimeArray to dtype float64`，发生在CGM读取阶段。此前31项测试遗漏了“最终连接筛选后空数据块”。本地pandas 2.1.0与2.3.2均复现空datetime Series经map后仍为datetime dtype，转float报同样异常。详见[失败与修复记录](../research_records/2026-10-11_133052_CGM读取失败/README.md)。

- 新增5个参数化用例：空/全无效/混合有效时间转换，以及分块大小1/100时整文件与分块被连接筛选清空。修复前**3 failed, 2 passed**，失败消息与截图相同。
- 修复后完整测试命令`/opt/anaconda3/bin/python -m pytest tests -q`：**36 passed in 18.33s**。
- 本地pheno重跑`--demo --stage all --output-dir ../tests/_artifacts/daily_within_person/fix_empty_cgm_20261011_v3`：退出码0，SYNTHETIC、84/84模型、4/4主要日SD模型完成；状态仍为夹具支持诊断导致的`COMPLETE_WITH_LIMITATIONS`。
- 新旧84行beta、SE、区间、p及两类q在rtol/atol=1e−12下保持一致；不改变质量门槛、最终连接选择或统计设计。
- pheno中的空、全NaT及混合时间均返回float64，保留NaN；所有Python文件通过3.8语法解析，shell语法通过。
- `show_results.py --screen 1`指纹检查通过；实际运行代码指纹与已验证合成运行一致。

合成输出及日志保存在上级`tests/_artifacts/daily_within_person/fix_empty_cgm_20261011_v3/`及相邻同名前缀console.txt；没有把合成结果当作真实发现。真实服务器须更新代码并从新目录重新运行。

## 上传前复查：2026-10-11.2（历史）

用户要求上传前再次检查代码及分析逻辑。修订详情见[复查记录](REVIEW_2026-10-11.md)。本次修复可选日期表空访视中止、分阶段恢复文本标识误读，并补强日期精度、失败检验清理、零交集报告和模型字段缺失审计。

- 最终完整测试：`/opt/anaconda3/bin/python -m pytest tests -q` → **31 passed in 17.03s**。解释器/统计库与下方首次测试环境相同。
- 14个Python文件通过Python 3.8语法解析；`bash -n run_analysis.sh`通过。仍未声称在真实Python 3.8环境执行过。
- 本地pheno全流程：`/opt/anaconda3/envs/pheno/bin/python run_analysis.py --demo --stage all --output-dir ../tests/_artifacts/daily_within_person/review_20261011_v2`，退出码0。
- 该运行版本2026-10-11.2、明确标记SYNTHETIC；80个虚构参与者、960个虚构日，**84/84模型、4/4主要模型完成**。`COMPLETE_WITH_LIMITATIONS`仍是原合成夹具中单人交互列份额触发支持提示。
- 全部84行的beta、SE、上下限、p和两类q，与前版`pheno_final`在rtol/atol=1e−12下相符（缺失位置相同）；正常输入的科学估计没有因边界修复而改变。
- 新增测试核对“一次性运行”与“build后analyze”整张交互表在rtol/atol=1e−12下一致；文本`NA`/前导零、数值精度和真正数值缺失分别保留。
- 零日期交集的合成测试明确得到`NO_ESTIMABLE_PRIMARY_MODELS`、84个保留的不可估计行及3屏报告，未产生p/q。
- `show_results.py --screen 3`核验新运行全部保存结果指纹通过。

最新合成输出和控制台日志位于上级`tests/_artifacts/daily_within_person/review_20261011_v2/`及相邻`review_20261011_v2_console.txt`，不在上传目录中。仅为测试，不能作为真实研究结果。

## 首次构建记录（历史版本2026-10-11.1）

下方保留首次构建的24项测试和原合成验证，最新状态以本页开头36项修复验证为准。

## 已完成

- 建立独立上传文件夹：`Data/diet_gut_cgm_interaction/daily_within_person/`。运行代码不导入旧interaction或meal模块。
- 实现check/build/analyze/all、可重复的新运行目录、数据/代码/结果指纹、状态核验、恢复锁、三屏输出。
- 完成饮食日、原始CGM日、唯一连接、菌群/协变量及取样日期接口；完整实现12个假设×7个模型设定。
- 所有Python文件通过Python 3.8语法解析，shell入口通过`bash -n`。这不等同于在真实Python 3.8及其全部旧依赖中运行；服务器以实际pheno环境为准。

## 测试结果

最终命令：

```bash
/opt/anaconda3/bin/python -m pytest tests -q
```

结果：**24 passed in 11.10s**。测试环境Python 3.13.5、NumPy 2.1.3、pandas 2.3.2、SciPy 1.15.3、pytest 8.3.4、statsmodels 0.14.4。

覆盖有实际失效风险的情形：

1. 普通日及Jerusalem夏令时23/25小时日，日SD/CV手算一致；完整日分母、长缺口和不完整首尾日。
2. CGM重复同值合并、冲突/无效值失效；跨CSV分块的重复食物；不把营养未知当零；NOVA两种分母区分。
3. 无偏移时间不自动视为UTC；缺身份、重复参与者、物种名歧义和多次菌群取样日期不随意推断。
4. 不等天数下的个人固定效应OLS/WLS与显式个人哑变量回归核对，聚类协方差与statsmodels `cov_cluster`一致（断言rtol≤1e−9）；完整协方差线性对比、个人恒定水平平移不变性。
5. 无个人内变化或秩缺陷拒绝估计；按模型完整日重新筛天数；缺纤维时保留全部预定家族与失败行。
6. 新目录完整构建后续跑；已完成目录禁止覆盖/重算；源文件改变拒绝恢复；篡改输出拒绝显示。
7. 整包复制到不同服务器前缀的Data结构后独立运行，并测试不显式提供`--data-root`时定位正确。

## 本地pheno全流程

最终运行命令：

```bash
/opt/anaconda3/envs/pheno/bin/python run_analysis.py --demo --stage all \
  --output-dir ../tests/_artifacts/daily_within_person/pheno_final
```

Python 3.11.15、NumPy 1.26.4、pandas 2.1.0、SciPy 1.16.3。80个虚构参与者、960个虚构日、92,160条虚构CGM；84/84模型估计完成，4/4主要日SD模型估计完成。输出为`SYNTHETIC`，状态`COMPLETE_WITH_LIMITATIONS`，原因是部分模型触发单人交互列信息份额的预设支持提示。

已知合成交互NOVA4×T068→日SD设为−0.3，主模型估计约−0.3045，符合生成目标。此数字只用于实现核对，不是HPP发现。`show_results.py --screen 1`的保存结果指纹核验通过。

这些合成运行及进度日志保存在上级 `tests/_artifacts/daily_within_person/`，不在本次上传目录中。早期`demo_first`为开发中的合成运行，`pheno_final`是首次构建版本的验证；最新验证已更新为本页开头的`review_20261011_v2`。

## 尚需服务器确认

真实输入路径、全量字段内容、逐日完整性、纤维/酒精字段完整性、个人内变化及每模型实际N；日期记录的真实性、未记录食物、当天活动/睡眠等仍不能由代码认证。未进行真实模型运行，未声称真实结果显著或复现旧交互。

上传后按[README](../README.md)的完整指令运行。原始与个人级输出留在服务器，回传三屏摘要；本地截图记录放入本模块`research_records/`。
