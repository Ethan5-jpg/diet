# Diet–CGM Alignment and Unadjusted Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建立AMED/hPDI与三个Primary CGM表型的baseline对齐数据，并产生6个无协变量OLS模型结果。

**Architecture:** 共享路径、CSV保真读取、键校验和统计工具放在 `scripts/analysis_utils.py`。`00_align_diet_cgm.py` 只负责样本流与一对一连接，`01_unadjusted_diet_cgm_models.py` 只负责逐结局完整案例OLS和FDR；runner顺序调用两步并集中写入 `outputs/`。

**Tech Stack:** Python 3.7、pandas、NumPy、标准库math/unittest、POSIX shell。

---

### Task 1: 对齐测试与共享工具

**Files:**
- Create: `Data/diet_microbiome_glucose_analysis/tests/test_diet_cgm_analysis.py`
- Create: `Data/diet_microbiome_glucose_analysis/scripts/analysis_utils.py`

**Steps:**
1. 写AMED、hPDI、CGM合成表及旧hPDI字段测试。
2. 运行测试并确认因模块缺失失败。
3. 实现ID文本保真、baseline过滤、必需字段及唯一键检查。
4. 再次运行测试。

### Task 2: 样本对齐

**Files:**
- Create: `Data/diet_microbiome_glucose_analysis/scripts/00_align_diet_cgm.py`

**Steps:**
1. 测试AMED/hPDI外连接后与CGM内连接的参与者集合。
2. 实现固定字段选择、one-to-one连接和membership报告。
3. 实现可选gut participant presence统计，不限制主分析表。
4. 验证一人一行和交集人数。

### Task 3: 无调整模型

**Files:**
- Create: `Data/diet_microbiome_glucose_analysis/scripts/01_unadjusted_diet_cgm_models.py`

**Steps:**
1. 写已知斜率/截距数据和结局特异缺失测试。
2. 实现带截距的一元OLS、Student-t推断和95%CI。
3. 实现6个固定模型与BH-FDR。
4. 验证beta、N、P值范围和FDR单调性。

### Task 4: Runner与文档

**Files:**
- Create: `Data/diet_microbiome_glucose_analysis/run_diet_cgm_analysis.sh`
- Create: `Data/diet_microbiome_glucose_analysis/README.md`

**Steps:**
1. 串联00和01，并将data/reports/logs统一放入outputs。
2. 记录hPDI最新CSV生成命令、输入字段、模型含义和输出路径。
3. 运行完整unittest、Python 3.7语法检查、Bash语法检查和合成CSV端到端演练。
