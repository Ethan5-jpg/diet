# HPP CGM Preprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建立可上传至 HPP 服务器并顺序运行的 01–05 CGM 清洗流水线，生成每名合格参与者一行的7表型 raw/clean/z 最终文件。

**Architecture:** 共享 CSV 保真读取、字段验证和写出工具放在 `scripts/cgm_utils.py`。01–05 每个脚本只完成一个 checkpoint，以前一步固定输出为下一步输入，并分别写数据和 QC 报告。

**Tech Stack:** Python 3.7、pandas、NumPy、unittest、POSIX shell。

---

### Task 1: 共享工具和 baseline selection

**Files:**
- Create: `Data/cgm_deal/scripts/cgm_utils.py`
- Create: `Data/cgm_deal/scripts/01_prepare_baseline_connections.py`
- Create: `Data/cgm_deal/tests/test_cgm_preprocessing.py`

**Steps:**
1. 先写合成三表及重复 connection 测试并确认失败。
2. 实现 ID 文本保真、daily day 派生、完整键验证和重复排序。
3. 验证一人一条 baseline、重复报告和并列时停止。

### Task 2: Connection QC

**Files:**
- Create: `Data/cgm_deal/scripts/02_cgm_connection_qc.py`
- Modify: `Data/cgm_deal/tests/test_cgm_preprocessing.py`

**Steps:**
1. 测试天数、loss、primary completeness 三个独立 flag。
2. 实现 pass/exclusion 文件和人数流报告。
3. 验证 phenotype 水平不参与 exclusion。

### Task 3: Core phenotype extraction

**Files:**
- Create: `Data/cgm_deal/scripts/03_extract_iglu_phenotypes.py`
- Modify: `Data/cgm_deal/tests/test_cgm_preprocessing.py`

**Steps:**
1. 测试7个固定表型及 raw 命名。
2. 测试百分比越界和负值变缺失但 participant 行保留。
3. 实现 legality report 和固定列输出。

### Task 4: Robust cleaning

**Files:**
- Create: `Data/cgm_deal/scripts/04_clean_cgm_phenotypes.py`
- Modify: `Data/cgm_deal/tests/test_cgm_preprocessing.py`

**Steps:**
1. 测试最短95%窗口、确定性 tie、8SD删除和5SD温莎化。
2. 实现逐 phenotype 参数和 participant outlier 报告。
3. 验证 raw 不变、clean 允许因极端值缺失。

### Task 5: Standardization and runner

**Files:**
- Create: `Data/cgm_deal/scripts/05_standardize_cgm_phenotypes.py`
- Create: `Data/cgm_deal/run_cgm_preprocessing.sh`
- Modify: `Data/cgm_deal/tests/test_cgm_preprocessing.py`

**Steps:**
1. 测试每个 clean Z-score 均值约0、样本标准差约1。
2. 实现最终固定字段和 `A10=not_evaluated`。
3. 建立遇错即停的一键入口并打印每一步输出。
4. 运行完整 unittest、Python 3.7 grammar 和禁用内置泛型检查。
