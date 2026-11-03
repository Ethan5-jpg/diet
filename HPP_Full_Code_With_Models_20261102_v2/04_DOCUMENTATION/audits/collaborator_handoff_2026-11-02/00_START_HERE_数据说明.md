# 合作者从这里开始：已处理数据可直接读入

此目录仅提供数据与预处理，不附预测模型、预测结果、预测值或已学菌群评分。

项目根下的 `collaborator_handoff_20261102_v1/` 是你们的独立工作目录。背景、清洗规则、单位和角色见 `README_先读.md`，字段见 `DATA_DICTIONARY.md`。

## 已预处理好的人级数据

`inputs/analysis_ready/full/{结局}/{暴露}_table_TRE_ONLY.parquet`

九结局：D_Glu、D_TotalFat、D_VAT、D_Liver_elasticity、D_Liver_velocity、D_Liver_viscosity、D_Liver_attenuation、D_Liver_dispersion、D_Liver_speed_of_sound。

两暴露：fiber_density、UPF；共18张全发现组可用表，已按各组合准入。表中含原协变量、饮食、Y_original、Y_processed、异常值/缩尾标记、Y_unit、diet_u/diet_h、input_row、家庭和折号。人数见 `manifests/analysis_ready_counts.csv`。

这些full表适合描述/全发现拟合。计算五折验证时，必须使用对应折的处理值和菌群转换，不能用full表阈值或全发现CLR代替折内转换。

## 已预处理好的菌群与折内结局

```text
preprocessing/processed/fold0/ (fold1–fold4与full同结构)
  oral_CLR_PC_TRE_ONLY.parquet
  gut_CLR_PC_TRE_ONLY.parquet
  oral_column_dictionary.json
  gut_column_dictionary.json
  D_Glu_processed_Y_TRE_ONLY.parquet
  ...其余八结局同名规则
  phenotype_transforms.json
```

菌群Parquet共4427行，含input_row、supported、CLR_0001…和PC01…；物种对应关系见column_dictionary。CLR值来自原有训练折缓存，未重新选择转换。unsupported行保存为缺失，不把原缓存的运算零填充视为有效菌群。PC和CLR均已有训练标准化。

结局Parquet也按4427行主索引对齐，含原Y、处理Y及标记、is_training_role、eligible_fiber_density、eligible_UPF。训练用is_training_role与对应eligible相交，验证用非训练角色与eligible相交。表型阈值仅由对应训练角色拟合，保存于phenotype_transforms.json。

也可使用 `examples/load_fiber_fold.py` 读取原NPZ缓存和折内Y；它不包含预测模型。

## 读取例子

在本目录启动pheno Python/Jupyter后：

```python
from pathlib import Path
import pandas as pd

root = Path.cwd()  # 确保cwd为本交接目录
t = pd.read_parquet(root / 'inputs/analysis_ready/full/D_Glu/fiber_density_table_TRE_ONLY.parquet')
oral = pd.read_parquet(root / 'preprocessing/processed/full/oral_CLR_PC_TRE_ONLY.parquet')
joined = t.merge(oral, on='input_row', validate='one_to_one')
# full仅用于全发现拟合。五折分析改读fold0–fold4的菌群/结局，并使用相应掩码。
```

新代码/结果放在 `experiments/你们的实验名/`。完整性检查：

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python examples/check_complete_package.py
```

## 日级与原输入

`inputs/daily/common_days_no_scores_TRE_ONLY.parquet` 是已经对齐、每人≥3共同有效日、带每人等总权重的日级数据（3028人/2782家庭/21279日）；无评分、预测或评分交互。

同目录另有发现组来源food/CGM日表，可自行重建准入。`inputs/discovery_v2/`保留整理后的原相对丰度、原Y和协变量，方便另选预处理。仅发现组有结局，确认角色表不含确认Y。

所有数据均在TRE内使用。当前70/30及家庭单层五折继续保持；新模型的选择限发现组。现有预处理是可复用起点，改变定义时请在自己的目录记录规则和来源。
