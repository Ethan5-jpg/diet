# 扩展饮食 × 菌群交互：六组敏感性补充

本文件夹可单独上传，已包含运行脚本、冻结的原拟合引擎、配置、依赖清单和方法说明。补充版本为 `2026-10-10.ext1`；原引擎保持 `2026-10-10.2`。

## 上传位置

把整个 `extension_sensitivity` 文件夹上传到服务器已有的 `diet_gut_cgm_interaction` 内，形成：

```text
/home/ec2-user/Desktop/HPP/Data/
├── 原有饮食、菌群、CGM 和协变量目录
└── diet_gut_cgm_interaction/
    ├── outputs/
    │   └── run_20261010T085601_096554Z_4d43e0/
    └── extension_sensitivity/                 ← 本次上传的完整文件夹
        ├── run_extension_sensitivity.sh
        ├── run_extension_sensitivity.py
        ├── show_results.py
        ├── interaction/
        ├── config/
        ├── requirements.txt
        ├── README.md
        ├── METHODS.md
        └── EXTENSION_METHODS.md
```

只需要上传这一个新文件夹。旧运行目录保留在上一级 `outputs/`；补充运行读取其中的配置、manifest 和结果表。输入数据仍使用原有 Data 内的文件。

## 完整服务器命令

```bash
conda activate pheno && \
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/extension_sensitivity && \
PYTHON=python bash run_extension_sensitivity.sh \
  --reference-run ../outputs/run_20261010T085601_096554Z_4d43e0
```

结束时打印 `EXTENSION 1/3`、`2/3`、`3/3` 和 `RESULT_DIR`。把三屏结果截图发回即可。新结果保存在本文件夹内的 `outputs_extension_sensitivity/`，每次运行使用一个新目录。

重新查看最近一次成功的真实补充运行：

```bash
python show_results.py --screen 1
python show_results.py --screen 2
python show_results.py --screen 3
```

查看指定运行时使用 `python show_results.py --run-dir /实际结果目录 --screen all`。

## 路径与环境

Data 根目录默认由脚本位置向上两级确定，和登录用户名、HPP 所在绝对位置无关。`--reference-run` 的相对路径相对于终端当前目录，上面的命令已包含正确的目录切换。若改变目录层级，可用 `--data-root /实际/Data` 显式指定输入根目录。

需要 Python 3.8+、NumPy、pandas、SciPy、Matplotlib，和原分析使用同一环境即可。缺包时再运行 `python -m pip install -r requirements.txt`。程序不会自动安装依赖。

真实运行使用旧结果保存的配置；随包 `config/analysis.json` 用于合成演示。输入文件内容必须和旧运行一致，否则中止并记录原因。迁移根目录不影响内容指纹核验。

## 这次补什么

为 AHEI、AMED、rEDIH、EAT13 和碳水供能比的全部 60 个扩展组合补齐六组敏感性，检查原有发现对人群选择、药物调整、非线性和额外交互项是否稳定。包含原未显著结果。

每组连同 hPDI、NOVA4 共拟合 84 项，主分析加六组敏感性合计 588 项；每个设定按固定 84 项 BH 校正。原显著 19 项只作为冻结的追踪集合。七饮食共同完整样本标记为 `common_all7_sample`，其定义与原两饮食共同样本不同。

程序核对旧文件、原核心代码及输入指纹，并检查原主分析和前五组重点敏感性共 204 项能否复现。全部扩展的系数、区间、q、条件斜率、样本流、诊断和图保存在结果目录。这是同队列、看过主结果后的敏感性分析，不能称为独立验证。方法详见 [EXTENSION_METHODS.md](EXTENSION_METHODS.md) 和 [原模型方法](METHODS.md)。

本地合成测试只验证软件。2026-10-10 19:01 已收到真实服务器完成摘要：588/588 项拟合、204/204 项复现，原19项中7项六组均保留。真实结果与19:10已核对更正的历史 q 值记录在[所属模块结果档案](../research_records/2026-10-10_扩展敏感性/19.01_服务器结果/README.md)。软件演示可执行 `PYTHON=python bash run_extension_sensitivity.sh --demo`，始终标记 `SYNTHETIC`，不读取真实数据。
