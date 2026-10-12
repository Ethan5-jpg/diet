# 扩展敏感性上传位置核对

2026 年 10 月 10 日 18:57:47，用户回传服务器编辑器目录截图。可见的新文件夹层级与补充运行要求一致：`Data/diet_gut_cgm_interaction/extension_sensitivity/`，且原 `outputs/` 与它同级。本次是部署记录，尚无补充拟合结果。

## 截图可见内容

```text
DATA/
├── cgm_deal/
├── co-variant/
├── diet_deal/
└── diet_gut_cgm_interaction/
    ├── config/
    ├── docs/
    ├── extension_sensitivity/
    │   ├── config/
    │   ├── interaction/
    │   ├── EXTENSION_METHODS.md
    │   ├── METHODS.md
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── run_extension_sensitivity.py
    │   ├── run_extension_sensitivity.sh
    │   └── show_results.py
    ├── interaction/
    ├── outputs/
    ├── tests/
    ├── METHODS.md
    ├── README.md
    ├── requirements.txt
    └── run_analysis.py （位于截图下缘）
```

折叠目录内的文件、旧 outputs 下的具体运行名称及文件内容没有展开，不能仅凭本图核验上传字节一致性或旧结果完整性。截图没有版本输出、样本数、系数、区间、p／q 或完成状态。程序运行时会核验原核心代码、旧结果和输入文件指纹。

## 对应运行指令

服务器绝对路径、环境名及旧运行名沿用此前终端截图记录；本图确认的是相对层级。

```bash
conda activate pheno && \
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/extension_sensitivity && \
PYTHON=python bash run_extension_sensitivity.sh \
  --reference-run ../outputs/run_20261010T085601_096554Z_4d43e0
```

正常结束后回传 EXTENSION 1/3、2/3、3/3 三屏；若中途报错，回传最后一屏报错。真实补充结果仍待服务器运行回传。

原图已从用户提供的桌面路径复制保存为 [18:57:47 部署截图](截屏2026-10-10%2018.57.47.png)，复制前后 SHA256 一致。来源与本记录的文件指纹见 [归档清单](归档清单.json)。
