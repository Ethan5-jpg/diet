# CGM preprocessing

本目录是可直接上传到服务器 `HPP/Data/cgm_deal` 的固定 CGM 预处理流程。源 CSV
保持在服务器原位置：

- `HPP/Data/Transfer/cgm/cgm.csv`
- `HPP/Data/Transfer/cgm/iglu.csv`
- `HPP/Data/Transfer/cgm/iglu_daily.csv`

## 运行

进入 HPP 项目目录并使用服务器的 `pheno` 环境：

```bash
cd Data/cgm_deal
bash run_cgm_preprocessing.sh
```

脚本遇到缺字段、重复键、daily coverage 不一致或无法按固定优先级选择 connection 时
会立即停止。成功后最终文件是：

```text
Data/cgm_deal/outputs/data/05_cgm_core_phenotypes.csv
```

所有运行产物统一保存在 `outputs/`：处理数据在 `outputs/data/`，核对表在
`outputs/reports/`，完整终端记录在 `outputs/logs/`。正式流程不会读取 parquet，
因为当前服务器 CSV 已恢复五个身份索引字段。

## 可选路径覆盖

如果服务器目录不同，可以在运行时设置 `CGM_SOURCE_DIR`、`CGM_OUTPUT_DIR`、
`CGM_DATA_DIR`、`CGM_REPORT_DIR`、`CGM_LOG_DIR` 或 `PYTHON_BIN`。默认无需设置。
