# 合作者数据交接：当前工作记录

用户2026-11-02明确要求服务器独立目录、复制数据和预处理，方便合作者自行建模；明确不提供模型。随后明确预处理后数据也必须提供。

目标服务器目录：
`/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1`

运行记录：
`/home/ec2-user/multisystem_runtime_20261021/collaborator_handoff_20261102_v1`

仅经Chrome的5bc05bbe… WorkSpaces操作；未操作14054…其他工作。用户打开本项目窗口后已接续成功。无SSH/API/个体数据下载，无权限变更。

第一阶段实际完成：runner19873、worker19880，actual_exit_code.txt=0。复制48个来源文件，来源字节154757748；基础包核验62个文件156955622字节。纤维×血糖读取检查3343人、五折家庭隔离通过，日级3028人/2782家庭/21279日，权重和通过。48来源哈希未变。仅复制发现组输入、折内CLR/PC/暴露基函数和纯转换器；不复制预测模型/结果/预测/评分。

补充阶段已实际完成：runner23652、worker23659，actual_exit_code_processed.txt=0。18个人级组合可用表、54个折/全发现结局处理表、12个CLR/PC Parquet及列字典均生成，逐表写后读回一致。最终哈希检查172个文件、347376963字节，通过。最终独立核验读取两阶段真实退出码并断言0；全包无预测模型文件、评分/预测列，48来源哈希仍未改变，四个Python文件仅纯转换器、读取和两个完整性检查。远程final_verification.json写入运行目录，终端显示DATA_ONLY_HANDOFF_VERIFIED_EXIT_0。

首读文件已生成：00_START_HERE_数据说明.md；README_先读.md和DATA_DICTIONARY.md均在根目录。experiments/供合作者使用。当前Chrome标签保持打开、终端为空闲提示符。

重要远程入口：
- inputs/analysis_ready/full/{endpoint}/{exposure}_table_TRE_ONLY.parquet：18张全发现已处理表，保留原值/处理值/标记/协变量/饮食/家庭/折号/input_row。
- preprocessing/processed/{fold0..fold4,full}/：54张Y处理表与12张口腔/肠道CLR/PC表；各折列字典与Y阈值JSON。
- preprocessing/folds/{fold0..fold4,full}/：既有NPZ和转换参数缓存。无food_model.json及任何代谢预测模型。
- inputs/daily/：发现组CGM日表、food日表、去掉学习评分/评分交互的共同日表。
- manifests/payload_sha256_final.json：最终清单；examples/check_complete_package.py核验。

full表仅用于全发现描述/拟合；CV必须相应折预处理。此包无确认Y，仅有全池角色元数据。不含模型结果或评分。

本地只存自行编写的通用代码/说明/哈希/聚合证据，不存任何个体数据。
