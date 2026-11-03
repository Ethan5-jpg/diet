# 数据字段与行对齐

仅数据和预处理，无预测模型/评分。实际schema/行数/类型见manifests/parquet_schemas.json。

## 人级

inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet共4427行发现角色。

- participant_id/cohort/research_stage：复合键，当前participant_id也唯一。
- input_row：0起始主索引，丰度/折缓存矩阵均按此对齐。
- split_unit_id：家庭/同住单位；cv_fold：固定0–4折；final_role：discovery。
- age_calendar_years：年龄年；sex_code：源编码，当前准入0/1，不自行推断标签。
- smoking_status：吸烟分类，缺失__UNKNOWN__。
- n_energy_days/n_fiber_days：对应有效饮食日数；主分析各自要求≥3。
- fiber_density：等日权平均纤维密度，g/1000kcal。
- mean_daily_energy_fiber/mean_daily_energy_UPF：相应窗口能量kcal/day。
- UPF_lower_equalday_fraction/UPF_upper_equalday_fraction：UPF上下界，0–1。

oral_raw_TRE_ONLY.npz/gut_raw_TRE_ONLY.npz数组abundance为相对丰度，列顺序见oral_taxa.json/gut_taxa.json。行对应input_row，不能排序后直接拼接。

D_*_raw_Y_TRE_ONLY.parquet为各结局可用子集，含复合键/input_row/Y_value/Y_unit。按键或input_row放回主表，不按结局表当前行号拼接。D_Glu单位mg/dL，其他以Y_unit为准。raw_Y需相应训练折预处理。

roles_v2_full只有全池角色无确认Y；discovery_roles_folds为发现家庭/折。

## 折转换

preprocessing/folds/fold0–fold4/transformed_discovery_TRE_ONLY.npz全部行向量对齐主索引。

- training是预处理训练角色，模型训练还需结局/饮食/协变量/supported准入。
- supported是双部位转换支持；缓存置零不代表有效样本。
- oral_z/gut_z为训练标准化CLR，各折字典和列数可不同，见oral_transform.json/gut_transform.json。
- oral_PC/gut_PC为训练PCA；参数在对应JSON。
- fiber_u/fiber_h、UPF_u/UPF_h为训练线性/非线性暴露基函数；diet_basis.json为单位、中心、结点。
- full仅全发现参考，不能用于五折验证。

UPF辅助输入：unknown_weights_TRE_ONLY.npz为CSR（data/indices/indptr/shape）；reference_food_presence有input_row/food_id；reference_food_labels和unknown_food_categories保留食品证据。需重建未知FoodID列映射时回原来源核对，不能猜列序；已有UPF折基函数可直接读取。

## 日级

inputs/daily/cgm_days_discovery_TRE_ONLY.parquet：发现组来源CGM日记录，保留来源Y/有效性字段。

inputs/daily/food_days_discovery_TRE_ONLY.parquet：发现组新版primary food日表，含日期、能量、纤维及有效性字段。

inputs/daily/common_days_no_scores_TRE_ONLY.parquet：已对齐且≥3共同有效日的最终子集，21279日/3028人/2782家庭。字段包括复合身份键/date/Y_value/Y_unit（若来源有）/energy_kcal/fiber_g/fiber_density_g_per_1000kcal/energy_eligible/fiber_eligible/split_unit_id/cv_fold/age_calendar_years/sex_code/n_i/person_weight/recording_day_index/weekday。无评分/预测列。

合并按复合键+规范化date一对一；n_i为共同日数、person_weight=1/n_i。不要把个人平均纤维替代每日暴露。来源表范围大于共同日表，未筛来源日不能直接当最终合格日。
