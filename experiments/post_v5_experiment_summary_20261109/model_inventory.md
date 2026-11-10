# V6–V14 模型流程逐项清单

八批预测实验共 **428 条批内流程计数**：V6–V11、V13、V14。历史参考不计入，没有对跨批重复算法去重。
其中 **111 条**有截图直接展示的完整双轮绝对 R²，另 **317 条**绝对分数未展示；相对诊断不能补填这些绝对分数。
V12 的四项主要生态指标、12 条匹配交互路线和三路线功能补充单列在[修饰分析清单](modification_inventory.md)，不混入本计数。

## V6：diet_model_batch_20261107_v6

新流程 11 条；其中 11 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| catboost_oral | catboost | oral / existing/original | 0.110465 | 0.111019 | 未单独展示 |
| catboost_oral_gut | catboost | oral_gut / existing/original | 0.107248 | 0.109009 | 未单独展示 |
| fusion | fusion | see method summary / existing/original | 0.114660 | 0.115187 | 未单独展示 |
| hgb_oral | hgb | oral / existing/original | 0.108898 | 0.110299 | 未单独展示 |
| hgb_oral_gut | hgb | oral_gut / existing/original | 0.103158 | 0.109644 | 未单独展示 |
| kernel_linear | kernel | see method summary / existing/original | 0.093084 | 0.091701 | 未单独展示 |
| kernel_main | kernel | see method summary / existing/original | 0.092169 | 0.092831 | 未单独展示 |
| kernel_spline | kernel | see method summary / existing/original | 0.092350 | 0.092681 | 未单独展示 |
| kernel_supervised | kernel | see method summary / existing/original | 0.092169 | 0.092831 | 未单独展示 |
| xgboost_oral | xgboost | oral / existing/original | 0.091460 | 0.089819 | 未单独展示 |
| xgboost_oral_gut | xgboost | oral_gut / existing/original | 0.103741 | 0.103712 | 未单独展示 |

[本轮目录](../diet_model_batch_20261107_v6/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V7：diet_model_expansion_20261107_v7

新流程 30 条；其中 30 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| lgbm_host | lgbm | host / existing/original | 0.022132 | 0.024691 | - |
| lgbm_oral | lgbm | oral / existing/original | 0.093339 | 0.099646 | - |
| lgbm_gut | lgbm | gut / existing/original | 0.019226 | 0.027698 | - |
| lgbm_balanced | lgbm | balanced / existing/original | 0.088615 | 0.100563 | - |
| ebm_oral | ebm | oral / existing/original | 0.087057 | 0.092082 | - |
| ebm_balanced | ebm | balanced / existing/original | 0.064024 | 0.070517 | - |
| hgb_hellinger_oral | hgb_hellinger | oral / existing/original | 0.102654 | 0.111609 | - |
| hgb_hellinger_balanced | hgb_hellinger | balanced / existing/original | 0.109145 | 0.117441 | - |
| hgb_presence_oral | hgb_presence | oral / existing/original | 0.109854 | 0.109388 | - |
| hgb_presence_balanced | hgb_presence | balanced / existing/original | 0.108174 | 0.107352 | - |
| gam_oral | gam | oral / existing/original | 0.082399 | 0.100255 | - |
| gam_balanced | gam | balanced / existing/original | 0.070595 | 0.081222 | - |
| tree_embed_oral | tree_embed | oral / existing/original | 0.055773 | 0.074713 | - |
| tree_embed_balanced | tree_embed | balanced / existing/original | 0.044060 | 0.049182 | - |
| pls_svr_oral | pls_svr | oral / existing/original | 0.032212 | 0.034998 | - |
| pls_svr_balanced | pls_svr | balanced / existing/original | 0.009440 | 0.002364 | - |
| factorization_oral | factorization | oral / existing/original | 0.017768 | 0.021657 | W |
| factorization_balanced | factorization | balanced / existing/original | -0.131730 | -0.165287 | W |
| host_residual_oral | host_residual | oral / existing/original | 0.099962 | 0.102485 | - |
| host_residual_balanced | host_residual | balanced / existing/original | 0.105334 | 0.108436 | - |
| nystrom_oral | nystrom | oral / existing/original | 0.076471 | 0.076572 | - |
| nystrom_balanced | nystrom | balanced / existing/original | 0.074263 | 0.074931 | - |
| gut_mixture | mixture | balanced / existing/original | 0.058278 | 0.065225 | - |
| additive_boost | additive | balanced / existing/original | 0.104149 | 0.111701 | - |
| ebm_diet_gut | ebm_interaction | pool / existing/original | 0.064024 | 0.070517 | - |
| additive_pc_interaction | pc_interaction | pool / existing/original | 0.105346 | 0.109333 | - |
| additive_gate_interaction | gate_interaction | pool / existing/original | 0.104068 | 0.111828 | - |
| inner_selector | fusion/selector | pool / existing/original | 0.103860 | 0.108123 | - |
| convex_blend | fusion/selector | pool / existing/original | 0.111562 | 0.117443 | - |
| ridge_stack | fusion/selector | pool / existing/original | 0.112019 | 0.119128 | - |

[本轮目录](../diet_model_expansion_20261107_v7/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V8：diet_model_overnight_20261108_v8

新流程 55 条；其中 18 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| anchor_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_hellinger_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_hellinger_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| nmf_hgb_oral | hgb | oral / existing/original | 0.108301 | 0.112685 | - |
| nmf_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| nmf_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| nmf_cat_balanced | cat | balanced / existing/original | 0.109011 | 0.112568 | - |
| hybrid_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| hybrid_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| hybrid_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| hybrid_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| pls_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| pls_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| pls_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| pls_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| screen_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| screen_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| screen_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| screen_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| joint_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| joint_hgb_balanced | hgb | balanced / existing/original | 0.106938 | 0.110466 | - |
| joint_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| joint_cat_balanced | cat | balanced / existing/original | 0.109047 | 0.111198 | - |
| rules_oral | rules | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| rules_balanced | rules | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| linear_leaf_oral | linear_leaf | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| linear_leaf_balanced | linear_leaf | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| robust_cat_oral | robust_cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| robust_cat_balanced | robust_cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| rotation_oral | rotation | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| rotation_balanced | rotation | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| neighbors_oral | neighbors | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| neighbors_balanced | neighbors | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| fm_safe_oral | fm | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| fm_safe_balanced | fm | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| additive_nmf_oral | additive | oral / existing/original | 0.106448 | 0.110383 | - |
| additive_nmf_balanced | additive | balanced / existing/original | 0.099211 | 0.104012 | 未单独展示 |
| linear_main_oral | ridge | oral / existing/original | 0.082085 | 0.084203 | 未单独展示 |
| linear_main_balanced | ridge | balanced / existing/original | 0.072603 | 0.078421 | 未单独展示 |
| sparse_diet_oral | sparse_interaction | pool / existing/original | 0.082085 | 0.084203 | 未单独展示 |
| sparse_diet_balanced | sparse_interaction | pool / existing/original | 0.072603 | 0.078421 | 未单独展示 |
| nmf_diet_oral | nmf_interaction | pool / existing/original | 0.106448 | 0.110383 | 未单独展示 |
| nmf_diet_balanced | nmf_interaction | pool / existing/original | 0.099211 | 0.104012 | 未单独展示 |
| convex_core | fusion/selector | pool / existing/original | 0.117428 | 0.121169 | - |
| convex_with_fm | fusion/selector | pool / existing/original | 0.117428 | 0.121215 | - |
| ridge_core | fusion/selector | pool / existing/original | 0.119314 | 0.124576 | - |
| convex_wide | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_wide | fusion/selector | pool / existing/original | 0.120396 | 0.122745 | - |
| convex_joint | fusion/selector | pool / existing/original | 0.117238 | 0.121392 | J |
| ridge_joint | fusion/selector | pool / existing/original | 0.120803 | 0.121851 | J |
| inner_selector_core | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| inner_selector_joint | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |

[本轮目录](../diet_model_overnight_20261108_v8/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V9：diet_model_adaptive_20261108_v9

新流程 64 条；其中 17 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| anchor_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| anchor_cat_balanced | cat | balanced / existing/original | 0.105924 | 0.110793 | - |
| hurdle_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| hurdle_hgb_balanced | hgb | balanced / existing/original | 0.107348 | 0.117711 | - |
| hurdle_cat_oral | cat | oral / existing/original | 0.107013 | 0.108575 | - |
| hurdle_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| log1p_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| log1p_cat_balanced | cat | balanced / existing/original | 0.104581 | 0.108512 | - |
| rank_ridge_oral | ridge | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| rank_ridge_balanced | ridge | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| rank_svr_oral | svr | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| rank_svr_balanced | svr | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| alr_enet_oral | enet | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| alr_enet_balanced | enet | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| alr_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| alr_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| balance_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| balance_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| balance_ridge_oral | ridge | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| balance_ridge_balanced | ridge | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| rclr_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| rclr_hgb_balanced | hgb | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| rclr_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| rclr_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| eco_ridge_oral | ridge | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| eco_ridge_balanced | ridge | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| eco_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| eco_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| ecohybrid_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| ecohybrid_cat_balanced | cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| ordered_cat_oral | ordered_cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| ordered_cat_balanced | ordered_cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| leafwise_cat_oral | leafwise_cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| leafwise_cat_balanced | leafwise_cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| tweedie_lgbm_oral | tweedie | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| tweedie_lgbm_balanced | tweedie | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| gauss_target_cat_oral | gauss_cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| gauss_target_cat_balanced | gauss_cat | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| residual_mlp_oral | residual_mlp | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| residual_mlp_balanced | residual_mlp | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| deep_rvfl_oral | deep_rvfl | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| deep_rvfl_balanced | deep_rvfl | balanced / existing/original | 未展示 | 未展示 | 未单独展示 |
| daily_hgb_oral | hgb | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| daily_hgb_balanced | hgb | balanced / existing/original | 0.106250 | 0.108324 | D |
| daily_cat_oral | cat | oral / existing/original | 未展示 | 未展示 | 未单独展示 |
| daily_cat_balanced | cat | balanced / existing/original | 0.108682 | 0.113566 | D |
| balance_diet_oral | weak_interaction | pool / existing/original | 0.053619 | 0.068544 | 未单独展示 |
| balance_diet_balanced | weak_interaction | pool / existing/original | 0.055858 | 0.070736 | 未单独展示 |
| eco_diet_oral | weak_interaction | pool / existing/original | 0.044204 | 0.044963 | 未单独展示 |
| eco_diet_balanced | weak_interaction | pool / existing/original | 0.042994 | 0.046851 | 未单独展示 |
| convex_core | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_core | fusion/selector | pool / existing/original | 0.116258 | 0.129846 | - |
| convex_wide | fusion/selector | pool / existing/original | 0.119491 | 0.126979 | - |
| ridge_wide | fusion/selector | pool / existing/original | 0.118120 | 0.129350 | - |
| convex_daily | fusion/selector | pool / existing/original | 0.118598 | 0.126942 | D |
| ridge_daily | fusion/selector | pool / existing/original | 0.117164 | 0.129255 | D |
| gated_soft | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| gated_medium | fusion/selector | pool / existing/original | 0.112516 | 0.123671 | - |
| gated_strong | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| tree_meta | fusion/selector | pool / existing/original | 0.109037 | 0.124472 | - |
| polynomial_stack | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| inner_selector | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |

[本轮目录](../diet_model_adaptive_20261108_v9/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V10：diet_model_context_20261108_v10

新流程 68 条；其中 16 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| hgb_oral_none | hgb | oral / none | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_none | cat | balanced / none | 未展示 | 未展示 | 未单独展示 |
| block_balanced_none | block | balanced / none | 未展示 | 未展示 | 未单独展示 |
| hgb_host_none | hgb | host / none | 未展示 | 未展示 | 未单独展示 |
| cat_host_none | cat | host / none | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_macro | hgb | oral / macro | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_macro | cat | balanced / macro | 未展示 | 未展示 | 未单独展示 |
| block_balanced_macro | block | balanced / macro | 未展示 | 未展示 | 未单独展示 |
| hgb_host_macro | hgb | host / macro | 未展示 | 未展示 | 未单独展示 |
| cat_host_macro | cat | host / macro | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_sodium | hgb | oral / sodium | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_sodium | cat | balanced / sodium | 未展示 | 未展示 | 未单独展示 |
| block_balanced_sodium | block | balanced / sodium | 未展示 | 未展示 | 未单独展示 |
| hgb_host_sodium | hgb | host / sodium | 未展示 | 未展示 | 未单独展示 |
| cat_host_sodium | cat | host / sodium | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_life | hgb | oral / life | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_life | cat | balanced / life | 未展示 | 未展示 | 未单独展示 |
| block_balanced_life | block | balanced / life | 未展示 | 未展示 | 未单独展示 |
| hgb_host_life | hgb | host / life | 未展示 | 未展示 | 未单独展示 |
| cat_host_life | cat | host / life | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_full | hgb | oral / full | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_full | cat | balanced / full | 0.109957 | 0.116989 | 未单独展示 |
| block_balanced_full | block | balanced / full | 未展示 | 未展示 | 未单独展示 |
| hgb_host_full | hgb | host / full | 未展示 | 未展示 | 未单独展示 |
| cat_host_full | cat | host / full | 未展示 | 未展示 | 未单独展示 |
| hgb_balanced_full | hgb | balanced / full | 未展示 | 未展示 | 未单独展示 |
| cat_oral_full | cat | oral / full | 未展示 | 未展示 | 未单独展示 |
| block_oral_full | block | oral / full | 未展示 | 未展示 | 未单独展示 |
| block_gut_full | block | gut / full | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_macro_life | hgb | oral / macro_life | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_macro_life | cat | balanced / macro_life | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_joint | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_joint | cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_fiber_days | hgb | oral / fiber_days | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_fiber_days | cat | balanced / fiber_days | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_distribution | hgb | oral / distribution | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_distribution | cat | balanced / distribution | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_ratios | hgb | oral / ratios | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_ratios | cat | balanced / ratios | 未展示 | 未展示 | 未单独展示 |
| spline_oral_full | spline | oral / full | 未展示 | 未展示 | 未单独展示 |
| spline_balanced_full | spline | balanced / full | 未展示 | 未展示 | 未单独展示 |
| svr_oral_full | svr | oral / full | 未展示 | 未展示 | 未单独展示 |
| svr_balanced_full | svr | balanced / full | 未展示 | 未展示 | 未单独展示 |
| extra_oral_full | extra | oral / full | 未展示 | 未展示 | 未单独展示 |
| extra_balanced_full | extra | balanced / full | 未展示 | 未展示 | 未单独展示 |
| kernel_oral_full | kernel | oral / full | 未展示 | 未展示 | 未单独展示 |
| kernel_balanced_full | kernel | balanced / full | 0.070024 | 0.075509 | 未单独展示 |
| hurdle_hgb_oral_full | hgb | oral / full | 未展示 | 未展示 | 未单独展示 |
| hurdle_hgb_balanced_full | hgb | balanced / full | 未展示 | 未展示 | 未单独展示 |
| hurdle_cat_balanced_full | cat | balanced / full | 未展示 | 未展示 | 未单独展示 |
| eco_cat_balanced_full | cat | balanced / full | 未展示 | 未展示 | 未单独展示 |
| block_pc_balanced_full | block | balanced / full | 0.077617 | 0.081908 | 未单独展示 |
| fiber_gut_linear | correction | pool / existing/original | 0.078841 | 0.080282 | 未单独展示 |
| fiber_oral_linear | correction | pool / existing/original | 0.075303 | 0.081908 | 未单独展示 |
| fiber_both_rcs | correction | pool / existing/original | 0.077617 | 0.081908 | 未单独展示 |
| fiber_gut_tree | correction | pool / existing/original | 0.109373 | 0.115086 | 未单独展示 |
| fiber_macro | correction | pool / existing/original | 0.077589 | 0.084164 | 未单独展示 |
| fiber_gut_kernel | correction | pool / existing/original | 0.071266 | 0.075590 | 未单独展示 |
| convex_base | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_base | fusion/selector | pool / existing/original | 0.113074 | 0.120920 | - |
| convex_macro | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_macro | fusion/selector | pool / existing/original | 0.119592 | 0.126345 | - |
| convex_full | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_full | fusion/selector | pool / existing/original | 0.119904 | 0.128421 | - |
| convex_diverse | fusion/selector | pool / existing/original | 0.122736 | 0.132459 | - |
| ridge_diverse | fusion/selector | pool / existing/original | 0.126227 | 0.137266 | - |
| convex_wide | fusion/selector | pool / existing/original | 0.122622 | 0.131805 | - |
| ridge_wide | fusion/selector | pool / existing/original | 0.125175 | 0.136837 | - |

[本轮目录](../diet_model_context_20261108_v10/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V11：diet_model_distribution_20261109_v11

新流程 79 条；其中 6 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| hgb_oral_full | hgb | oral / full | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_full | cat | balanced / full | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_joint | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_joint | cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_joint_upf | hgb | oral / joint_upf | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_joint_upf | cat | balanced / joint_upf | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_shrink_global | hgb | oral / shrink_global | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_shrink_global | cat | balanced / shrink_global | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_shrink_local | hgb | oral / shrink_local | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_shrink_local | cat | balanced / shrink_local | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_shrink_uncertainty | hgb | oral / shrink_uncertainty | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_shrink_uncertainty | cat | balanced / shrink_uncertainty | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_quality | hgb | oral / quality | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_quality | cat | balanced / quality | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_covariance | hgb | oral / covariance | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_covariance | cat | balanced / covariance | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_rff | hgb | oral / rff | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_rff | cat | balanced / rff | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_prototype | hgb | oral / prototype | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_prototype | cat | balanced / prototype | 未展示 | 未展示 | 未单独展示 |
| hgb_oral_sliced | hgb | oral / sliced | 未展示 | 未展示 | 未单独展示 |
| cat_balanced_sliced | cat | balanced / sliced | 未展示 | 未展示 | 未单独展示 |
| hgb_host_joint | hgb | host / joint | 未展示 | 未展示 | 未单独展示 |
| cat_host_joint | cat | host / joint | 未展示 | 未展示 | 未单独展示 |
| linear_tree_oral_joint | linear_tree | oral / joint | 未展示 | 未展示 | 未单独展示 |
| linear_tree_balanced_joint | linear_tree | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| dart_oral_joint | dart | oral / joint | 未展示 | 未展示 | 未单独展示 |
| dart_balanced_joint | dart | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| posterior_cat_oral_joint | posterior_cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| posterior_cat_balanced_joint | posterior_cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| huber_cat_oral_joint | huber_cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| huber_cat_balanced_joint | huber_cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| bayes_oral_joint | bayes | oral / joint | 未展示 | 未展示 | 未单独展示 |
| bayes_balanced_joint | bayes | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| ard_oral_joint | ard | oral / joint | 未展示 | 未展示 | 未单独展示 |
| ard_balanced_joint | ard | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| adaptive_enet_oral_joint | adaptive_enet | oral / joint | 未展示 | 未展示 | 未单独展示 |
| adaptive_enet_balanced_joint | adaptive_enet | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| spc_hgb_oral_joint | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| spc_hgb_balanced_joint | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| spc_cat_oral_joint | cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| spc_cat_balanced_joint | cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| quadratic_hgb_oral_joint | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| quadratic_hgb_balanced_joint | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| quadratic_ridge_oral_joint | ridge | oral / joint | 未展示 | 未展示 | 未单独展示 |
| quadratic_ridge_balanced_joint | ridge | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_balanced_joint | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| cat_oral_joint | cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| block_pc_oral_joint | block | oral / joint | 未展示 | 未展示 | 未单独展示 |
| block_pc_balanced_joint | block | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| resid_host_hgb | hgb | host / joint | 未展示 | 未展示 | 未单独展示 |
| resid_oral_hgb | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| resid_balanced_hgb | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| resid_oral_cat | cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| resid_oral_bayes | bayes | oral / joint | 未展示 | 未展示 | 未单独展示 |
| resid_ridge_hgb | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| resid_oralbase_oral | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| resid_oralbase_balanced | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| cfiber_gut | fiber | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| cfiber_oral | fiber | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| cfiber_both | fiber | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| cfiber_macro | fiber | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| convex_oral | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_oral | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| positive_oral | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| convex_balanced | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_balanced | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| positive_balanced | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| convex_new | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_new | fusion/selector | pool / existing/original | 0.126451 | 0.134122 | - |
| positive_new | fusion/selector | pool / existing/original | 0.127859 | 0.134546 | - |
| convex_distribution | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| ridge_distribution | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| positive_distribution | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| convex_combined | fusion/selector | pool / existing/original | 0.127149 | 0.135216 | - |
| ridge_combined | fusion/selector | pool / existing/original | 0.128020 | 0.139652 | - |
| positive_combined | fusion/selector | pool / existing/original | 0.130555 | 0.139855 | - |
| sparse_new | fusion/selector | pool / existing/original | 未展示 | 未展示 | 未单独展示 |
| sparse_combined | fusion/selector | pool / existing/original | 0.128340 | 0.140103 | - |

[本轮目录](../diet_model_distribution_20261109_v11/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V13：diet_model_complementary_20261109_v13

新流程 55 条；其中 5 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| hgb_pc_oral | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_pc_balanced | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| cat_pc_oral | cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| cat_pc_balanced | cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| mars_pc_oral | mars | oral / joint | 未展示 | 未展示 | 未单独展示 |
| mars_pc_balanced | mars | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| gp_pc_oral | gp | oral / joint | 未展示 | 未展示 | 未单独展示 |
| gp_pc_balanced | gp | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_sir_oral | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_sir_balanced | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| cat_sir_oral | cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| cat_sir_balanced | cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| mars_sir_oral | mars | oral / joint | 未展示 | 未展示 | 未单独展示 |
| mars_sir_balanced | mars | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| gp_sir_oral | gp | oral / joint | 未展示 | 未展示 | 未单独展示 |
| gp_sir_balanced | gp | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_save_oral | hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| hgb_save_balanced | hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| cat_save_oral | cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| cat_save_balanced | cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| mars_save_oral | mars | oral / joint | 未展示 | 未展示 | 未单独展示 |
| mars_save_balanced | mars | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| gp_save_oral | gp | oral / joint | 未展示 | 未展示 | 未单独展示 |
| gp_save_balanced | gp | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| bin_cat_pc_oral | bin_cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| bin_cat_pc_balanced | bin_cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| bin_hgb_pc_oral | bin_hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| bin_hgb_pc_balanced | bin_hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| quantile_cat_pc_oral | quantile_cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| quantile_cat_pc_balanced | quantile_cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| bin_cat_save_oral | bin_cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| bin_cat_save_balanced | bin_cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| bin_hgb_save_oral | bin_hgb | oral / joint | 未展示 | 未展示 | 未单独展示 |
| bin_hgb_save_balanced | bin_hgb | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| quantile_cat_save_oral | quantile_cat | oral / joint | 未展示 | 未展示 | 未单独展示 |
| quantile_cat_save_balanced | quantile_cat | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| graph_oral | graph | oral / joint | 未展示 | 未展示 | 未单独展示 |
| graph_balanced | graph | balanced / joint | 未展示 | 未展示 | 未单独展示 |
| convex_oral | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_oral | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_oral | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_balanced | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_balanced | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_balanced | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_non_tree | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_non_tree | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_non_tree | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_new | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_new | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_new | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 0.102229 | 0.108749 | - |
| convex_combined | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 0.127244 | 0.134341 | - |
| ridge_combined | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 0.128262 | 0.136995 | - |
| positive_combined | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 0.129425 | 0.139372 | - |
| sparse_new | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| sparse_combined | fusion/selector | prediction pool; see model name/config / joint / original frozen context | 0.123818 | 0.138018 | - |

[本轮目录](../diet_model_complementary_20261109_v13/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。

## V14：diet_model_structural_20261110_v14

新流程 66 条；其中 8 条已收到完整两轮绝对 R²。

| 流程 | 学习器/结构 | 输入/背景 | 第0轮 R² | 第1轮 R² | 截图标记 |
|---|---|---|---:|---:|---|
| hgb_sir_oral | hgb | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_sir_balanced | hgb | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_sir_oral | cat | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_sir_balanced | cat | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_rff_sir_oral | hgb | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_rff_sir_balanced | hgb | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_rff_sir_oral | cat | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_rff_sir_balanced | cat | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_pilot_gradient_oral | hgb | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_pilot_gradient_balanced | hgb | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_pilot_gradient_oral | cat | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_pilot_gradient_balanced | cat | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| local_sir_oral | local | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| local_sir_balanced | local | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| forest_local_sir_oral | forest_local | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| forest_local_sir_balanced | forest_local | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ntk_sir_oral | ntk | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ntk_sir_balanced | ntk | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| diffusion_sir_oral | diffusion | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| diffusion_sir_balanced | diffusion | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| svr_sir_oral | svr | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| svr_sir_balanced | svr | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| rbf_ard_sir_oral | rbf_ard | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| rbf_ard_sir_balanced | rbf_ard | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| local_rff_sir_oral | local | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| local_rff_sir_balanced | local | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| forest_local_rff_sir_oral | forest_local | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| forest_local_rff_sir_balanced | forest_local | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ntk_rff_sir_oral | ntk | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ntk_rff_sir_balanced | ntk | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| diffusion_rff_sir_oral | diffusion | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| diffusion_rff_sir_balanced | diffusion | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| svr_rff_sir_oral | svr | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| svr_rff_sir_balanced | svr | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| rbf_ard_rff_sir_oral | rbf_ard | oral / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| rbf_ard_rff_sir_balanced | rbf_ard | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_paired_pc_balanced | hgb | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_paired_pc_balanced | cat | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| hgb_cca_balanced | hgb | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| cat_cca_balanced | cat | balanced / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_oral | convex | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_oral | ridge | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_oral | positive | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_balanced | convex | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_balanced | ridge | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_balanced | positive | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_non_tree | convex | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_non_tree | ridge | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_non_tree | positive | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| convex_new | convex | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ridge_new | ridge | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| positive_new | positive | prediction pool; see model name/config / joint / original frozen context | 0.109193 | 0.109225 | - |
| convex_bank | convex | prediction pool; see model name/config / joint / original frozen context | 0.127244 | 0.134341 | - |
| ridge_bank | ridge | prediction pool; see model name/config / joint / original frozen context | 0.128262 | 0.136995 | - |
| positive_bank | positive | prediction pool; see model name/config / joint / original frozen context | 0.129425 | 0.139372 | - |
| convex_combined | convex | prediction pool; see model name/config / joint / original frozen context | 0.128182 | 0.134184 | - |
| ridge_combined | ridge | prediction pool; see model name/config / joint / original frozen context | 0.128999 | 0.136047 | - |
| positive_combined | positive | prediction pool; see model name/config / joint / original frozen context | 0.131369 | 0.139524 | - |
| sparse_new | sparse | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| sparse_combined | sparse | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| local_stack_bank | local | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ntk_stack_bank | ntk | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| rbf_ard_stack_bank | rbf_ard | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| local_stack_combined | local | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| ntk_stack_combined | ntk | prediction pool; see model name/config / joint / original frozen context | 未展示 | 未展示 | 未单独展示 |
| rbf_ard_stack_combined | rbf_ard | prediction pool; see model name/config / joint / original frozen context | 0.125098 | 0.128473 | - |

[本轮目录](../diet_model_structural_20261110_v14/)；数值来源、表示及配置路径见 [evidence_inventory.json](evidence_inventory.json)。
