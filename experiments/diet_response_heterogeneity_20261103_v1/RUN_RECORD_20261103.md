# HPP 口肠菌群饮食异质性：第一轮模型实验记录

日期：2026-11-03（北京时间）。本轮由浏览器直接操作 AppStream 远程终端完成。

## 完成状态

- 纤维密度 × D_Glu；15类候选模型，84种设置/折，家庭五折合计420份拟合产物。
- 3343人、3059家庭；验证人数661、677、668、665、672。
- 程序退出码0，状态completed_and_readback_verified；主运行约135.7秒，不含部署、浏览器操作和后续复核。
- 运行后postrun_audit.json为PASS：源码payload、每个源文件哈希、退出码、产物数量、人群及家庭数均核对通过。
- 在每折复现归档匹配null及联合CLR交互lambda=10的验证预测，误差要求2e-8；折内Y也一致。
- 确认组Y未访问。所有个体数据、预测、拟合参数留在TRE。

## 服务器位置

```text
/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/
  collaborator_handoff_20261102_v1/experiments/
    diet_response_heterogeneity_20261103_v1_attempt02/
      code/
      deployment.json
      launch_pid.txt
      bootstrap.log
      logs/execution.json
      logs/pilot.log
      logs/actual_exit_code.txt
      results/registration.json
      results/summary.json
      results/model_comparison.csv
      results/fits_TRE_ONLY/
      results/held_predictions_and_contrasts_TRE_ONLY.parquet
      results/artifact_manifest_TRE_ONLY.json
      postrun_audit.py
      postrun_audit.json
```

新目录写入，未覆盖既有输入、学长代码或旧结果。先前哈希失败的部署目录diet_response_heterogeneity_20261103_v1保留，尚未启动真实模型；attempt02为本次成功运行。

## 可追溯版本

- 参考仓库：Ethan5-jpg/diet；已读取的main为2f77bf14767d9ae5f9397781c5c5c4b1d5582d52。
- 首次TRE运行时源码尚未发布，运行登记使用UNPUBLISHED_SOURCE_SHA256标识。本目录保留同一份源码；发布后的版本以本目录Git提交历史定位，不改写首次运行登记。
- 服务器部署payload SHA256：7adeeb972e45394909a60308b1494c588b6c89ed12b97e1c7ecfb74aa66380a7。
- 逐文件哈希见源码包source_sha256.json和本包local_deployment_receipt.json。
- postrun_audit.py为运行后新增复核源码，置于实验根目录，不属于上述原始8文件payload。
- 发布方式：GitHub接口曾返回403 Resource not accessible by integration；账号仓库权限与应用连接权限不同。本次改用工作环境的Git通道发布，保留既有源文件和部署哈希，不修改GitHub应用授权设置。具体提交以Git历史及成功push回执为准。

## 环境与运行命令

Python3.11.5、NumPy1.26.4、pandas2.1.0、SciPy1.11.4、pyarrow15.0.0；使用/home/ec2-user/miniconda/envs/pheno/bin/python。

成功启动命令相当于在code目录运行：

```bash
/home/ec2-user/miniconda/envs/pheno/bin/python -u launch.py UNPUBLISHED_SOURCE_SHA256:7adeeb972e45394909a60308b1494c588b6c89ed12b97e1c7ecfb74aa66380a7
```

launch.py先执行合成数值核验，再调用run_pilot.py；完整实际参数、路径和资源限制保存在logs/execution.json。模型读取既有折内接口，独立保存至当前attempt02/results。再次运行应使用新的实验/结果目录；不要覆盖当前目录。参考README中的直接运行方式，source目录的部署布局须保持正确。

合成检查包括独立联合矩阵解、Elastic Net独立约束优化、最大惩罚退化为null、改变验证Y不改变训练参数、预测重构、重复背景列、非线性对比恒等式、NPZ顺序写入及拒绝覆盖。

## 公平比较约定

所有模型均保留相同宿主背景、每部位5个无惩罚PC、完整双部位CLR主效应岭回归（均方损失尺度lambda=1）。只改变交互块的部位、CLR/PC表示、惩罚及有限非线性形式。宿主背景含饮食样条、饮食×年龄/性别。没有把新增主效应的收益当作交互收益。

CLR、PC5、PC10岭交互的lambda网格0.01、0.1、1、10、100；Elastic Net的alpha为0.2/0.8，lambda/lambda_max为1/0.3/0.1/0.03。非线性为线性及限制立方饮食基函数分别与每部位PC5相乘。

各模型家族用五折平均MSE选参；这些五折也用于报告成绩，属于开发估计。背景选择来自已有开发结果，当前不是独立外层嵌套验证。

## 结果

下表除详细说明外采用终端汇总的六位小数；完整精度在TRE summary.json。

|模型|交互部位|R²|对匹配null的ΔR²|选中设置|
|---|---|---:|---:|---|
|无菌群交互|null|0.078272|0|固定背景|
|岭CLR|口腔|0.078130|-0.000142|lambda100|
|岭PC5|口腔|0.078259|-0.000014|lambda100|
|岭PC10|口腔|0.078270|-0.000002|lambda100|
|Elastic Net CLR|口腔|0.078272|0|alpha0.2/fraction1，零交互|
|非线性PC5|口腔|0.078251|-0.000021|lambda100|
|岭CLR|肠道|0.080559|+0.002286|lambda10|
|岭PC5|肠道|0.080693|+0.002421|lambda0.1|
|岭PC10|肠道|0.083675|+0.005403|lambda0.1|
|Elastic Net CLR|肠道|0.078272|0|alpha0.2/fraction1，零交互|
|非线性PC5|肠道|0.080958|+0.002685|lambda0.1|
|岭CLR|口腔＋肠道|0.079404|+0.001132|lambda10|
|岭PC5|口腔＋肠道|0.079004|+0.000732|lambda1|
|岭PC10|口腔＋肠道|0.081510|+0.003238|lambda1|
|Elastic Net CLR|口腔＋肠道|0.078272|0|alpha0.2/fraction1，零交互|
|非线性PC5|口腔＋肠道|0.078534|+0.000262|lambda1|

匹配null：MSE110.4595509293327，RMSE10.509973878622741，R²0.07827223012893003。

### 当前最值得跟进：肠道PC10线性交互

- R²0.08367503706727109，ΔR²0.005402806938341054。
- 家庭聚类配对损失得到的条件ΔR²95%区间约[0.0004155,0.0103901]。
- 以所选同一参数计算，各折相对null的ΔR²均为正（5/5）。这不等于五次独立复制。
- 出折斜率校准系数约0.6992，条件95%区间[0.3405,1.0579]；单位为每5 g/1000 kcal纤维、每1训练SD评分的血糖关联差异。
- 固定10→15 g/1000 kcal饮食对比中，模型给出的中心化肠道修饰部分SD约0.8221 mg/dL。该数是模型预测幅度，不是真实个体干预反应的SD。
- 五折均通过预设饮食对比的边际5–95%训练支持检查；边际支持不代表联合positivity充分。

口腔＋肠道PC10同样5/5折ΔR²为正，但整体增量小于仅肠道PC10。其条件ΔR²区间约[0.0001719,0.0063042]。联合模型中口腔斜率校准约−0.1817，区间[−0.5412,0.1778]；肠道约0.7046，区间[0.3509,1.0582]。

肠道非线性PC5为3/5折ΔR²正；总体增量区间跨零，约[−0.0007521,0.0061227]。本轮没有显示它优于肠道PC10线性交互。三个稀疏家族均选择最大惩罚、零交互，未提供少数菌交互的支持。

## 解释边界与后续

这些结果支持继续追踪低维肠道线性交互，而非宣布已找到确认有效的个体营养反应模型。各条件区间没有覆盖全部调参、选型及跨折共享训练不确定性；没有使用确认组检验。模型比较中的口腔无增量不等于口腔在生物学上完全无作用。

后续优先核对肠道PC10的原物种载荷对齐、各折方向和原单位幅度稳定性，限定候选进行家庭层面的更严格模型选择评估，再按冻结方案使用保留确认组。不要仅根据本轮最高R²扩大搜索或读取确认结果挑模型。

运行中读取尚未结束的日志曾出现FileNotFound；完成后日志/退出码可读。后续实时日志可优先写既有本地runtime路径，再封存到实验目录；不要把一次日志读取失败当作训练失败。本次最终退出码和产物核验均通过。
