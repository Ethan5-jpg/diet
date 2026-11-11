# Experiment 2 Effect Modification v1

**Effect modification analysis starts here.**

本实验检验纤维与血糖的关系是否随菌群改变。正式上传目录为 v1，现有 101 条路线，包含上一版的 83 条和六类新方法。每类新方法分别使用口腔、肠道和两者联合的菌群输入。所有路线使用同一共同基线和家庭外折，每条在内折试 3 档设置及零交互。

主要结果是 89 条扩展候选组成的选择器 select_full 在第 0 轮的表现。第 1 轮对同一批家庭重新分折。主汇总对比 12 条基础岭、上一版的 71 条扩展候选、本次新增的 18 条，以及同一扩展池的保守选择流程。只使用发现集中的 3,343 人和 3,059 个家庭。主要指标是折外 ΔR²，+0.010 和 +0.050 分别表示多解释 1 和 5 个百分点。

## 上传和运行

截图中的 **v1** 文件夹位于交接目录下的 **experiment2_Effect_Modification/**。该位置可以直接运行。

    collaborator_handoff_20261102_v1/
        experiments/
            diet_effect_modification_20261109_v12/
                .python_path
        experiment2_Effect_Modification/
            v1/
                config.json
                server_run.sh

执行以下命令，使用 V12 保存的 Python 解释器。预检成功才启动完整实验。

    cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiment2_Effect_Modification/v1 &&
    IFS= read -r HPP_EXP2_PYTHON < ../../experiments/diet_effect_modification_20261109_v12/.python_path &&
    export HPP_EXP2_PYTHON &&
    bash server_run.sh --preflight &&
    bash server_run.sh --threads 2

预检应显示 **EXP2_CONFIG experiment2_effect_modification_20261111_v1_101 routes=101**。看到 **EXP2_SOURCE_READY discovery=4427 people=3343 households=3059** 表示数据入口检查通过。完整运行有十条 **OUTER_COMPLETE**，校验完成后会显示 **EXP2_COMPLETE** 和第 1 屏汇总。

回传截图优先第 1、9、10 屏，分别显示主比较、本次新增的 18 条路线和固定剂量诊断。第 8 屏显示局部剂量支持和保守选型。第 2、3、5、6、7 屏保留历次路线明细，第 4 屏显示完整拟合诊断。

    bash server_run.sh --screen 1
    bash server_run.sh --screen 9
    bash server_run.sh --screen 10

终端中断后，用相同线程数和运行名续跑。

    bash server_run.sh --threads 2 --resume

可以加 **--limit-outer 1** 先跑一折再暂停。若进程被强制终止且旧锁仍存在，在原进程所在服务器确认它已经退出后，用 **--resume --recover-stale-lock**。若出现 **EXP2_STOP**，保留报错附近的日志便于排查。

默认运行名为 **experiment2_effect_modification_20261111_v1_101**。运行结果保存在 **v1/runs_TRE_ONLY/** 内。目录中的旧进程完成后再更新代码。83 条和 65 条路线的旧运行名分别为 `experiment2_effect_modification_20261111_v1` 和 `experiment2_effect_modification_20261110_v1`，可以通过 `--run-id` 查看对应报告。续跑使用该运行的原代码、配置和环境。在新终端里先重新设置 **HPP_EXP2_PYTHON**。

[方法和指标](METHODS.md) · [101 条路线本地验证记录](research_records/LOCAL_VERIFICATION_101_ROUTES.md)
