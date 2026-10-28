# CAM96k Uniform Random：训练阶段 rollout 补充

本轮只新增独立 Random 批次。已有36次学习训练、选优、180学习策略冻结和60 Random冻结的合同、记录、目录及监控器均不改变；不调用旧提交链。代码基点为 `a7075a3d1d9cf6cb9deab9be1ae17a8b1ff57fed`。

入口 `python -B -m analysis.cam96k_random_training` 是历史 `168028fa28a32f9d41a7f7697e27ee4a391b79b5` 的 Uniform 训练入口移植，保留原采样、RNG、训练生命周期及日志归约。旧文件不在当前分支，故新增独立模块。核心算法、环境、奖励和已有 HPC 启动器没有修改。

## 固定协议

- P5/N4/K3/gap25、CAM96000 bit；独立 rollout seed8–13各2500周期，每周期100个1ms时隙。
- 每seed只 `reset_world(seed)` 一次，然后 `start_episode(0..2499)`。保持跨周期AoI、干扰及物理状态；需求/截止时间按原训练环境重置。无评估warmup，不删最初5周期。
- 原 `configs/training100_fresh_sources.json` 四份配置的环境字段须一致；只将CAM32000改为96000，训练预算改2500，设备CPU、checkpoint模式none、运行名另设。配置内algorithm/network/output_root是未使用的历史模板字段；本入口不构造学习器，不使用其旧输出路径。实际写入路径只有新批次。
- 测试逐项对齐六种HPC学习配置的环境字段。实际历史训练轨迹未逐槽对账，不将相同seed声称为已核实的历史外生轨迹匹配。
- 每时隙每车队独立均匀RB/模式，功率 `1+29*u` dBm，`u~Uniform[0,1)`；不是线性mW均匀。
- 动作 `PCG64(SeedSequence([seed,0x52414E44,1]))`，按RB[P]、mode[P]、power_fraction[P]顺序抽样。环境使用自身 `default_rng(seed)`，动作不消耗物理RNG。这是训练种子协议，未套用冻结world/stream协议。
- 无Actor、Critic、优化器、梯度、policy、best100或选模；六个seed是独立rollout重复。`last100`仅为末100周期汇总。
- 保存每周期总量与分母、逐车队数据：AoI、CAM截止成功/事件数、线性mW、dBm、reward及三分量、严格AoI>50、V2I attempt/reset、截止剩余bit。reward沿用float32日志后float64归约，global项按每车队计入，权重必须为1；功率先逐槽转mW再归约。
- CAM取周期末第100时隙；reset来源为已核实环境的post-step AoI≈1，不用AoI下降代替。记录动作直方图、初末RNG、冷启动及1/100/每500/末周期的物理探针。
- 每500周期探针是进度数据，非可恢复checkpoint。本轮不引入恢复框架：技术中断保留原attempt，在新attempt从头重跑该seed，记录额外成本。无按性能重跑。

## CPU Slurm（无需独立pilot）

先读 `HPC_EXECUTION_RULES.md`。使用独立干净checkout，固定交付SHA；不修改旧运行checkout及远端独立scheduler分支。以下变量均显式export，正式命令使用实际交付SHA：

```bash
export PROJECT_DIR=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/deployment-checkouts/cam96k_random_<sha>
export PYTHON_BIN=/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python
export AOI_EXPECTED_COMMIT='<交付完整SHA>'
export STUDY_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study
export BATCH_ID=random96k_<唯一编号>
export AOI_RANDOM_BATCH="$STUDY_ROOT/CAM96k-Random-training/P5_N4_gap25/$BATCH_ID"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" -B -m analysis.cam96k_random_training init --study-root "$STUDY_ROOT" --batch-id "$BATCH_ID" --expected-commit "$AOI_EXPECTED_COMMIT"
"$PYTHON_BIN" -B -m analysis.cam96k_random_training preflight --batch-root "$AOI_RANDOM_BATCH"
"$PYTHON_BIN" -B -m analysis.cam96k_random_training submit --batch-root "$AOI_RANDOM_BATCH" --partition Q10
"$PYTHON_BIN" -B -m analysis.cam96k_random_training submit --batch-root "$AOI_RANDOM_BATCH" --partition Q10 --submit
```

已有同批次时不重复init或submit；先读manifest/records并查Slurm。提交器保留intent及每一步receipt；通信结果不明也拒绝再次提交完整链。默认2 CPU/4GiB/4h，无GPU；`--partition`、`--account`、`--hours`可按真实站点配置，运行中不更新固定checkout。

仅两个作业：`0-5%2`数组，task0..5对应seed8..13；随后 `afterok:<array>` CPU汇总及ZIP导出。每task一个Python进程，无本地嵌套进程池。数组初始并发2；根据Slurm分区、CPU/内存分配和账户/QOS可用配额，将同一数组提高至最多6，而非另提任务凑8。查询失败保留2，所有计算经Slurm调度。不重启旧monitor；Grok查看本短CPU批次即可。

```bash
# 从 records/submission.json 获取真实job号，确认资源/配额后才调整：
scontrol update JobId=<rollouts_job> ArrayTaskThrottle=<2到6>
squeue -r -j <rollouts_job>,<summary_job>
sacct -j <rollouts_job>,<summary_job> --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS -P
```

已接受完整结果仅在配置/来源/文件身份/全部计数复核一致时跳过；原生COMPLETE写出而record缺失时先验收，不再rollout。半成品拒绝覆盖，失败保留。查明确无活进程后才可移除已死本任务留下的锁，记录原因。重试只提交缺失/失败seed的新attempt，不重交6个全矩阵；汇总依赖据实际恢复链新建并保留原失败链。

恢复单元的底层命令（经CPU sbatch执行；环境变量AOI_RANDOM_STAGE、AOI_RANDOM_ATTEMPT、AOI_RANDOM_REASON同样受外壳支持）：

```bash
"$PYTHON_BIN" -B -m analysis.cam96k_random_training run-unit --batch-root "$AOI_RANDOM_BATCH" --seed 10 --attempt a02 --reason '具体技术原因'
"$PYTHON_BIN" -B -m analysis.cam96k_random_training summary --batch-root "$AOI_RANDOM_BATCH" --attempt a01
```

若汇总写到一半，保留 `analysis/a01`，另用 `--attempt a02`。summary为只验收/汇总，不触发rollout。不允许为了通过而删除坏行、换seed或改容差。站点调度适配优先用参数及sbatch覆盖；如需改本轮外壳，用独立脚本副本保留diff、实际副本及来源，Python固定checkout仍不变。算法/预算/统计修改先停止。

## 验收与交付

- 6个被接受run，seed集合恰为8..13；每run2500周期、250000环境步、12500截止事件。
- 总15000周期行、75000逐车队行、1500000环境步、7500000车队时隙、75000截止事件；无warmup、学习更新、模型或新增冻结评估。
- 逐车队/周期sum/count一致，AoI/CAM/reward/功率/剩余bit及动作直方图验证通过；sample SD按6个独立seed，slot/周期/车队不增加训练重复数。
- 产物 `analysis/a01/{episode_metrics,episode_platoon_metrics,summary,summary_across_seeds,episode_across_seeds}.csv`、`verification.json`、`REPORT_CN.md`。
- 同目录自动生成并逐文件核验 `core_evidence.zip`：仅CSV/JSON/报告/来源及接受记录；不含权重、NPZ、完整轨迹或日志，不制作SHA256清单。
- Grok另交简短runtime_report：实际commit/目录/命令、job IDs、sacct、并发调整、失败/重试、原始输出体积及环境；MaxRSS未提供则null，不能估算填入。该报告与sacct放批次根，避免事后改已封存records而影响ZIP复核。
- technical PASS只表示来源、计算和计数通过；不要求Random或某算法表现更好。

本地测试命令：`python -B -m pytest tests/test_cam96k_random_training.py -q`；包括短CPU运行、历史实现精确回归、相同seed复现、双进程与串行一致、合成正式计数、提交dry-run/幂等保护。测试不运行正式2500周期，不代表HPC已经通过。
