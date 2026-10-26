# CAM96k seed8–11：20 次训练与完整断点恢复

本入口为 `analysis.cam96k_seed8_11`，schema 为 `cam96k_seed8_11_full_state_v1`。
旧 v1/v2 和已完成的 seed12/13 批次保留原身份、契约及文件；本批次不混入旧结果。
代码开发阶段只运行开发测试和独立 smoke；正式训练及完整冻结评估交给 Grok 监督启动。

## 实验矩阵与配置来源

| ID | 方法 | seed |
|---|---|---|
| 0–3 | Shared MAPPO-TDec | 8,9,10,11 |
| 4–7 | Independent MAPPO-TDec | 8,9,10,11 |
| 8–11 | Shared MAPPO-Combined | 8,9,10,11 |
| 12–15 | Independent Modified MADDPG-TDec | 8,9,10,11 |
| 16–19 | Shared Modified MADDPG-TDec | 8,9,10,11 |

全部 CAM=96000 bit/周期、P5/N4/K3/gap25、100 时隙/周期、1ms/时隙、从头训练 2500 周期。
总预算 50000 周期、5000000 训练环境步。使用同一组 seed8–11，绝不加载 seed12/13 或旧32k权重。

继续以 `configs/cam96k_v2_sources.json` 的已接受 seed12 **配置**为各方法模板。
MAPPO 和 Independent MADDPG 来源为 Training100 a01（算法2对应 Independent MADDPG），原运行提交为
`04606e15f26f316728430e53a0e0d1e225bdb3e2`；Shared MADDPG 来源为 Shared-MADDPG-TDec/shared2 a01，
原提交为 `c459115c58d7a465f3a4f2358a6385c649e42019`。
相对这些32k模板只改 seed、cam_bits、run_name、output_root；相对上一批96k训练只改 seed/身份路径并新增完整保存。
`preview` 给出每个有效配置、差异、来源提交及快照计划。保存/恢复选项在批次及运行元数据单独记录。

MADDPG 保持 fixed 探索 sigma=0.3、target smoothing sigma=0.2/clip=0.5、synchronous_joint，
共享实现及损失聚合、所有 Critic、目标网络、replay、Adam、更新频率均不变。
冻结动作 noise=0 不会关闭训练探索或 target smoothing。没有 NoiseDecay 或 Parvini 非TDec变体。

## 调度与冻结评估

一个训练监督进程，训练并发上限2。每种方法先 seed8/9 同时运行，再 seed10/11 同时运行；
该方法四次训练全部完成并验收后进入下一方法。依次为三种 MAPPO、Independent MADDPG、Shared MADDPG。
保留 study 训练槽、监督锁、单元锁。完整检查点不会提高训练并发。

全部20次训练验收后，一个冻结评估监督进程，默认并发8，命令显式 `--eval-workers 8`。
5方法 ×4训练seed ×worlds1001–1005 ×repeat0 =100个 best100 单元，每world预热5、计分100周期。
各单元独占目录及日志，保留原设备、数值精度、动作解码和RNG协议。
MAPPO随机采样，MADDPG动作noise0，不学习、不评估完整final或批量中间快照。
worlds1001–1005仍为探索集。旧96k Uniform只读核验复用，stream12/13不改为训练seed8–11。

`verify-training`及评估监督启动前保留全部模型、周期策略和完整训练状态的深度验收。
每个评估子进程必须核对已有深度验收记录，并重新计算全部20个已接受训练输入文件的SHA256，
包括配置、provenance、完成标记、指标、best/final、选择记录及两类快照和索引。
文件或验收记录变化即拒绝运行；子进程不重复分配所有训练replay/Critic。
实际使用的冻结策略仍由原严格加载器检查角色、配置和网络结构，设备、采样、数值及RNG协议不变。

## 三类保存文件

每次正式训练继续保存原 `policy_best_training.pt` 和 `policy_final.pt`。
best100仍逐周期rolling100、严格改善、最早并列，每seed独立选择。
MAPPO best100仍在待执行PPO更新前保存，MADDPG在当周期在线更新后保存。

每500周期同时保留：

```text
policies/policy_episode_000500.pt       # 推理策略；独立5Actor/共享1Actor
policies/index.json                    # periodic_policy_snapshot
training_checkpoints/training_episode_000500.pt
training_checkpoints/index.json        # periodic_full_training_checkpoint
```

两个周期集合均恰为 `[500,1000,1500,2000,2500]`，原子写入、索引包含SHA256，拒绝覆盖已有周期文件。
MAPPO在当周期应执行的PPO更新完成后保存；未达到更新边界时完整保存未消费rollout。
MADDPG在当周期在线更新后保存。保存不额外learn/update/consume，不消耗RNG、不改变参数或train/eval状态。
两类episode2500检查点中的Actor必须与final一致，不能标为best100。

完整状态schema为 `cam96k_full_training_state_v1`，包含：

- MAPPO全部Actor/Critic、各Adam参数组及动量/step、value normalization缓冲、PPO计数和未消费rollout。
- MADDPG全部Actor/target Actor、10个局部任务Critic及目标、全局双Q及目标、各Adam、学习/Actor/目标更新计数、完整replay及位置。
- 环境物理状态及私有RNG、Python/NumPy/Torch/CUDA RNG、网络模式、累计训练指标。
- rolling100窗口、全部选择历史、当前best100元数据及对应Actor权重，避免恢复后丢失历史最优。
- 有效配置/hash、算法/共享/seed、episode/环境步/policy_version、保存时序、Git身份和运行库版本。

best/final及`policies`仍为推理文件，不能替代完整训练检查点。
完整恢复要求同提交、干净正式工作区、同配置/路径、同设备及运行库；不支持跨硬件/版本精确复现承诺。
episode2500状态也保留，可在最后权重或完成标记写出前中断时继续完成收尾，不增加训练步。
已有COMPLETE的训练拒绝恢复，先验收已有产物。

## 正式命令

```powershell
$repo = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction0804'
$python = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\.envs\aoi_cuda310\python.exe'
$study = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction-diagnostics\actor-sharing-study'
Set-Location -LiteralPath $repo
# $batch 由交付的Grok prompt给出，绑定最终交付提交，不能使用旧v2 manifest。
& $python -B -m analysis.cam96k_seed8_11 preview --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preview failed' }
& $python -B -m analysis.cam96k_seed8_11 preflight --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preflight failed' }
& $python -B -u -m analysis.cam96k_seed8_11 train --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'training failed; diagnose and preserve artifacts' }
& $python -B -m analysis.cam96k_seed8_11 verify-training --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'twenty-run training gate failed' }
& $python -B -u -m analysis.cam96k_seed8_11 evaluate --batch-root $batch --eval-workers 8
if ($LASTEXITCODE -ne 0) { throw 'evaluation failed' }
& $python -B -m analysis.cam96k_seed8_11 summary --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'summary failed' }
```

后台使用 `scripts/run_cam96k_seed8_11_supervisor.ps1`；每次启动使用唯一日志和receipt。
必须检查监督原生退出码及receipt的exit_code/validation_exit_code，只有两者为0才通过。
启动成功或PID消失不等于完成。具体隐藏后台启动、日志父目录和收尾命令见交付prompt。

## 中断恢复及历史保护

默认train不自动恢复部分训练，也不自动OOM重试。先确认故障原因及所有原监督/单元进程已退出。
残留锁只能核对PID、启动时间、命令行后人工归档；代码不自动删除锁。
原监督正常退出时已获锁会释放，重新执行会深度复核并跳过已验收单元。
确认为技术中断后可显式恢复所有未完成单元的最新完整边界：

```powershell
& $python -B -u -m analysis.cam96k_seed8_11 train --batch-root $batch --resume-incomplete --reason 'Confirmed technical interruption; original processes exited'
if ($LASTEXITCODE -ne 0) { throw 'recovery failed; retain all evidence' }
```

这仍是一个监督进程、两训练槽和原方法屏障。新单元仍从头开始；部分单元需有有效索引。
例如1100周期中断从1000恢复，回放1001起的数据；2500预算不变，实际重算成本另计。
需要指定某单元的较早边界时使用：

```powershell
# 仅在监督已退出时运行；ID与检查点必须属于同一单元。
& $python -B -u -m analysis.cam96k_seed8_11 train-unit --batch-root $batch --id 0 --attempt a01 --resume-incomplete --resume-episode 500 --reason 'Confirmed technical interruption; selecting indexed episode500'
if ($LASTEXITCODE -ne 0) { throw 'unit recovery failed' }
```

先严格核验所选完整文件及全部保留前缀的SHA/config/seed/Git/runtime，再把所选边界之后的周期文件、
未完成final/旧选择表和指标移至run内唯一`recoveries/rollback_*`目录，保留原始字节及原因记录。
恢复当前best、指标、选择器和索引，再继续相同episode序列。保留的较早周期文件不重写。
回滚有`RECOVERY_PENDING.json`事务记录，回滚中断需重发同一边界；不能换检查点掩盖故障。
无有效完整检查点时保留失败attempt，技术诊断后使用新attempt/真实reason从头训练；不伪装resume。
文件损坏或身份不符直接失败，不悄悄降级至更早边界，不因成绩差而重跑。

验证入口：

```powershell
$run = Join-Path $batch 't\a01\cam96k_shared_tdec_s8'
& $python -B -m analysis.cam96k_seed8_11 verify-full-checkpoints --batch-root $batch --run-dir $run
if ($LASTEXITCODE -ne 0) { throw 'full-state checkpoint verification failed' }
& $python -B -m analysis.cam96k_seed8_11 verify-snapshots --batch-root $batch --run-dir $run
if ($LASTEXITCODE -ne 0) { throw 'policy snapshot verification failed' }
```

Python检查入口为`runtime.full_training_checkpoint.load_full_checkpoint(path, expected_seed=8, expected_episode=500)`；
真正恢复由`runtime.runner.train(..., full_checkpoint_interval=500, periodic_policy_interval=500, training_reward_best=True,
resume_training_checkpoint=path)`及上面的锁定CLI完成。
不接受policy_only/best/final冒充full checkpoint，不绕过原模型严格校验。

正式验收要求20个accepted training、50000训练周期、5000000训练环境步、100个周期策略、100个完整检查点、
20个best100、20个final；每run两类周期集合恰为500/1000/1500/2000/2500。
冻结要求100个accepted eval、500车队行、50000 CAM deadlines、5000000 scored agent slots、每方法4训练seed、smoke=false。
保留原CAM分子分母、mean/worst AoI、线性mW、截止剩余bit、payload比例、失败剩余bit、reward分量、逐world/车队和per_seed/across_seeds统计。

## 开发验证

`tests/test_cam96k_seed8_11.py`覆盖五结构、完整保存开关中性、未消费PPO rollout恢复、
MADDPG目标/replay/优化器恢复、真实rolling100与best权重一致、角色/seed/episode/hash拒绝、
生产500触发的合成状态验证、回滚事务中断、20/100计数及训练2/评估8调度。
旧Training100、v1/v2及旧checkpoint回归继续运行。开发专用短间隔必须显式smoke。
生产网络CUDA短smoke与中断恢复证据和资源采样保存在新批次交付报告内；不计入正式结果。
