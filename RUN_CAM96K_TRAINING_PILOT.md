# CAM96k training pilot v2：五种方法、周期快照与冻结评估

当前入口为 `analysis.cam96k_training_pilot_v2`，schema 为 `cam96k_training_pilot_v2`。
旧 v1 的入口、六单元 schema、批次、prompt 和 smoke 历史保留，不能把旧 manifest 改成十单元或更改旧提交身份。
本文件描述 v2；v1 的历史操作说明可在提交 `048ce8f0c765b91384039a9ba07c60e88007d6a2` 中查看。

## 计划及固定配置

| ID | 方法 | 共同 seed | 接受的 32k 配置来源 |
|---|---|---|---|
| 0、1 | Shared MAPPO-TDec | 12、13 | Training100-best / shared_tdec |
| 2、3 | Independent MAPPO-TDec | 12、13 | Training100-best / independent_tdec |
| 4、5 | Shared MAPPO-Combined | 12、13 | Training100-best / shared_combined |
| 6、7 | Independent Modified MADDPG-TDec | 12、13 | Training100-best / algorithm2 |
| 8、9 | Shared Modified MADDPG-TDec | 12、13 | Shared-MADDPG-TDec / shared2 / training/a01 |

来源配置及配置/完成标记 SHA256 存在 `configs/cam96k_v2_sources.json`。
MAPPO 与 Independent MADDPG 来源运行提交为 `04606e15f26f316728430e53a0e0d1e225bdb3e2`；Shared MADDPG 为 `c459115c58d7a465f3a4f2358a6385c649e42019`。
来源只用于代码、配置核对，不加载历史权重、replay 或训练进度。
各方法相对其已接受来源只改变 CAM、运行名称和输出路径；周期保存策略独立记入新 manifest，不改变训练配置的原超参数。

十次均 CAM=96,000 bit/周期，P5/N4/K3/gap25，1ms/时隙，每周期 100 时隙，从头训练 2500 周期。
共 25,000 周期、2,500,000 训练环境步。各 seed 独立 best100，另存 final。
MADDPG 训练探索 sigma=0.3 fixed；冻结动作 noise=0。训练 target policy smoothing sigma=0.2、clip=0.5 保持原样。
Shared MADDPG 保留历史一个共享 Actor、十个局部任务 Critic、全局双 Q 及对应目标网络；共享目标为 `(global_loss+sum_leader_task1_task2_losses)/P`，一次 Adam/目标 Actor 更新。
不启用 NoiseDecay，不增加 Parvini 非 TDec MADDPG。

## Windows 环境与正式批次

```powershell
$ErrorActionPreference = 'Stop'
$repo = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction0804'
$python = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\.envs\aoi_cuda310\python.exe'
$study = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction-diagnostics\actor-sharing-study'
$batch = Join-Path $study 'CAM96k-training-pilot\cam96k_20261025_v2'
Set-Location -LiteralPath $repo
```

使用现有 Python3.10.20、torch2.11.0+cu126 和 RTX4090，不升级依赖。
正式 manifest 由开发交付后绑定最终提交创建；Grok 使用已经创建的新批次，不重新 init。
需要创建独立批次时，先核对交付 SHA 和干净工作区，再执行唯一名称的 init；不能覆盖原目录。

```powershell
$commit = (git rev-parse HEAD).Trim()
# 仅对尚不存在的新批次执行 init；交付的新正式批次已初始化时跳过。
# & $python -B -m analysis.cam96k_training_pilot_v2 init --study-root $study --batch-id NEW_UNIQUE_BATCH_ID --expected-commit $commit
& $python -B -m analysis.cam96k_training_pilot_v2 preview --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preview failed' }
& $python -B -m analysis.cam96k_training_pilot_v2 preflight --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preflight failed' }
& $python -B -m analysis.cam96k_training_pilot_v2 status --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'status failed' }
```

preflight 检查十个来源、依赖、CUDA 运算、实际初始 CAM、MADDPG replay 与系统资源。
生产 replay 每进程 49,250,000 bytes，两路合计 98,500,000 bytes；网络、优化器、诊断和 CUDA 上下文另计，实际峰值由 smoke 资源记录给出。
资源不足直接报告，不自动改并发或 OOM 重试。

## 训练两路，评估八路，分别配置

训练默认并发 **2**：单个监督进程按 `(0,1) → (2,3) → (4,5) → (6,7) → (8,9)` 执行。
每组两个 seed 全部完成并验收后才进入下一组，不跨组混排、不启动多个监督进程。
study 训练槽仅限制训练，不应用于冻结评估。监督锁和单元锁保留，遗留锁不自动删除。

```powershell
& $python -B -u -m analysis.cam96k_training_pilot_v2 train --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'training failed; preserve attempts' }
& $python -B -m analysis.cam96k_training_pilot_v2 verify-training --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'ten-run training verification failed' }
```

只有十次训练全部通过，才执行 50 个 best100 单元。冻结默认并发 **8**，单个评估监督进程，每个子进程独占一个 `(方法, seed, best100, world, repeat0)` 单元。

```powershell
& $python -B -u -m analysis.cam96k_training_pilot_v2 evaluate --batch-root $batch --eval-workers 8
if ($LASTEXITCODE -ne 0) { throw 'evaluation failed; preserve attempts' }
& $python -B -m analysis.cam96k_training_pilot_v2 summary --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'summary failed' }
```

每 world 冷重置一次，预热5周期不计分、计分100周期；worlds1001–1005 为已使用探索集。
MAPPO 随机采样 repeat0；MADDPG noise0，保留各自设备、精度、动作解码和 RNG 派生。
默认不评估完整 final 或周期快照。Uniform 只读核验后复用，stream12/13 不是训练 seed。

后台运行使用 `scripts/run_cam96k_v2_supervisor.ps1`。它等待 Python 监督进程退出，检查原生退出码，再执行 verify-training/summary，最后原子发布唯一完成回执。
启动成功或 PID 消失不等于通过；必须检查回执的 `exit_code=0`、`validation_exit_code=0`，再核对批次验收文件。
交付 prompt 包含隐藏后台启动、日志父目录、PID 记录和回执检查的完整命令。

## 周期策略快照

每个正式 run 除 best100/final 外，保留：

```text
policies/policy_episode_000500.pt
policies/policy_episode_001000.pt
policies/policy_episode_001500.pt
policies/policy_episode_002000.pt
policies/policy_episode_002500.pt
policies/index.json
```

角色为 `periodic_policy_snapshot`，schema 为 `periodic_policy_snapshot_v1`。
独立方法包含全部五个 Actor；共享方法包含一个共享 Actor；Actor 所需状态随 state_dict 保存。
payload 包含算法/变体配置、共享标志、seed、episode、环境步、policy_version、保存时序、配置/hash、Git 身份和角色。
索引记录各文件大小、SHA256 及元数据。保存采用原子写入和独占防覆盖，保存前后恢复 Python/NumPy/Torch/CUDA RNG，不改变参数或 train/eval 模式。

MAPPO 在该周期结束、原计划的 PPO 更新后保存；不额外 consume rollout 或更新。
MADDPG 在对应周期在线更新后保存，不额外调用 learn、不改变 replay、探索或 target smoothing。
episode2500 快照必须与 final 的 Actor 张量逐一相同；角色和元数据不同，文件 SHA256 可不同。
原 best100 仍逐周期 rolling100、严格改善、最早并列；MAPPO best100 仍在待执行 PPO 更新前保存。
快照不参与 best100 选择，也不宣称支持训练恢复。

明确加载/验证入口：

```powershell
$run = Join-Path $batch 't\a01\cam96k_shared_tdec_s12'
& $python -B -m analysis.cam96k_training_pilot_v2 verify-snapshots --batch-root $batch --run-dir $run
if ($LASTEXITCODE -ne 0) { throw 'snapshot verification failed' }
```

Python 可使用 `runtime.periodic_policy.load_periodic_policy(path, expected_seed=12, expected_episode=500)` 或 `load_periodic_actors(...)`。
加载器独立验证角色、索引 SHA256、配置/hash、Git、seed、episode、policy_version、网络数量和严格 state_dict，不伪装成 best/final，不依赖伪造的 COMPLETE。

## 监控、验收与重试

`records/heartbeat_<ID>_<attempt>.json` 每周期更新；MADDPG 分别记录 critic learning steps、Actor optimizer steps 和 policy_version，不套用 PPO 更新数。
子进程日志、进程退出码、每组资源采样都在新批次内。资源记录包含 GPU 总用量、CPU 可用内存及每个子进程工作集。
若没有进度，先检查心跳、进程、日志和 I/O，不只看 GPU 利用率。保留失败 attempt，不因成绩重跑。

```powershell
& $python -B -m analysis.cam96k_training_pilot_v2 status --batch-root $batch
Get-ChildItem -LiteralPath (Join-Path $batch 'records') -Filter 'heartbeat_*.json'
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine
nvidia-smi
```

技术失败诊断后，只重跑实际失败单元，使用新 attempt 和真实原因；policy-only 不支持完整恢复。
已完成单元完整复核后跳过；原生完成但接受记录缺失时验收既有产物；禁止覆盖或自动删除。

正式最终验收至少为：accepted_training_runs=10、training_episodes_total=25000、training_environment_steps_total=2500000、periodic_snapshot_records=50、accepted_eval_units=50、platoon_rows=250、cam_deadlines=25000、scored_agent_slots=2500000、n_training_seeds_per_method=2、smoke=false。
训练验收另要求10个 best100、10个 final、各 run 快照周期集合恰为[500,1000,1500,2000,2500]。
统计保留 CAM 分子分母、线性mW、mean/worst AoI、截止剩余bit、payload比例、失败剩余bit（无失败NA）和 reward 分量。
先 per_seed 再 across_seeds；world/周期/车队/Random stream 不增加训练 seed 数。

## 开发 smoke 与历史保护

`init --smoke` 必须使用含 smoke 的新唯一批次名。
MAPPO 每 seed 5周期，MADDPG 每 seed 1周期，保持完整生产网络、replay、设备和原优化参数；总3400训练环境步。
开发专用快照间隔1周期，生产间隔仍固定500。冻结 smoke 为10 run × worlds1001/1002，共20单元，预热5、计分2，默认八路。
`compare-smoke` 对混合八路中的前8单元做串行参考，要求全部原始数组及物理 RNG probe 完全相同。
短周期规则和结果均明确标smoke，不能混入正式结果。生产500周期触发、2500快照/final一致性及 best100 时序使用合成测试，不启动正式长训练。

```powershell
# 仅对独立 smoke 批次执行；正式批次禁止调用 compare-smoke。
# & $python -B -m analysis.cam96k_training_pilot_v2 compare-smoke --batch-root $smokeBatch
```

v1、既有 smoke、历史权重、配置、接受记录和真实 Git 身份均保留。
不 push、不合并main、不升级环境、不启动seed8–11、128k/160k、NoiseDecay或额外方法。
