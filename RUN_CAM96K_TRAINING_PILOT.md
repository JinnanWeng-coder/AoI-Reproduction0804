# CAM96k 从头训练与冻结评估

本入口只运行三种 MAPPO（Shared TDec、Independent TDec、Shared Combined），各保留 seed12、seed13。
正式预算是每次 2500 周期 × 100 时隙，共 1,500,000 环境步。CAM 为十进制 96,000 bit/周期。
P5、N4、K3、gap25、1ms/时隙及原算法、奖励、移动和信道语义保持原样。

开发基线为 `ee05154233c9122f5cf6e57e13cf372a60bdf843`。
使用已验证哈希的 `configs/training100_fresh_sources.json`，复用 Training100 的配置派生和模型选择。
本轮源模型本身在 96k 训练，现有冻结评估器直接保留该 CAM 值；不需要引入归档分支的跨负载修改。
Combined 仍是联合观测输入、输出五个车队总回报价值的集中式 Critic。

## 环境与批次

在 Windows PowerShell 中设置：

```powershell
$ErrorActionPreference = 'Stop'
$repo = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction0804'
$python = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\.envs\aoi_cuda310\python.exe'
$study = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction-diagnostics\actor-sharing-study'
$batch = Join-Path $study 'CAM96k-training-pilot\cam96k_20261025_v1'
Set-Location -LiteralPath $repo
```

Python 3.10.20，PyTorch 2.11.0+cu126，CUDA runtime 12.6，RTX 4090。
preflight 会逐项检查 `requirements.cuda.lock.txt`、CUDA 矩阵运算、六份配置和环境实际初始需求。
使用现有环境，不升级依赖，不自动回退 CPU。

正式批次仅在干净的 `research/cam96k-training-pilot*` 分支创建，固定当前提交；之后每条命令复查提交。
交接 prompt 必须明确填写交付 SHA；下面的变量用于已核对 SHA 后的日常操作，不替代交接核对。

```powershell
$commit = (git rev-parse HEAD).Trim()
# init 只执行一次。已存在的批次直接使用 preview/status；不删除或重建。
& $python -B -m analysis.cam96k_training_pilot init --study-root $study --batch-id cam96k_20261025_v1 --expected-commit $commit
if ($LASTEXITCODE -ne 0) { throw 'init failed' }
& $python -B -m analysis.cam96k_training_pilot preview --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preview failed' }
& $python -B -m analysis.cam96k_training_pilot preflight --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preflight failed' }
```

`preview` 不训练，输出六个单元的全部有效配置和相对于历史配置模板的字段差异。
`init`、preflight、status 不启动训练或冻结 rollout。

## 正式训练（由 Grok 启动）

```powershell
& $python -B -u -m analysis.cam96k_training_pilot train --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'training failed; inspect preserved logs and attempts' }
& $python -B -m analysis.cam96k_training_pilot verify-training --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'six-run verification failed' }
```

训练并发固定上限 **2**，顺序为：

| 组 | ID | 算法 | 同时启动的 seed |
|---|---|---|---|
| 1 | 0、1 | Shared MAPPO-TDec | 12、13 |
| 2 | 2、3 | Independent MAPPO-TDec | 12、13 |
| 3 | 4、5 | Shared MAPPO-Combined | 12、13 |

每组两个 seed 全部完成并验收后，才进入下一组。每个 seed 使用独立进程、初始化、输出、日志和 checkpoint。
只跳过经过完整重新验收的已接受单元。进程退出为零并不等于验收通过。
批次 supervisor 锁、单元锁与 study 范围的两个训练槽防止重复监督、重复写入和超过两路训练。
锁采用独占创建；不自动清除遗留锁。遇到锁冲突先检查记录的 PID 与命令行，不终止无关任务。

`t/a01/cam96k_<model>_s<seed>/` 中保存 `policy_best_training.pt`、`policy_final.pt`、完整训练数组、配置、来源和选择 CSV。
best100 仍按连续 100 周期的平均训练回报、严格改善和最早并列规则选择。
best 保存于窗口末周期结束且待执行 PPO 更新之前；final 在第 500 次更新后保存。
不比较两个 seed 来淘汰其中一个，不使用冻结指标挑 checkpoint。

## 监控、失败与重试

```powershell
& $python -B -m analysis.cam96k_training_pilot status --batch-root $batch
Get-ChildItem -LiteralPath (Join-Path $batch 'logs')
Get-Content -LiteralPath (Join-Path $batch 'records\heartbeat_0_a01.json')
nvidia-smi
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine
```

`records/heartbeat_<id>_<attempt>.json` 每周期原子更新，包含周期、环境步、PPO 更新数、耗时、PID 和 UTC 时间。
训练子进程每 5 周期输出进度，监督进程每 15 秒输出存活状态。资源可结合进程工作集与 `nvidia-smi` 查看。
`records/process_*.json` 记录实际启动命令、日志、PID、退出码、耗时。
没有进度时先检查心跳、进程、日志与资源；不能仅凭 GPU 利用率低判定卡死。

只读 policy checkpoint 不支持可靠续训。保留所有失败 attempt，诊断技术原因后才使用新的 attempt 从头重跑：

```powershell
# 示例：仅修复确有技术故障的 ID 0，原因必须写实；不因成绩低或曲线差重跑。
& $python -B -u -m analysis.cam96k_training_pilot train-unit --batch-root $batch --id 0 --attempt a02 --reason '填写已确认的技术故障原因'
```

若原生 COMPLETE 已存在但接受记录未写完，重新执行会验收原产物，不重复训练。
若两种 attempt 都有完整结果，停止并调查；不自行按分数挑选。

## 冻结评估与汇总（六次训练全部通过后）

```powershell
& $python -B -u -m analysis.cam96k_training_pilot evaluate --batch-root $batch --eval-workers 2
if ($LASTEXITCODE -ne 0) { throw 'evaluation failed; inspect preserved attempts' }
& $python -B -m analysis.cam96k_training_pilot summary --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'summary verification failed' }
```

只使用各 seed 自己的 best100。3 方法 × 2 训练 seed × worlds1001–1005，共 30 个单元，repeat0。
每 world 开始时冷重置一次，5 个预热周期不计分，然后 100 个计分周期，每周期 100 时隙。
保留现有 policy RNG 派生与物理 RNG 隔离；随机采样；无 Critic、优化器或 rollout buffer；权重不变。
这些 world 是已用过的探索集，不能描述为未见确认集。冻结并发独立设置 1–8，不改变训练两路上限。

输出 `tables/world_platoon.csv`、`world_summary.csv`、`per_seed.csv`、`across_seeds.csv` 和 `verification.json`。
包含 CAM 成功分子/分母、AoI、逐样本转换后的线性 mW、截止剩余 bit、payload 交付比例、失败剩余 bit（无失败记 NA）、reward 分量。
world 内最差车队先汇总该车队全部计分周期，再取最小完成率或最大 AoI。
先分别报告两个训练 seed，再汇总；训练 seed 数始终是 2，world/周期/车队均不是额外训练 seed。
汇总和验收可以重复执行；既有表格必须与重新计算完全一致，不覆盖已有证据。

既有 96k Uniform 仅在完整环境与协议身份核验后复用：
`../AoI-Reproduction-diagnostics/actor-sharing-study/Random-frozen-eval/P5_N4_gap25/random_uniform_stream12_13_cam64000_96000_world1001_1005_20261024_v1/`。
两条 Random stream 是随机动作流，不是两个训练模型。不重新训练或运行 Random。
历史记录里的 `9e84d25 + dirty` 保留原样，不改成事后归档提交。

```powershell
$uniform = Join-Path $study 'Random-frozen-eval\P5_N4_gap25\random_uniform_stream12_13_cam64000_96000_world1001_1005_20261024_v1'
& $python -B -m analysis.cam96k_training_pilot reuse-uniform --batch-root $batch --uniform-root $uniform
if ($LASTEXITCODE -ne 0) { throw 'Uniform reuse verification failed' }
```

这条命令不运行 rollout，重新核对十个 96k 单元的完整环境、协议、数组、分子分母和 SHA256，
在新批次 `protocol/uniform_reuse.json` 中记录来源和复用指标；不改动旧批次。

## 开发 smoke

独立批次名必须含 `smoke`，`init --smoke` 将每个训练限制为 5 周期，并将每个冻结 smoke 限制为 world1001、5 个预热＋2 个计分周期。
六个 seed 共 3000 训练环境步，六个冻结 smoke 共 4200 环境步，其中仅 1200 步计分。
smoke 的 final 只用于检查加载和冻结路径，不能作为正式 best100 或正式结果。
best100 完整窗口、严格加载、策略版本和同分边界通过合成序列测试；不为它进行正式训练。
smoke 允许开发工作区 dirty，真实记录 dirty 和运行时 HEAD；正式批次不允许。

```powershell
# 开发用新批次示例；不要对正式批次传 --smoke。
& $python -B -m analysis.cam96k_training_pilot init --study-root $study --batch-id smoke_NEW_UNIQUE_ID --expected-commit $commit --smoke
```

禁止启动 128k/160k、seed8–11、MADDPG 或完整 final 冻结评估；不 push、不并入 main、不清理历史目录。
