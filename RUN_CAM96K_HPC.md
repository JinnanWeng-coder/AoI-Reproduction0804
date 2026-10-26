# HPC-EEE：36 次全新训练、180 学习策略＋60 Uniform CPU 评估

先读本仓库 [HPC 执行规则](HPC_EXECUTION_RULES.md)。本入口为
`analysis.cam96k_hpc`，schema=`cam96k_hpc36_cpu240_v1`。
这是独立 HPC 批次，不接管或修改实验室 Windows 批次。**不设独立 HPC pilot**。
一次提交全部正式任务及依赖，不等于同时运行全部任务，不保证无缝获得调度资源。

## 固定矩阵

训练 ID：0–5 Shared MAPPO-TDec；6–11 Independent MAPPO-TDec；12–17 原有
Shared MAPPO-Combined；18–23 Independent Modified MADDPG-TDec；24–29 Shared
Modified MADDPG-TDec；30–35 新 Shared MAPPO-NoTDec。各方法依次 seed8–13。
所有 36 个 run 均从头训练，不能把实验室旧权重计入新增训练。

CAM=96000 bit，P5/N4/K3/gap25，2500 周期×100 个1ms槽。
总训练90000周期、9000000环境步。五种既有方法复用嵌入的已接受配置；
新 NoTDec 只从 Shared-TDec 模板切换明确的新 variant。所有原来源标签及路径作为
历史参考保留，不要求 HPC 挂载 Windows 盘，也不伪造历史来源文件已存在。
历史 MADDPG profile=`paper_faithful` 原样保留，沿用原 CAM96k 训练对象路径，
不将历史配置改标签来通过 active-new-config 的 profile gate。

36 个 run 全部验收，重新计算每个 best100 rolling100 训练回报后，封存
`records/selection.json`：每方法六 seed 中最高者胜出，精确同分取小 seed；
run 内保持原逐周期严格改善、最早并列规则。MAPPO best 在待执行 PPO 更新前保存，
MADDPG 在当周期在线更新后保存。不得使用冻结成绩选择 seed、final 或中间快照。

学习评估 ID0–179：world-major（1001–1030），每 world 按上述六方法顺序。
每方法固定同一个选定 best100；MAPPO 随机 repeat0，MADDPG 动作 noise0。
180 单元、900 车队行、90000 CAM 截止事件、9000000计分车队时隙。
实际每方法只评估一个训练 seed，不计算六训练seed平均/SD。

Uniform ID180–239：stream12 的30world，再 stream13 的30world。无训练、无选模，
stream 不计训练 seed。沿用已验证 fd99f8810a834e8 的 `draw_actions` 和 PCG64：
`SeedSequence([stream,world,0x52414E44,1])`，每槽依次抽 RB[P]、mode[P]、fraction[P]；
功率1–30dBm均匀，绝不是mW均匀；动作 RNG 不消耗物理 RNG。
生成本轮独立结果，不伪装旧五world为30world。
60单元、300车队行、30000截止事件、3000000计分车队时隙独立验收。

240 单元都使用 CPU、不申请GPU；每world冷启动一次、5 warmup不计分＋100 scored，
每周期100槽。物理环境、payload/deadline/带宽/功率范围一致。
CPU采样流与历史CUDA采样流不承诺相同；本批锁定统一CPU协议，记录实际运行库与设备。
world1001–1030仍是已用探索集，不能称新未见确认集。

## 固定代码身份和传输

交付的 Git bundle 含 `research/HPC`，用 `git bundle verify` 核验后导入新唯一引用，
建立独立 `research/HPC-run-*` 分支或 detached 快照。保留既有 HPC checkout，不 reset/覆盖。
`algorithm_commit` 是交付实现 SHA；`launcher_commit` 是本批实际完整 HEAD。
默认两者相同。Grok如需改HPC路径、日志、资源、依赖/并发，可在使用前提交以下白名单文件：

- `hpc/cam96k_hpc_common.sh`
- `hpc/aoi_cam96k_hpc_train.sbatch`
- `hpc/aoi_cam96k_hpc_eval_cpu.sbatch`
- `hpc/aoi_cam96k_hpc_analyze.sbatch`
- `hpc/submit_cam96k_hpc.py`

这些启动提交必须是 algorithm_commit 的后代，其他文件不得变化，不能修改算法或
实验配置以修复调度问题。新批次绑定真实 launcher HEAD，原算法SHA单独保留。
任务启动后不改其代码依赖或旧manifest；完整恢复仍须同 launcher、分支、设备、运行库。
可动态 `scontrol ... ArrayTaskThrottle=...`，不需要修改运行快照。
如果运行中需要修正尚未执行的调度脚本，必须另建干净scheduler快照并提交白名单改动；
保持原 `PROJECT_DIR`、`AOI_LAUNCHER_COMMIT` 和manifest不变，导出
`AOI_SCHEDULER_DIR`、`AOI_SCHEDULER_COMMIT` 指向新调度快照，只用其sbatch启动后续任务。
common会校验该提交相对原算法提交仅改白名单，并记录每job真实scheduler提交、diff和脚本副本；
Python算法仍由原固定checkout执行，因此既不拒绝授权调度修改，也不把新脚本冒充原SHA。
不要把 `PROJECT_DIR` 指向新checkout而造成已接受训练来源身份漂移。

## 初始化、作业内预检、一次提交全部链路

HPC默认根 `/eeedata/sgxjw2/Parvini-TVT2023-reproduction`，研究根为同级
`AoI-Reproduction-diagnostics/actor-sharing-study`，Python参考
`/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python`。Grok须先查实际存在和版本；
需要Python3.10/Torch2.11.0+cu126及锁定依赖，缺失则报告，不自动升级或降低配置。
本地路径只是开发工作区，绝不能照搬到Linux。

在固定快照中，完整命令由交付prompt提供。基本入口：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PROJECT_DIR="$PWD" PYTHON_BIN AOI_BATCH_ROOT AOI_LAUNCHER_COMMIT
"$PYTHON_BIN" -B -m analysis.cam96k_hpc init --study-root "$STUDY_ROOT" --batch-id "$BATCH_ID" --algorithm-commit "$ALGORITHM_COMMIT"
"$PYTHON_BIN" -B -m analysis.cam96k_hpc preflight --batch-root "$AOI_BATCH_ROOT"
"$PYTHON_BIN" -B -m hpc.submit_cam96k_hpc submit --batch-root "$AOI_BATCH_ROOT"
"$PYTHON_BIN" -B -m hpc.submit_cam96k_hpc submit --batch-root "$AOI_BATCH_ROOT" --submit
"$PYTHON_BIN" -B -u -m hpc.submit_cam96k_hpc monitor --batch-root "$AOI_BATCH_ROOT" --interval 60
```

可配置 `--partition`、`--cpu-partition`、`--account`、`--train-hours`、`--eval-hours`。
默认 Q10/L20 训练4CPU/32G/12h；CPU评估4CPU/16G/4h；验收选模及汇总4CPU/32G/4h。
这些是申请值，不是本地测得的生产任务耗时。按集群限制先核查、必要时在提交前调整。
不得为校准另外排队 pilot。观察已提交正式任务的 elapsed/MaxRSS/原始产物，记录成本，
只能在授权范围调整尚未运行任务资源/并发，不因成绩好坏决定继续。

提交器登记并顺次一次调用四次sbatch：

```text
train:    --array=0-35%2          GPU, 每task仅一个run
gate:     afterok:<train>        CPU, 全部36训练验收＋封存六胜者
evaluate: afterok:<gate>         --array=0-239%2, CPU含Random
finish:   afterok:<evaluate>     CPU, 双组验收/汇总/export
```

依赖确保训练、选模、评估、汇总不交叠。数组起始2，监控器结合 Slurm 节点分区、已分配
CPU/内存/typed-L20、关联账户限制证据调整 ArrayTaskThrottle，最多8；Slurm始终负责真正
资源授权，低GPU利用率不是分配依据。账户/分区资源信息不可见时不扩至8。
无两个seed组屏障，无每task再启动2/8子进程，无第二个Random数组造成并发倍增。
监控不是计算单元，不另外占GPU；应在站点允许的登录/会话/已申请控制CPU资源上执行。

提交前检查 `df -h`、inode、用户/账户quota。preflight输出所有MADDPG replay原始字节和
180份完整状态的replay存储下界，**不含网络、Adam、指标、恢复副本**；必须额外预留足够
真实空间。不能用精简证据包大小估算原始占用。可观察正式run逐步修正预测。

## 保存、完整恢复、失败依赖

每run保留best100/final；500/1000/1500/2000/2500同时保存独立策略与完整状态。
全部为180条策略、180条完整检查点、36best100、36final。
复用原 `cam96k_full_training_state_v1`，包含网络/目标、Adam、归一化、replay/未消费rollout、
环境和全部RNG、网络模式、计数、指标、rolling100和历史best Actor。NoTDec只有全局与本地
总回报两套价值流，本地第二storage槽为零兼容位，无第二任务网络/优化器/归一化。
保存不更新参数、不额外消耗RNG；NoTDec和旧五方法均做短序列对比。

技术中断：先查sacct、原task PID/节点/命令行/锁，确认原进程退出；残留锁人工归档，不自动删。
完整检查点须当前run/seed/episode/配置/launcher/设备/库一致，不能用best/final恢复。
例如原ID30在1200中断，从1000恢复（明确重算1001之后，原被回滚尾部由核心归档）：

```bash
export AOI_ATTEMPT=a01 AOI_RESUME_EPISODE=1000 AOI_RETRY_REASON='verified technical interruption, original process exited'
sbatch --array=30%2 hpc/aoi_cam96k_hpc_train.sbatch
```

原生COMPLETE存在但后置验收失败：先验证已有产物，同ID命令会验收并登记，不重训。
无有效完整边界或确认为损坏：保留旧attempt，技术诊断后使用新attempt和真实reason从头，
不能因低reward重跑。评估部分目录用新attempt；已有原生EVAL_COMPLETE先核对数组再补验收。

训练数组某task失败会令旧gate/evaluate/finish依赖无法满足。不能重复提交36或240矩阵。
保留原提交记录，确认仅这些后续作业处于DependencyNeverSatisfied/held且未执行，记录并
取消这些已失效的后续依赖作业。只补失败或未运行绝对ID；任务覆盖齐后重发新gate，再用
afterok依赖提交尚未运行评估ID数组及新finish。若评估部分已完成，只补其缺口；已有记录
会严格核验后跳过。所有新job/命令、取消原因、sacct及真实attempt写进批次records。
提交回包不确定时检查intent和Slurm，绝不再次运行整个submit命令。

## 汇总与回传

`tables/world_platoon.csv`、`world_summary.csv`、`selected_policy_summary.csv`、
`training_candidates.csv`，以及单独Random逐world/车队、聚合与verification。
保留CAM分子分母、AoI、平均线性mW、截止剩余bit、交付比例、无失败时NA、reward各分量。
严格核对方法×seed、方法×world、Random stream×world×platoon集合，不只总数。
技术PASS与claim_support分开；不要求共享/NoTDec优于其他方法。

正常finish自动生成 `core_evidence.tar.gz`，只含CSV/JSON/报告/来源与完成记录，
NPZ、policy、完整训练状态及原始日志留HPC，不另做SHA256清单。
回传runtime_report、实际命令/jobIDs/sacct/失败与资源/真实原始体积、完整world列表及中性摘要。
MaxRSS若为空，保留null/空值，不能估计。监控结束后最新sacct也要作为独立文件回传。
不启动其他算法、payload、新world、final或中间快照批量评估。

## 本地开发验证

`tests/test_cam96k_hpc.py`做配置/矩阵/NoTDec/保存中性/六结构恢复/脚本身份/Uniform回归；
`analysis.smoke_cam96k_hpc --root <唯一带smoke目录> --training-device cuda:0`
为六周期、100槽、小网络/replay的开发专用fresh-process检查，含CPU串行/8进程逐数组和
外生RNG一致性。所有输出smoke=true，不是正式训练候选，不以其耗时预测生产规模。
