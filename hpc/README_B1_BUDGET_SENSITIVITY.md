# B1 预算敏感性：Tier 1

本入口落实 2026-09-26 已确认方案。只新增24个250k从头训练和48个最终策略冻结评估；历史24个50k策略、48个评估仅引用，历史NPZ不在B1分析中重新加载。独立/shared × combined/TDec 两种价值学习配置 × training seeds8–13。250k=2500 episodes×100 slots；固定500次PPO更新、每次10 epochs。shared有1 actor，actor Adam总步数5000；independent有5 actors，总步数25000，每actor5000。现有训练器仅保存最终policy；不提供续训或中间预算策略。

固定代码基准 `d6121e876777841dc3d63800f7fc62ebb8fe9b41`。B1只新增外围入口。`code_audit` 要求当前checkout在research/mappo、干净，并且相对基准的整个 `aoi_v2x_reproduction/` 包及 `analysis/evaluate_mappo_feasibility_reward.py` 无差异。实际B1实现commit由批次创建时写入manifest，后续必须保持一致。历史训练commit另外保留，不冒记为实现commit。

## 路径与来源

默认仓库：`/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804`。
研究根：`/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study`。
唯一新批次根：`$STUDY_ROOT/B1-budget-sensitivity/P5_N4_gap25/<batch_id>`。batch_id仅字母、数字、下划线、连字符；存在即拒绝。

- independent两个配置：`existing-evidence/tdec-ab-v1/P5_N4_gap25/training/runs`，历史commit f0226ab。
- shared/combined：`E5-sharing-critic/P5_N4_gap25/training/runs`，历史commit b662789。
- shared/TDec：`existing-evidence/shared-actor-v1/P5_N4_gap25/training/runs`，历史commit 81c6a59。
- 开发评估：combined在E5，TDec两臂在shared-actor-v1，不能从训练目录猜评估目录。
- E3历史评估：`E3-locked-test/P5_N4_gap25/evaluations`；其原lock及技术修复附记只读。

历史小表必须已在原研究目录：

1. `E5-sharing-critic/P5_N4_gap25/analysis/e5_training_per_episode.csv`：唯一12000行历史episode；不用shared-actor中重叠副本。
2. 24个历史训练目录各自 `learning_diagnostics.json`：共2400个update。
3. `N1-training-world-revisit/P5_N4_gap25/analysis/n1_e5_development_world_agent.csv`。
4. `W1-service-intervals/P5_N4_gap25/` 下source_inventory、flow_statistics、flow_interval_frequency CSV及verification.json。
5. `E3-locked-test/P5_N4_gap25/analysis/` 下source_inventory、c1_world_agent、c2_flow_statistics、c2_interval_frequency CSV及verification.json。

preflight核实历史config/COMPLETE/policy payload及48份评估元数据、源路径、配置绑定和旧小表覆盖/闭合。读取policy payload只为确认历史来源，不实例化模型或环境。缺输入必须报具体缺口，不能补跑历史评估。

## 创建、映射及执行

在固定commit的干净仓库执行（先设置 `PYTHONDONTWRITEBYTECODE=1`，避免仓库缓存；初始化完成后TMPDIR/XDG_CACHE_HOME都指向新批次）：

```bash
python -B -m analysis.mappo_b1_contract init --study-root "$STUDY_ROOT" --batch-id "$BATCH_ID"
python -B -m analysis.mappo_b1_contract preflight --batch-root "$BATCH_ROOT"
```

init只创建一次，写24份派生config、manifest、历史来源快照和preflight状态。config由逐run历史resolved config派生，仅改episodes、run_name、output_root。历史sharing缺失/None保持false语义并单列记录。任何意外默认值填充或配置差异均失败。预算不改变checkpoint_every的原值，也不声称它会生成中间策略。

| train id | 条件 | seed |
|---|---|---|
| 0–5 | independent/combined | 8–13 |
| 6–11 | shared/combined | 8–13 |
| 12–17 | independent/TDec | 8–13 |
| 18–23 | shared/TDec | 8–13 |

eval id为 `2*train_id + world_set_index`；index0为development213–218，index1为e3_reused303–308。两集合各6world、warm5、scored100×100、baseline/stochastic/sequential_warm、原SeedSequence策略RNG规则。303–308是已查看的历史锁定复用集，不能称本轮未见确认。

完整pilot train IDs=0,6,12,18；eval IDs=0,1,12,13,24,25,36,37。正式总数包含pilot。其余train IDs=1–5,7–11,13–17,19–23；eval IDs=2–11,14–23,26–35,38–47。

三个sbatch都要求 `AOI_BATCH_ROOT` 和 `AOI_EXPECTED_COMMIT`，并调用 `hpc/b1_common.sh` 验证批次/代码。提交时必须显式把 `--output`、`--error` 指向 `$BATCH_ROOT/slurm_logs`。训练默认L20×1、8CPU、32G、2h；shared用 `--cpus-per-task=4`。评估默认L20×1、4CPU、16G、1h；汇总4CPU、16G、2h、无GPU。训练与评估先完成完整pilot，记录sacct的Elapsed/MaxRSS/ExitCode和输出体积，再按实测调整剩余作业内存/时限/并发；不改超参。默认最多6个GPU任务并发。

```bash
python -m analysis.mappo_b1_contract check-train --batch-root "$BATCH_ROOT" --id 0 --attempt a01
python -m analysis.mappo_b1_contract check-eval --batch-root "$BATCH_ROOT" --id 0 --attempt a01
```

训练验收读取真实最终payload的episode2500、config绑定、250000步、500条有限update、记录的actor Adam步数，并核查训练数组shape/有限值。评估验收核对来源policy、episode2500、协议及commit，读取所需8个NPZ字段，检查shape `(6,100,100,5)`（global少agent轴）、post-step复位一致、跨episode逐槽递推、CAM终点二值。

## 失败attempt

默认attempt=a01，训练在 `training/a01/runs/<run>`，评估在 `evaluations/<world_set>/a01/<run>/<eval_id>`。批次、attempt、完成记录均拒绝覆盖。失败保留原目录、started记录及Slurm日志；确需从头重试时只用新attempt（例如 `AOI_ATTEMPT=a02`）并设置 `AOI_RETRY_REASON` 记录具体技术原因，array仅包括失败ID。成功记录已存在的ID不得再次执行。

若原生COMPLETE/EVAL_COMPLETE已写出而B1验收记录缺失，入口阻止重训练/重评估。先核查根因，使用对应 `check-train`/`check-eval` 加 `--record` 对原attempt验收并补B1记录。只有完整校验通过才能写记录；若仍失败则回报，不以再次运行代替诊断。后续汇总从成功记录读取实际attempt，不将失败attempt重复计数。

## 汇总及交付

`analysis.summarize_mappo_b1_budget analyze --batch-root "$BATCH_ROOT" --attempt a01` 由CPU sbatch运行；输出 `analysis/a01`，存在即拒绝。失败保留现场；技术修复后的重新汇总使用新的分析attempt。

四贡献沿用已核实的worst-platoon-A表核算函数（独立拷贝在 `analysis/mappo_b1_accounting.py`，不写正式G表）。每臂每预算先跨world按agent合并整数sum/count，再各自选最大AoI；CAM另取最小。完整并列集合用有理数比较，取最小AoI并列索引用于单个分解并保留并列范围。完整周期半开 `[u_j,u_(j+1))`，S100封顶面积；le50/mid50/gt100/boundary分别保留，long_complete仅mid50+gt100。无复位/单复位均保留，world不连接。

各集合独立输出50k及250k的720行world-agent、720行流核算、120行跨world agent、24行own-worst端点，以及新频数表。正式比较C-a=shared50k−independent250k、C-b=shared250k−independent250k、C-c=independent250k−independent50k。n=6、df=5、描述性95% t区间及正/负/精确零数；CAM差单位百分点。组均、功率、payload、reward和AoI尾部辅助指标保留。曲线小表合计72000 episode（旧12000+新60000）、14400 update（旧2400+新12000）；不将训练episode最差曲线称固定评估学习曲线。

`verification.json` 分列 `technical_status` 与 `claim_support=NOT_ASSESSED`。技术PASS与收益符号无关；脚本不判定等效、追上、消失或机制。预期新计数24训练、48eval、288策略×world、14400000 scored agent-slots、144000 CAM终点、48端点/1440 world-agent/240 pooled-agent。

确认sacct和全部技术记录后：

```bash
python -m analysis.summarize_mappo_b1_budget package --batch-root "$BATCH_ROOT" --attempt a01
```

在HPC需GNU tar及zstd。精简包 `$BATCH_ROOT/core_evidence.tar.zst` 仅纳入批次protocol/config/record JSON、CSV、报告，成功run的config/provenance/COMPLETE和评估元数据。NPZ、policy、完整轨迹、缓存与完整日志留HPC；不生成SHA256清单。sacct报告请保存到protocol目录的CSV供打包。包已有则拒绝覆盖。任何失败仅报具体缺口，不自动启动Tier2、T2或其他实验。
