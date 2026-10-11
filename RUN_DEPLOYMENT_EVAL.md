# 冻结部署表现与稳定性：D1 运行说明

本轮复用 24 个历史 50k 模型（18 MAPPO、6 Modified MADDPG-TDec），新增训练为 0。
实现基准为 `09c38b48e98bae66f2c4ade3e3dee2f7b5d44a54`；运行 commit 必须采用本轮交付回复中的实际提交。
算法、环境、reward、学习器和默认超参保持基准版本。所有正式运行在服务器完成。

| 阶段 | 训练 seed | world | 执行 | policy-world-repeat | 累积量行数 | scored agent-slots |
|---|---|---|---|---:|---:|---:|
| pilot | 8、9 | 213、214 | 4 随机条件各 3 repeats；MADDPG noise0 一次；额外两臂 TDec MAPPO deterministic 各一次 | 60 | 300 | 3,000,000 |
| main | 8–13 | 30 个历史排除后锁定 world | 五主条件，4 随机条件各 3 repeats，MADDPG noise0 一次 | 2340 | 11700 | 117,000,000 |
| gap-transfer | 8–13 | 213–218 | gap25 训练 Algorithm 2，noise0/0.3，目标 gap5/35，repeat0 | 144 | 720 | 7,200,000 |

每个单元一个 world，一次冷启动，5 warmup + 100 scored episodes，每 episode 100 slots、5 platoons。
保存的 NPZ 仅含 scored，形状 `(1,100,100,5)`；不要再丢弃头五个 episode。
main 不包括 pilot 的额外 deterministic MAPPO 条件。pilot 与 main 使用不同 world，不混合推断。

## 1. 安全同步及固定执行 checkout

以下命令只供 HPC Grok 在服务器执行。先核对 `FETCH_HEAD` 等于交付的完整 commit；有其他工作或远端新提交时保留现场。
使用独立 detached execution worktree，后续不要 fetch/merge/pull 或更新这个 worktree。

```bash
set -euo pipefail
export BASE=/eeedata/sgxjw2/Parvini-TVT2023-reproduction
export PYTHON_BIN=/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python
export DIAGNOSTICS_ROOT="$BASE/AoI-Reproduction-diagnostics"
export EVIDENCE_PACK="$BASE/tmp/evidence_pack_v0_20261011.zip"
export METADATA_ADDENDUM="$BASE/tmp/metadata_addendum_v1.zip"
cd "$BASE/AoI-Reproduction0804"
test "$(git branch --show-current)" = research/mappo
test -z "$(git status --porcelain --untracked-files=all)"
git fetch origin research/mappo
export AOI_EXPECTED_COMMIT="$(git rev-parse FETCH_HEAD)"
# 此处将 AOI_EXPECTED_COMMIT 与本轮交付回复中的完整 SHA 核对，不一致则停止。
git merge-base --is-ancestor HEAD "$AOI_EXPECTED_COMMIT"
git merge --ff-only "$AOI_EXPECTED_COMMIT"
export PROJECT_DIR="$BASE/deployment-checkouts/$AOI_EXPECTED_COMMIT"
mkdir -p "$BASE/deployment-checkouts"
test ! -e "$PROJECT_DIR"
git worktree add --detach "$PROJECT_DIR" "$AOI_EXPECTED_COMMIT"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
git diff --exit-code 09c38b48e98bae66f2c4ade3e3dee2f7b5d44a54 HEAD -- aoi_v2x_reproduction
for f in hpc/deployment_common.sh hpc/aoi_deployment_{pilot,main,gap_transfer,summary}.sbatch; do bash -n "$f"; done
"$PYTHON_BIN" -B -c 'import torch,numpy,matplotlib; print(torch.__version__,numpy.__version__,matplotlib.__version__)'
command -v sbatch
command -v sacct
command -v tar
command -v zstd
test -f "$EVIDENCE_PACK"
test -f "$METADATA_ADDENDUM"
```

已有同 SHA execution worktree 可以核对 commit/clean 后复用。Python、分区、账户、GPU型号、CPU数、内存、时限、并发和项目/诊断路径均可配置。
本地没有提交 Slurm 作业；`analysis.submit_deployment` 只有带 `--submit` 才实际调用 sbatch。

## 2. 新批次、来源 preflight、锁定 main world

```bash
export BATCH_ID="deploy_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
export AOI_BATCH_ROOT="$DIAGNOSTICS_ROOT/actor-sharing-study/D1-frozen-deployment/P5_N4_gap25/$BATCH_ID"
"$PYTHON_BIN" -B -m analysis.deployment_contract init \
  --diagnostics-root "$DIAGNOSTICS_ROOT" --batch-id "$BATCH_ID" \
  --evidence-pack "$EVIDENCE_PACK" --metadata-addendum "$METADATA_ADDENDUM"
export TMPDIR="$AOI_BATCH_ROOT/tmp" XDG_CACHE_HOME="$AOI_BATCH_ROOT/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
"$PYTHON_BIN" -B -m analysis.deployment_contract preflight --batch-root "$AOI_BATCH_ROOT" --load-payload
"$PYTHON_BIN" -B -m analysis.summarize_frozen_deployment evidence-regression --evidence-pack "$EVIDENCE_PACK"
```

`protocol/manifest.json` 固定实现 SHA、协议、条件、来源及输入包；`sources.json` 固定 24 个 config、COMPLETE、实际模型路径、原训练 commit、config hash、文件 size/mtime。preflight 加载和核对 payload，不实例化环境、不 rollout。
MAPPO 独立 TDec 的训练根为 `existing-evidence/tdec-ab-v1`，不是其 dev 评估目录 `shared-actor-v1`。
默认训练来源：shared TDec=`existing-evidence/shared-actor-v1/P5_N4_gap25/training/runs`；independent TDec=`existing-evidence/tdec-ab-v1/.../training/runs`；shared Combined=`E5-sharing-critic/.../training/runs`。
MADDPG 通过 E1 resolver 和历史 run_manifest 解析 canonical/historical TDec run，实际加载 `checkpoints/latest.pt`。
初版 evidence inventory 的 MADDPG `policy_path` 可能是 periodic actor 文件；这里按 run 识别，并按 addendum/E1 核实 latest500，禁止改用 periodic/best。

如服务器来源迁移，可给 init 追加 `--models-json /绝对路径/models.json`。
格式为 24 个对象的数组，每项含 `model_id`（shared_tdec / independent_tdec / shared_combined / algorithm2）、`training_seed`（8–13）、`checkpoint`（绝对路径）。检查仍执行，不允许改模型预算或算法身份。
manifest 中的单元明确指定模型、目标场景、world、repeat、执行方式和动作种子；array ID 仅索引这份固定清单。

锁定前人工检查诊断树之外的实际运行目录、评估目录及 Slurm 记录；如果还有研究根，用 `--scan-root` 补充。脚本默认扫描整个诊断树和执行 checkout 的 hpc 目录，解析 JSON、CSV、评估命名和日志；扫描失败或命中候选会 BLOCK。
`selection_validation_seeds` 也排除，不能因“仅预留”直接使用。

```bash
# 先完成人工历史执行记录检查，再执行。编号规则只依赖历史是否占用。
"$PYTHON_BIN" -B -m analysis.deployment_contract lock-worlds \
  --batch-root "$AOI_BATCH_ROOT" --candidate-start 1001 --confirm-reviewed
"$PYTHON_BIN" -B -c 'import json,os; from pathlib import Path; p=Path(os.environ["AOI_BATCH_ROOT"]); print(json.loads((p/"protocol/world_lock.json").read_text()))'
```

默认从 1001 起升序取未占用的前 30 个，写入 world_lock/main.json 后不可改。
可用 `--candidate-worlds 逗号分隔的30个编号` 指定预先确定候选，仍做全量历史排除。
BLOCK 时保留审计，不运行 main；技术纠正后用新的 `--audit-name world_use_audit_02`，不覆盖原审计。
这里锁定的是本轮新世界；旧 E3 303–308、dev 213–218 和训练/预留世界均排除。

## 3. Pilot

`--submit` 前先读 dry-run 输出、核对 60 单元、资源与路径。GPU default=Q10/l20、4CPU、16G、2h，允许按实际节点修改。
如果不用 GPU，传 `--gres '' --device cpu`。有账户要求时追加 `--account 实际账户`。

```bash
"$PYTHON_BIN" -B -m analysis.submit_deployment --batch-root "$AOI_BATCH_ROOT" --phase pilot --concurrency 6
export PILOT_JOBS="$("$PYTHON_BIN" -B -m analysis.submit_deployment \
  --batch-root "$AOI_BATCH_ROOT" --phase pilot --concurrency 6 --submit \
  | "$PYTHON_BIN" -c 'import json,sys; print(":".join(j["job_id"] for j in json.load(sys.stdin)["jobs"]))')"
export AOI_PHASE=pilot AOI_EVIDENCE_PACK="$EVIDENCE_PACK" AOI_ANALYSIS_ATTEMPT=a01
export PILOT_SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:$PILOT_JOBS" \
  --export=ALL --partition=Q10 --cpus-per-task=4 --mem=16G --time=03:00:00 \
  --output="$AOI_BATCH_ROOT/slurm_logs/%x_%j.out" --error="$AOI_BATCH_ROOT/slurm_logs/%x_%j.err" \
  hpc/aoi_deployment_summary.sbatch)"
```

等 pilot 和 CPU summary 结束。用 `sacct -j "${PILOT_JOBS//:/,},${PILOT_SUMMARY_JOB%%;*}" --format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,ReqMem` 检查 COMPLETED/0:0。
核对 `analysis/pilot/a01/verification.json` 的 technical_status=PASS、60单元、300行、3m agent-slots、30000 CAM deadline；核对重复间外生探针一致和 repeat0 桥接表。
桥接失败须查看设备/数值差异并如实记录，不调容差、不改种子、不把 gap5/35 与旧 gap25 比。
真实 checkpoint 加载、实际 GPU 采样、完整数组与冻结检查由 Grok 在本阶段完成；本地仅测试双对象与合成数据。
用 pilot sacct 的 elapsed/MaxRSS 及 `du -sb "$AOI_BATCH_ROOT/evaluations/pilot"` 估算 main 总量和资源，记录到新 `protocol/resource_calibration.json`。资源调整不改变训练、动作、协议或样本数；继续不取决于哪个算法较好。

## 4. Main

提交器要求已锁定 30 world 且本批次 pilot 有 technical PASS。
为适配 MaxArraySize=1001，main 分三组：local ID 0–999 + offset0；0–999 + offset1000；0–339 + offset2000。
三组 afterok 串联，所以默认并发 6 是整个 main 批次上限，不是三倍并发。一次 unit 只运行一个 world/repeat。

```bash
# 以下时限/内存应先按 pilot 实测校准。
"$PYTHON_BIN" -B -m analysis.submit_deployment --batch-root "$AOI_BATCH_ROOT" --phase main --concurrency 6 --mem 16G --time 02:00:00
export MAIN_JOBS="$("$PYTHON_BIN" -B -m analysis.submit_deployment \
  --batch-root "$AOI_BATCH_ROOT" --phase main --concurrency 6 --mem 16G --time 02:00:00 --submit \
  | "$PYTHON_BIN" -c 'import json,sys; print(":".join(j["job_id"] for j in json.load(sys.stdin)["jobs"]))')"
export AOI_PHASE=main AOI_ANALYSIS_ATTEMPT=a01
export MAIN_SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:$MAIN_JOBS" --export=ALL \
  --output="$AOI_BATCH_ROOT/slurm_logs/%x_%j.out" --error="$AOI_BATCH_ROOT/slurm_logs/%x_%j.err" \
  hpc/aoi_deployment_summary.sbatch)"
```

检查所有 array task 和 CPU summary 的 sacct。正式验收应为2340单元、11700行、117m scored agent-slots、1,170,000 CAM deadline。
六 seed 全部保留，每条件 30 world；世界/动作 repeat/episode/slot 不扩大独立训练 n。

## 5. Gap transfer

这批只运行已有 gap25 Algorithm 2 的 noise0/0.3、seed8–13、目标 gap5/35、213–218、repeat0。
可在 pilot 技术通过后执行。MAPPO gap 对照从 evidence pack 的既有24 cell/144 policy-world/720车队行复用；不产生新 MAPPO rollout。

```bash
export GAP_JOBS="$("$PYTHON_BIN" -B -m analysis.submit_deployment \
  --batch-root "$AOI_BATCH_ROOT" --phase gap-transfer --concurrency 6 --submit \
  | "$PYTHON_BIN" -c 'import json,sys; print(":".join(j["job_id"] for j in json.load(sys.stdin)["jobs"]))')"
export AOI_PHASE=gap-transfer AOI_ANALYSIS_ATTEMPT=a01 AOI_EVIDENCE_PACK="$EVIDENCE_PACK"
export GAP_SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:$GAP_JOBS" --export=ALL \
  --output="$AOI_BATCH_ROOT/slurm_logs/%x_%j.out" --error="$AOI_BATCH_ROOT/slurm_logs/%x_%j.err" \
  hpc/aoi_deployment_summary.sbatch)"
```

验收新增144单元、720行、7.2m scored agent-slots、72000 CAM deadline；复用MAPPO另列，不加到新增计数。
`gap_alignment_*.csv` 使用共同统计口径，对照标为 `data_role=reused`。本组是旧world观察性扩展，不能称新世界确认。

## 6. Summary / export 与失败处理

sbatch 的 CPU summary 调用以下确切命令；如果手动后处理，仍只在服务器执行。

```bash
"$PYTHON_BIN" -B -m analysis.summarize_frozen_deployment summary \
  --batch-root "$AOI_BATCH_ROOT" --phase main --attempt a02 --evidence-pack "$EVIDENCE_PACK"
# 只有需要新的后处理 attempt 时才使用上面的 a02；a01 输出禁止覆盖。
"$PYTHON_BIN" -B -m analysis.summarize_frozen_deployment export --batch-root "$AOI_BATCH_ROOT" --phase pilot --attempt a01
"$PYTHON_BIN" -B -m analysis.summarize_frozen_deployment export --batch-root "$AOI_BATCH_ROOT" --phase main --attempt a01
"$PYTHON_BIN" -B -m analysis.summarize_frozen_deployment export --batch-root "$AOI_BATCH_ROOT" --phase gap-transfer --attempt a01
tar -I zstd -tf "$AOI_BATCH_ROOT/analysis/main/a01/core_evidence.tar.zst"
```

每个阶段独立归档，`analysis/<phase>/<attempt>/core_evidence.tar.zst` 仅含 protocol、CSV/JSON/中文报告/字段说明/PNG图；模型、NPZ、完整轨迹、缓存、日志保留服务器，不制作 SHA256 清单。
主要表：world_repeat_platoon（明确sum/count）、world_platoon_repeat_pooled、world_endpoints、repeat_endpoints、seed_endpoints、legacy_seed_endpoints、condition_summary、completion_records、exogenous_verification；gap另有历史对齐表。
scene-first 先在 world/platoon 内汇总 repeat，再取自己的最大AoI/最小CAM，再对world均值，最后六seed描述统计；AoI与CAM可以选中不同车队。
world sample SD、训练seed sample SD、repeat sample SD分别报告。repeat SD含每次重选最弱车队和固定scene-first最弱车队两种定义；确定性单repeat为null。
六seed时描述性t95使用df5；pilot两seed不伪装为六seed。缺失值不补0，不作显著性筛选或自动主张判断。

一个 unit 仅在完整键、实际来源/文件身份、commit、协议、动作种子、原生完成、数组和累积量重新验收相符时跳过。
失败目录全部保留。存在原生 EVAL_COMPLETE 但缺少 DEPLOYMENT_COMPLETE 时，先尝试接纳已有轨迹：

```bash
"$PYTHON_BIN" -B -m analysis.evaluate_frozen_deployment \
  --batch-root "$AOI_BATCH_ROOT" --phase pilot --unit-id 0 --attempt a01 --accept-existing
# 无原生完成时可由 Grok 重试该失败单元，使用新 attempt 和真实技术原因：
export AOI_ATTEMPT=a02 AOI_RETRY_REASON='记录真实失败原因'
# 用原来的 phase sbatch，array仅列失败local ID，保留正确AOI_TASK_OFFSET；成功单元不会重跑。
```

已有原生完成但验收冲突时停止并回报，禁止覆盖/重新评估它来掩盖错误。实际代码修复须新交付commit和新批次；运行中不更新旧批次checkout。
后处理失败用新的 analysis attempt。缺模型、源数据或元数据时报告最小缺口，不训练、不自动补评估其他实验。
Grok回传：完整SHA、batch路径、固定world列表、所有JOB_ID、sacct资源/退出信息、各阶段verification、归档路径及任何未通过项。

## 7. 原 pilot 的 MADDPG profile 技术修复与只读复用

原提交 `3258d35b66dbc00f4a987a76159e8a7167cb6413` 的44个MAPPO单元已经完成；Algorithm 2的16个单元（36–51）在rollout前失败。
原因是历史`paper_faithful`配置误入MAPPO目标派生函数的active-training校验。
修复后的MADDPG目标派生保留原profile/semantic_version和全部有效配置，只改变目标scenario、device和is_formal_result，检查动作/观测/agent兼容性。
不把历史profile重命名成reproduction_baseline，不放宽config.py训练校验。
新preflight在不实例化环境的前提下检查24个源模型的gap5/25/35目标配置派生。

旧checkout、旧batch、44个完成标记及16个失败现场全部只读保留。先按第1节安全同步到本轮修复提交，创建新的固定SHA execution worktree；**不要更新旧checkout，也不要编辑旧manifest的commit**。
然后执行以下修复专用命令（仅HPC Grok执行）：

```bash
export OLD_BATCH="$DIAGNOSTICS_ROOT/actor-sharing-study/D1-frozen-deployment/P5_N4_gap25/deploy_20261011T081733_25999"
export EVIDENCE_PACK="$("$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; print(json.loads((Path(sys.argv[1])/"protocol/manifest.json").read_text())["input_packs"]["evidence_pack"]["path"])' "$OLD_BATCH")"
export METADATA_ADDENDUM="$("$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; print(json.loads((Path(sys.argv[1])/"protocol/manifest.json").read_text())["input_packs"]["metadata_addendum"]["path"])' "$OLD_BATCH")"
export BATCH_ID="deploy_repair_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
export AOI_BATCH_ROOT="$DIAGNOSTICS_ROOT/actor-sharing-study/D1-frozen-deployment/P5_N4_gap25/$BATCH_ID"
export TMPDIR="$AOI_BATCH_ROOT/tmp" XDG_CACHE_HOME="$AOI_BATCH_ROOT/cache" MPLCONFIGDIR="$AOI_BATCH_ROOT/cache/matplotlib"
export TORCH_HOME="$AOI_BATCH_ROOT/cache/torch" CUDA_CACHE_PATH="$AOI_BATCH_ROOT/cache/cuda"
"$PYTHON_BIN" -B -m analysis.deployment_contract init \
  --diagnostics-root "$DIAGNOSTICS_ROOT" --batch-id "$BATCH_ID" \
  --evidence-pack "$EVIDENCE_PACK" --metadata-addendum "$METADATA_ADDENDUM" --reuse-pilot "$OLD_BATCH"
"$PYTHON_BIN" -B -m analysis.deployment_contract preflight --batch-root "$AOI_BATCH_ROOT" --load-payload
"$PYTHON_BIN" -B -m analysis.summarize_frozen_deployment evidence-regression --evidence-pack "$EVIDENCE_PACK"
export AOI_DEVICE=cuda:0 AOI_ATTEMPT=a01 AOI_TASK_OFFSET=0
unset AOI_RETRY_REASON
export REPAIR_JOB="$(sbatch --parsable --array=36-51%6 --partition=Q10 --gres=gpu:l20:1 \
  --cpus-per-task=4 --mem=16G --time=02:00:00 --export=ALL \
  --output="$AOI_BATCH_ROOT/slurm_logs/%x_%A_%a.out" --error="$AOI_BATCH_ROOT/slurm_logs/%x_%A_%a.err" \
  hpc/aoi_deployment_pilot.sbatch)"
export AOI_PHASE=pilot AOI_ANALYSIS_ATTEMPT=a01 AOI_EVIDENCE_PACK="$EVIDENCE_PACK"
export REPAIR_SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:${REPAIR_JOB%%;*}" --export=ALL \
  --output="$AOI_BATCH_ROOT/slurm_logs/%x_%j.out" --error="$AOI_BATCH_ROOT/slurm_logs/%x_%j.err" \
  hpc/aoi_deployment_summary.sbatch)"
```

`init --reuse-pilot`限定原3258d35 pilot，要求全部24个来源及输入包身份一致、44个MAPPO完整键/数组/累积量重新验收通过、原MADDPG没有原生完成。
它检查MAPPO执行、reward记录、环境和动作RNG代码相对原commit未变，写新`protocol/reused_pilot.json`引用旧run，**不复制NPZ、不回写旧provenance**。
任一缺口阻止登记，不自动重跑44个MAPPO。新任务仅36–51，共16次policy-world-repeat。
汇总自动读取旧44+新16；逐流保留原评估commit，completion_records另列analysis implementation commit、evaluation_implementation_commit、data_role和reused_from_batch。
verification应为technical_status=PASS、policy_world_repeats=60、new_policy_world_repeats=16、reused_policy_world_repeats=44、300行、3m scored agent-slots、30000 CAM deadline。
使用sacct检查16个task和CPU summary；PASS后才按第6节export pilot。失败保留新旧现场并报告，不修改容差、seed或超参。
本次技术修复交接只恢复pilot并验收；不顺带lock-worlds、提交main/gap-transfer或新训练。
本地105项与服务器48项来自不同测试选择，不能写成服务器已通过105项；各自保留真实命令、数量及日志。

## 本地核验入口

```bash
python -B -m pytest tests/test_frozen_deployment.py -q
python -B -m analysis.deployment_contract dry-run --phase pilot
python -B -m analysis.deployment_contract dry-run --phase main --worlds 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030
python -B -m analysis.deployment_contract dry-run --phase gap-transfer
```

上面的 main 编号仅为 CLI 合成测试输入，不代表已完成服务器历史核查或锁定。
