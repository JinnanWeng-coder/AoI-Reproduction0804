# W1 既有冻结评估的 V2I 服务间隔后处理

本入口只读已有轨迹，不加载 checkpoint/policy，不实例化环境，也不调用 evaluator。0 新训练、0 新评估。combined/TDec 是两种价值学习配置，不简化为仅 critic 网络结构差异。范围固定为 independent/shared × combined/TDec、训练 seed 8–13、开发世界 213–218。输入为 24 个评估 cell、144 个策略×世界运行、720 条 world×agent 流。

## HPC 路径和输入

默认项目 checkout：`/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804`。研究根：`/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study`。输出仅在研究根下 `W1-service-intervals/P5_N4_gap25`。该路径以外的 existing-evidence、E5、N1、MAPPO_results、源 NPZ/元数据均只读。

combined 两臂开发评估在 `E5-sharing-critic/P5_N4_gap25/evaluations`；TDec 两臂开发评估在 `existing-evidence/shared-actor-v1/P5_N4_gap25/evaluations`。独立 TDec 的训练 config 在 `existing-evidence/tdec-ab-v1/P5_N4_gap25/training/runs`，与开发评估根不同。TDec 历史对账表在 `existing-evidence/zero-shot-v1/analysis/service_regularity`，不算新增轨迹。

协议要求 policy_final episode 500、P5/N4/gap25、n_rb=3、stochastic/baseline/sequential_warm、每世界冷启动一次、5 warmup + 100 scored episodes、100 slots/episode、5 agents。源 `metrics.npz` 中 `scored_episode` 轴已排除 warmup，形状 `(6,100,100,5)`，不得再删前五集。event 优先 `reset_event`；如缺失，只有逐槽 post-step `aoi_ms≈1` 才可替代，并记录 event_source。两者都有时必须逐元素一致。只从 NPZ 解压所需事件字段。

## 执行（由 Grok 在 HPC 操作）

先安全同步 Git，确认 `research/mappo` checkout 干净且 HEAD 等于交付 commit；若 HPC 有本地改动，停止并报告，不能 reset/强推。创建输出、日志、tmp、cache 后，先 preflight：

```bash
PROJECT_DIR=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804
STUDY_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study
RESULT_ROOT="$STUDY_ROOT/W1-service-intervals/P5_N4_gap25"
PYTHON_BIN=/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python
mkdir -p "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache/preflight"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache/preflight"
"$PYTHON_BIN" -m analysis.audit_mappo_w1_service_intervals --study-root "$STUDY_ROOT" --preflight-only
```

检查 `preflight.json` 为 PASS、`source_inventory.csv` 恰 24 个无重复源、四条件各六 seed、世界顺序与数组形状/事件来源正确。`bash -n hpc/aoi_mappo_w1_analyze.sbatch`，然后以 `AOI_EXPECTED_COMMIT=<交付完整SHA> sbatch hpc/aoi_mappo_w1_analyze.sbatch` 提交单个 CPU 作业；无 GPU/array。只有 preflight 通过才提交。脚本会检查分支、提交、干净工作树与输出根。

## 统计与验收

每个 world×agent 展平连续 scored episodes，绝不跨 world、不筛 V2V。完整间隔 `L=diff(reset indices)`；首末等待仅是观察窗边界。无复位流只记一个 M=10000 的唯一可见观察窗，单次复位无完整间隔，未定义的均值/CV² 为 CSV 空值。AoI cap 不截断 L，阈值固定严格 `L>100`。

输出根中的 `fields_and_boundaries.md` 给出两种权重的 CCDF 精确定义。事件加权按完整间隔频数合并；流等权先算有完整间隔流的分布再平均，覆盖数显式报告。`flow_interval_frequency.csv` 可精确重建 CCDF；`seed_ccdf.csv` 含阈值 0、100 和全部观测 L 的共同阈值网格，另有跨 seed 条件 CCDF 摘要与逐 seed 共享差及其摘要。先形成 24 个条件×seed 结果，再用六个 training seed 求 mean、sample SD、描述性 t 区间与同 seed 的 shared−independent 差。不得把流/间隔当独立训练重复，不按 screen_success 筛选，不做星号或因果分解。

验收 `verification.json` 须是 technical PASS，包含 24 cell、144 policy-world、720 流、7,200,000 agent-slots、24 seed 行、频数矩/边界/CCDF 检查与 TDec/dev 旧表对账 PASS。它不表示共享效果优于独立。历史 pooled 约 4.004→4.923，而 equal-flow 约 8.377→6.337，用来识别权重混用；新程序不硬编码这些结果。源缺失、协议或事件语义冲突时停止，报告最小缺口，不补跑 evaluator。作业退出码、`slurm_logs`、全部 CSV/JSON/报告文件和 Git 工作树都需检查。

## 精简证据包

只在 W1 输出根打包：`source_inventory.csv`、`preflight.json`、`flow_statistics.csv`、`flow_interval_frequency.csv`、`seed_statistics.csv`、`seed_ccdf.csv`、`condition_summary.csv`、`condition_ccdf_summary.csv`、`sharing_effect_per_seed.csv`、`sharing_effect_summary.csv`、`sharing_ccdf_effect_per_seed.csv`、`sharing_ccdf_effect_summary.csv`、`verification.json`、`index_examples.json`、`report_cn.md`、`fields_and_boundaries.md`。不含 NPZ、policy、checkpoint、完整轨迹、cache、slurm 日志，不生成 SHA256 清单。

```bash
cd "$RESULT_ROOT"
tar -I zstd -cf core_evidence.tar.zst source_inventory.csv preflight.json flow_statistics.csv flow_interval_frequency.csv seed_statistics.csv seed_ccdf.csv condition_summary.csv condition_ccdf_summary.csv sharing_effect_per_seed.csv sharing_effect_summary.csv sharing_ccdf_effect_per_seed.csv sharing_ccdf_effect_summary.csv verification.json index_examples.json report_cn.md fields_and_boundaries.md
tar -I zstd -tf core_evidence.tar.zst
```
