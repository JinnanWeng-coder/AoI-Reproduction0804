# E3 预锁定新世界确认（非 W2）

本协议固定既有24个 final-500 MAPPO policy：independent/shared × combined/TDec 两种价值学习配置 × training seed 8–13。修订后的候选新世界按固定顺序 **303、304、305、306、307、308**。它们不是新的训练 seed，且尚待 HPC 对完整历史执行记录查重，不能仅凭本地精简证据宣布未使用。新世界只在 `lock.json` 成功写入后生效；锁定前禁止 pilot 或任何策略评估。213–218 永远是开发世界，不混入 E3。

v1 候选 301–306 的锁定前审计被阻止：301/302 出现在全部源训练配置的 `selection_validation_seeds` 中。虽然 Grok 未找到两者的实际冻结评估，仍在任何 pilot 前按更严格边界修订为 303–308；并非按效果选世界。旧 BLOCKED 审计须在 E3 根保留为 `world_use_audit_previous_<旧commit前7位>.json`。本地可见的结构化 `eval_seed` CSV 仅记录 213–218，本地没有完整 HPC 评估元数据/Slurm 记录。新名单的历史清白仍须 Grok 在 HPC 重新核查，若冲突/无法核实，停止，不再换名单。机器合同为 `hpc/e3_locked_test_protocol.json`，列明24个来源、世界、协议、指标、权重和输出位置。`lock.json` 在 HPC 记录实际代码 commit、UTC 锁定时间、审阅声明、来源清单与合同快照。不能从 pilot 效果反向更改这些字段。

## 输入和只读边界

默认仓库 `/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804`；研究根 `/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study`；唯一新输出根 `$STUDY_ROOT/E3-locked-test/P5_N4_gap25`。independent 两配置 policy 来自 `existing-evidence/tdec-ab-v1/P5_N4_gap25/training/runs`；shared-TDec 来自 `existing-evidence/shared-actor-v1/P5_N4_gap25/training/runs`；shared-combined 来自 `E5-sharing-critic/P5_N4_gap25/training/runs`。来源 config/COMPLETE/policy、旧评估和 MAPPO_results 均只读。

评估复用 `analysis.evaluate_mappo_feasibility_reward` 的 baseline 入口及原有 payload 验证、RNG 规则：每策略/世界 `SeedSequence([training_seed,world,0x4D415050])`；每世界冷启动一次，stochastic、无外部噪声或功率干预、sequential_warm、5 warmup＋100 scored ×100 slots、5 agents。保留运行归一化/策略参数冻结；scored NPZ 不二次删 warmup。

## 远端顺序与安全停机

1. 安全 fetch、快进到交付 commit，确认 `research/mappo` checkout 干净。检查 E3 输出根已有内容；已有 `lock.json` 或 evaluations 时先核对，不重锁、不覆盖完成运行。
2. 本次已有旧 BLOCKED 审计，使用 `python -m analysis.mappo_e3_contract history-audit --study-root "$STUDY_ROOT" --supersede-blocked-audit`；入口仅在未锁定、未评估且旧报告确为另一候选名单的 BLOCKED 时，先保存旧报告，再生成新 `world_use_audit.json`。扫描根现包括 `actor-sharing-study` 与诊断树 `MAPPO_results`，读取 JSON 世界字段、CSV 世界列、正式 evaluation 目录和显式 Slurm 世界记录。若在全新 E3 根运行，省略 `--supersede-blocked-audit`。Grok 另须核对完整历史 EVAL_COMPLETE/provenance、运行目录、Slurm 执行记录和其他研究根，区别计划文本与实际运行。自动扫描 clear 不是人工审阅的替代。若世界冲突、元数据缺口或无法确认，停在此步，不先 pilot、不静默换号。
3. `python -m analysis.mappo_e3_contract source-preflight --study-root "$STUDY_ROOT"`：在 HPC 逐项加载24个 policy payload，仅用于验证来源/config/COMPLETE/final-500、actor sharing 兼容；不实例化环境或运行策略。通过后、且历史审阅无冲突，执行 `python -m analysis.mappo_e3_contract lock --study-root "$STUDY_ROOT" --confirm-reviewed`。此旗标是 Grok 对完整历史执行记录的人工审阅声明。`check-lock` 可随时复核 commit、世界顺序和合同。
4. `bash -n` 两个 sbatch 后，先提交 `--array=0,6,12,18%4` 四个完整 cell pilot，每个含全部六世界。技术校验通过后提交 `--array=1-5,7-11,13-17,19-23%6` 余下20 cell。pilot 是正式24 cell的组成部分，不能重复或剔除。完整且合同一致的 E3_COMPLETE 由 `check-cell` 验证后跳过；不完整/冲突目录必须保留并报告，禁止静默覆盖或从多次结果择优。技术失败记录作业ID、错误、目录状态；不得按效果决定续跑。
5. 24个 cell 均 `check-cell` 通过后，提交 CPU `hpc/aoi_mappo_e3_locked_analyze.sbatch`。技术验收不以共享效应方向决定。分析结果仅写 E3 根的 `analysis/`；现有来源不回写。

## 统计与证据

C1 主指标为每seed的 worst-agent mean AoI（较低有利）和 worst-agent binary CAM（较高有利）。先按同 agent 汇总六世界，再取五 agent 的 max/min，不先逐世界取最差。CAM是每scored episode终点事件，总计72,000 agent-episode，不以7,200,000 agent-slots作分母。辅助报告 mean AoI、AoI>50、at-cap、mean/worst payload、mean CAM、combined reward、线性mW功率。保留逐world×agent精确 sum/count。

C2从同一批NPZ复用 W1 的 reset/完整间隔函数：world内连接scored episodes、不跨world、不筛V2V；L=相邻复位索引差，严格 L>100 为主阈值；报告原始数、每万agent-slots频度、完整间隔事件占比、流等权超阈值比例、reset rate、两种均值及两种完整CCDF。全条件统一网格预定为 `{0,5,50,100} ∪ 所有观测完整L`，并保留原始频数；无复位/单复位流保留、统计NA与有效分母显式列明。六个training seeds先各自归约，再报告条件mean、sample SD、描述性95% t区间及同seed shared−independent差和符号数；不把world/agent/interval当独立训练重复。

预期规模：0新训练，24新评估cell，144策略×世界运行，720条流，7,200,000 scored agent-slots，72,000 CAM截止期事件，24条condition×seed摘要。完整间隔有效流数不预设。`analysis/verification.json` technical PASS 仅表示来源、协议、完整性和计算正确，不表示C1/C2得到新世界支持。

精简包 `E3-locked-test/P5_N4_gap25/core_evidence.tar.zst` 仅纳入 `hpc/e3_locked_test_protocol.json` 的副本（打包时从仓库以 `tar -C` 添加）、E3根 `lock.json`、新旧 `world_use_audit*.json`、`source_inventory.json`、24个 `E3_COMPLETE.json`、`analysis/*.csv`、`analysis/verification.json`、`analysis/index_examples.json`、`analysis/fields_and_boundaries.md`、`analysis/report_cn.md`。不要打包NPZ、policy、checkpoint、完整轨迹、缓存、完整日志或SHA256清单。训练commit与评估commit分列，不互相冒充。
