# CAM96k：seed8–13 选每方法一个best100，在30个world评估

当前组合入口为 `analysis.cam96k_selected6`，schema=`cam96k_selected_seed8_13_v1`。
本协议替代此前“seed8–11各自评估5个world”的后续计划。历史批次、旧评估和旧manifest均保留原样。

## 训练和保存

新增训练仍为5方法 × seed8/9/10/11 =20次。复用上一版已验证的完整训练与恢复实现，
在组合批次独立的`training_batch`中运行，ID、训练参数、预算、两路并发和组屏障均不变：
Shared MAPPO-TDec → Independent MAPPO-TDec → Shared MAPPO-Combined → Independent Modified MADDPG-TDec → Shared Modified MADDPG-TDec。
每方法先seed8/9、再seed10/11，四个seed均验收后进入下一方法。

各run从头训练2500周期、250000环境步，CAM96000，P5/N4/K3/gap25、100时隙/周期、1ms/时隙。
20次新增训练共50000周期、5000000环境步；旧seed12/13的10次训练只读复用。
仍保留best100、final、每500周期的完整训练状态和独立周期策略，两个周期集合均为500/1000/1500/2000/2500。
新训练应验收100份完整训练检查点、100份周期策略、20份best100和20份final。
完整恢复内容、原子落盘、RNG、回滚归档及断点命令说明见[上一版恢复说明](RUN_CAM96K_SEED8_11.md)。

组合入口委托的子批次使用原训练schema，保留该schema的旧逐seed评估字段用于兼容。
本次只执行其训练及训练验收；组合入口的preview/status/评估调度采用本文件的150单元契约。
Grok不得调用子批次的evaluate或summary。所有正式冻结命令都通过`analysis.cam96k_selected6`执行。

## 先锁定策略，再评估

每方法共6个候选，即训练seed8、9、10、11、12、13各自的`policy_best_training.pt`。
先验收全部20次新训练及10次历史训练，再从原始训练指标重新计算各run的逐周期rolling100，
核对其best100分数、最早最优周期、选择表、权重SHA256、配置和原始Git身份。
同方法的6个best100中，选择**rolling100训练reward最高**者。精确同分按seed较小者确定。
单run原有best100规则仍为逐周期rolling100、严格改善、最早并列，MAPPO更新前保存时序不变。
不比较最后一周期reward，不选final或中间周期快照，不使用任何评估结果选seed。

完整30候选及5个胜者写入不可覆盖的`records/selection.json`。
选定后每方法在全部30个world固定使用同一seed、同一best100 SHA256。
候选缺失、损坏、配置漂移、分数与训练历史不符、或选定后输入发生变化，直接拒绝推进。
历史训练提交和当前评估提交分别记录，不将seed12/13训练身份改为当前Git提交。

## 150个冻结评估单元

5方法 × 每方法1个选定checkpoint × worlds1001–1030 × repeat0 =150单元。
一个评估监督进程，默认8路，独立于训练上限2；各单元独占目录、日志及锁，另有8个独立评估槽。
每world预热5周期、计分100周期、100时隙/周期。MAPPO保持随机采样repeat0，MADDPG冻结动作noise0。
保留原cuda:0、数值精度、动作解码和RNG派生，不学习、不构造训练replay或训练优化器。
MADDPG训练fixed sigma0.3及target smoothing sigma0.2/clip0.5继续保持，评估noise0与训练噪声是不同参数。
worlds1001–1030标为探索集；历史selected150已经使用过这些world，不宣称它们为未见确认集。
旧Uniform及其他历史评估保留，本轮不扩展或重跑Random/Uniform。

统计输出`world_platoon.csv`（750行）、`world_summary.csv`（150行）及`selected_policy_summary.csv`（5行）。
保留CAM分子分母、mean/worst AoI、线性mW、剩余bit、payload交付比例、失败剩余bit、reward分量。
每方法候选训练seed数为6，**实际评估训练seed数为1**；30个world不能计为30个训练seed。
不生成六训练seed平均或跨训练seed标准差，结果应表述为“每方法从六seed中选定的策略表现”。

正式最终计数为：candidate runs=30、selected policies=5、eval units=150、platoon rows=750、
CAM deadlines=75000、scored agent slots=7500000、candidate training seeds/method=6、evaluated training seeds/method=1、smoke=false。

## 运行入口

新交付prompt给出已初始化并绑定最终SHA的组合批次路径。所有长任务由Grok启动，本轮开发不启动。

```powershell
$repo = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\AoI-Reproduction0804'
$python = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\.envs\aoi_cuda310\python.exe'
Set-Location -LiteralPath $repo
# $batch使用交付prompt中的组合批次，不使用旧批次或其training_batch替代。
& $python -B -m analysis.cam96k_selected6 preview --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preview failed' }
& $python -B -m analysis.cam96k_selected6 preflight --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'preflight failed' }
& $python -B -u -m analysis.cam96k_selected6 train --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'training or 30-candidate selection failed' }
& $python -B -m analysis.cam96k_selected6 select --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'selection gate failed; evaluation forbidden' }
& $python -B -u -m analysis.cam96k_selected6 evaluate --batch-root $batch --eval-workers 8
if ($LASTEXITCODE -ne 0) { throw '150-unit evaluation failed' }
& $python -B -m analysis.cam96k_selected6 summary --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'summary failed' }
```

后台wrapper为`scripts/run_cam96k_selected6_supervisor.ps1`，train结束后验收30候选并封存选择，
evaluate结束后执行150单元summary。检查唯一receipt中exit_code及validation_exit_code均为0，
receipt缺失不能视为成功，保留原进程锁、单元锁、技术失败attempt、新launch身份和历史身份保护。

训练恢复使用同一组合入口`train --resume-incomplete --reason '真实技术原因'`，或
`train-unit --id N --attempt a01 --resume-incomplete --resume-episode 500 --reason '真实技术原因'`。
代码不会自动清理残留锁；须先核对PID、创建时间、命令行，确认持有者退出后归档本批次锁。
评估技术失败保留原attempt，使用`eval-unit --id N --attempt a02 --reason '真实技术原因'`；已接受结果复核后跳过。
不要因reward较低重训，不自动OOM重试或降并发，不以新checkpoint替换已经封存的胜者。

## 开发验证范围

新测试覆盖六seed完整性、全部可能胜者、同分、错误角色/非有限分数、封存及损坏拒绝、150矩阵、
20训练委托不变、训练/评估槽独立、750/75000/7500000计数以及不伪造六seed平均。
开发评估smoke明确使用已存在的seed12/13候选子集、真实best100、短计分预算，验证5方法原生冻结主路径、
八路混合并发及串行逐数组/RNG一致性；它不代表已完成seed8–13正式选模，也不计入150正式单元。
