# 冻结部署方法事实（供写作引用）

本文档描述已实现协议，不报告尚未运行的实验结论。

- 复用 P5/N4/K3/gap25、50k环境步、seed8–13的18个 MAPPO final-500 policy 和6个 Modified MADDPG-TDec（Algorithm 2）latest-500 checkpoint；同一Algorithm 2模型用于noise0和0.3，两者不是两批训练。
- 五个主执行条件：shared MAPPO-TDec stochastic；independent MAPPO-TDec stochastic；shared MAPPO-Combined stochastic；Algorithm 2 noise0；同一Algorithm 2 noise0.3。Combined/TDec称两种价值学习配置，不缩减为仅网络结构差异。
- 算法、环境、reward、网络、PPO更新、归一化和默认超参保持代码基准09c38b4；本轮新增训练为0。没有在线学习、权重适应或新baseline。
- MAPPO使用既有HybridActor：RB/mode为categorical，功率为Beta。stochastic沿用per-agent采样顺序；deterministic使用既有RB/mode argmax和Beta均值。共享actor仍对各车队本地观测单独采样动作，不复制一份动作给所有车队。
- MAPPO部署复用既有actor-only loader，无critic/optimizer/rollout构造。MADDPG复用E1 frozen evaluator和源choose_action；explore=False，显式noise_std=0或0.3，在归一化动作空间加高斯噪声并沿用源动作clip。MADDPG沿用E1的系统加载与完整参数/optimizer状态快照核对，但不调用学习更新。
- 每个world-repeat独立冷启动一次，world_seed=world；依次5 warmup+100 scored episodes，每episode100个1ms槽、5车队；AoI、前槽干扰等world状态连续跨episode，CAM需求/deadline在episode起点重置。
- 环境使用PaperEnviron私有NumPy generator；动作采样和噪声与其分离。repeat0：MAPPO Torch seed由SeedSequence([training_seed,world,0x4D415050]).generate_state(1,uint32)生成；MADDPG用default_rng(SeedSequence([training_seed,world,0xA01]))。repeat>0在上述words后追加[0x44504C59,repeat]。动作执行方式不进入环境seed。
- 同一world改变动作repeat或噪声时，保持外生生成过程；闭环观测和业务状态不要求相同。验收精确比较初始/最终环境RNG、车辆/信道探针，源码核对外生随机调用不依赖动作。设备差异可能影响repeat0历史轨迹桥接，桥接表独立保留，不通过换种子或改容差消除差异。
- 原始数组只含scored；逐模型/执行/目标/world/repeat/platoon输出sum/count。AoI为逐槽post-step封顶值，严格AoI>50；CAM是每episode最后一槽success，和payload完成比例分开。功率逐槽dBm→mW后累计。global reward在车队账本中重复标识同一world标量，不能当五次独立reward。
- reward_combined=task1+task2+源global_actor_weight×global。账本包含剩余CAM需求项、AoI项、V2I收入项和功率罚项；复用现有环境定义并验证闭合。V2I attempt来自mode==0；reset来自经源码核验的post-step AoI≈1，连续复位保留；无可信来源则留空。
- scene-first：同一world/platoon先合并repeats的sum/count，然后AoI取最大、CAM取最小；两个最弱车队可以不同。再跨world汇总，每训练seed一行，最后跨seed。legacy：每车队先跨world/repeat合并，再取各自极值。两口径同时保留。
- 分开报告业务水平、同seed跨world sample SD、跨seed sample SD及同world动作repeat sample SD。repeat变异同时给出每repeat重选最弱车队，以及固定scene-first最弱车队后的SD（极值并列时等权平均这些车队的SD）。单repeat SD未定义，不能填0或和其他SD相加。
- 六training seeds是推断重复单位；world、repeat、episode、slot不扩大独立n。描述性95%t区间df=5、t*=2.570581835636314；pilot仅2seed不使用六seed区间。technical_status与claim_support分离，不按screen_success筛选，不自动标显著性或将跨零称为等效。
- pilot=60 policy-world-repeat；main=2340，30个新world在服务器按历史排除固定；gap-transfer新增Algorithm 2=144，复用既有MAPPO间距24cell/144policy-world另列。gap-transfer的213–218为旧开发world，属于观察性扩展，不是新世界确认。
- 全部raw NPZ/checkpoint/完整轨迹留服务器；精简证据含来源、完成记录、累计量CSV、各级摘要、verification、报告及PNG。没有修改历史provenance或正式旧表。

本地能确认合成数据、薄适配接线、统计归约与脚本语法；真实checkpoint加载、实际GPU rollout、Slurm资源和正式产物验收需服务器记录支持。
