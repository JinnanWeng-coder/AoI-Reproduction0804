# HPC CAM96k 方法事实（供写作核对，不是结果结论）

- 六种学习条件均250k环境步、六训练seed；另有无训练Uniform Random两个动作stream。
- 旧Shared MAPPO-Combined保持联合观测、P输出总回报价值，独立于Actor；未重命名为NoTDec。
- 新Shared MAPPO-NoTDec：一个共享HybridActor；一个联合观测输入、标量输出全局V；
  五个独立、只输入各自局部观测的本地V，隐藏层沿用原单个本地Critic。
- 本地回报是原task1+task2，不平均、不再加全局回报；全局及本地分别算GAE，
  A_i=global_actor_weight*A_global+A_local,i，之后原逐agent优势归一化与PPO不变。
- 原网络接口直接输出原始回报尺度价值；value norm作用于损失中的预测/目标。
  GAE读取这些raw值，不错误地再施加一次反归一化；全局和本地保留独立运行归一化状态。
- 内部复用三stream rollout容器：task1槽装local-sum，task2槽恒零；无task2网络、Adam或norm。
  完整恢复保存这一容器和全部真正存在的模块，诊断输出改名local，不把零槽冒称第二任务。
- Actor与Critic不共享网络，不照搬Parvini实现中detach全局loss或乘2的细节。
  NoTDec与TDec网络数量/参数量不同，实际记录parameter_counts和wall time，未做参数量匹配实验。
- Modified MADDPG依旧TDec，共享/独立两种Actor；固定探索sigma0.3，target smoothing0.2/clip0.5，
  synchronous_joint及原学习器不变。冻结noise0不等于关闭训练探索/target smoothing。
- 对Parvini两目录的本地Q结构核查确认：有TDec为两个本地任务Q，无TDec为单本地Q；
  新条件只是MAPPO中取消本地任务拆分的对应检验，不声称复现MADDPG单Q算法。
- 从每方法六个best100仅按rolling100训练回报选一份，再在30个已用探索world冻结。
  这是每方法一个选定策略的部署表现；30world不是30个训练重复，不能写六seed均值/SD。
- 全部冻结CPU，同CPU协议下串并行验证；不声称CPU与历史Windows/CUDA随机动作逐数组相同。
- Uniform沿用PCG64 SeedSequence([stream,world,0x52414E44,1])，依次RB/模式/功率fraction独立均匀；
  功率1–30dBm均匀，不是mW均匀，不学习、不选模、不使用神经网络。
- 每world冷启一次，5预热+100计分周期×100槽，CAM96000 bit。截止成功看末槽CAM，
  平均功率先转线性mW。失败剩余bit只对失败条件计算，无失败为NA，不补0。
- 学习180与Random60单元分别验收，合计240；不混用训练环境步与计分车队时隙。
