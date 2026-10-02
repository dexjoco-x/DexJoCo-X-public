# Benchmark Protocol

每种方法在 7 种手 × 6 个任务上评测。每个 hand–task 单元执行 3 次独立评估，每次 50 个 rollout，总计 150 个 rollout；每个七手配置共 6,300 个 rollout。

场景种子为 `10_000_000 + task_index × 1000 + episode_index`，其中 `episode_index=0…149`。策略种子为 17…166。Being-H0.5 输出 30×80 动作块，每次执行前 8 步，再重新规划。成功由环境内统一的几何、接触和时序条件判定。

Native、FAAS 和 DexLatent 使用相同观测、任务指令、臂控制、场景与执行时序，只改变手部动作表示。模型服务必须返回完整、已反归一化且尚未解码的动作块。评估器保存 repeat、场景种子、策略种子、checkpoint 元数据、协议哈希和逐条成功字段。

结果只有在 150 个 rollout 完整、协议哈希一致、无重复 seed 且 checkpoint 身份可核验时，才可进入论文表格。
