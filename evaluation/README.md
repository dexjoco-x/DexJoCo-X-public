# 策略接入与评测

评测器连接策略服务，将图像、机器人状态、掩码和任务指令发送给策略，接收动作块并在仿真中执行。

## 运行评测

先启动 [π₀.₅ / Ego-π 策略服务](../examples/pi05/README.md)，再执行：

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/allegro_hammer
```

`--task all` 评测该手型的六个任务，`--video` 保存录像。每个手–任务组合执行三次独立评估，每次 50 个 rollout；结果记录逐条成功、重复编号、场景和策略种子、checkpoint 与协议信息。

成功由环境的几何、接触和时序条件判定。策略观察不包含物体真值。场景种子为 `10_000_000 + task_index × 1000 + episode_index`，`episode_index` 为 0…149；策略种子为 17…166。

## 状态与动作

策略服务返回物理单位下、已反归一化的 `H × 80` 动作块。评测器每次执行前 8 步，再请求下一块；`--action-horizon` 与服务的预测步数一致。Ego-π 的 50 个交替 token 对应 25 个物理时间步。

| 排列 | 动作 | 状态 |
|---|---|---|
| `arm40_v1` | 右侧 0:40，左侧 40:80；每侧腕部 0:6、手部 6:38、保留 38:40 | 每侧腕部位置/rotvec 0:6、手部 6:38、保留 38:40 |
| `grouped80_v1` | 右臂 0:6、左臂 6:12、右手 12:44、左手 44:76、保留 76:80 | 右腕位置/四元数 0:7、左腕 7:14、右手 14:46、左手 46:78、保留 78:80 |

腕部位置以米为单位，旋转以弧度为单位。单臂任务通过掩码关闭缺失侧，手通道顺序和范围见 [任务合约](contracts)。

`--representation` 支持三种手部动作表示：`native` 为手自身的控制坐标，`faas` 为功能对齐槽位，`dexlatent` 为学习得到的潜变量。排列、表示和归一化统计量应与策略一致。DexLatent 额外安装 `./simulator[dexlatent]`，并通过 `BENCHMARK7_DEXLATENT_CHECKPOINT` 提供匹配的编码器。

## 接入其他策略

实现 `infer(observation, noise=...)`，返回有限、已反归一化的 `H × 80` 数组。加载函数 `my_policy.load(checkpoint_path)` 返回该策略对象：

```bash
python evaluation/serve80.py --factory my_policy:load \
  --checkpoint /path/to/checkpoint \
  --layout arm40_v1 --representation native --action-horizon 30
```

适配器负责将模型自身的动作格式转换为评测接口，模型动作头无需固定为 80 维。

## 短流程调试

使用 `--episodes 1 --diagnostic --max-steps 16` 检查服务和仿真连接。诊断输出带 `diagnostic_only: true`，与完整的成功率评测分开记录。

环境版本与模型哈希记录在 `models.json`，数据兼容关系见 [版本说明](../docs/VERSIONING.md)。
