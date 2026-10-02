# 七手六任务评测接口

`runtime.py` 使用冻结的 42 个模型、控制器、成功规则与相机配置。汉诺塔为单右臂任务，XHand 的修正安装方向与当前数据版本保持一致。公开源码版已经包含必要模型资产；完整示范数据和权重另行管理。

```bash
pip install -e './simulator[benchmark]'
python evaluation/evaluate.py --help
python evaluation/rpc_smoke.py --help
```

策略只接收图像、机器人状态、掩码和任务指令，不接收物体真值。每个手–任务组合执行三次独立 50-rollout 评估；结果记录重复编号、场景/策略种子、checkpoint 信息、协议哈希和成功字段。策略服务的动作块需已反归一化，仍保持指定的动作表征，由运行时解码并执行。

## 80 维排列

| 排列 | 动作 | 状态 |
|---|---|---|
| `grouped80_v1` | 右臂 0:6、左臂 6:12、右手 12:44、左手 44:76、保留 76:80 | 右腕位置/四元数 0:7、左腕 7:14、右手 14:46、左手 46:78、保留 78:80 |
| `arm40_v1` | 每侧一个 40 维块；腕部 0:6、手部 6:38、保留 38:40 | 每侧一个 40 维块；腕部位置/rotvec 0:6、手部 6:38、保留 38:40 |

两种排列均支持 Native、FAAS、DexLatent，但 checkpoint 和归一化统计量必须匹配排列与 codec，不能直接互换。单臂任务关闭缺失侧的状态、动作和图像掩码。

## 评测

使用自有 π0.5 checkpoint 的独立接入示例见 [π0.5 策略服务](../examples/pi05/README.md)。示例不依赖本项目的训练实现，支持单臂 32 维及 Ego-π 左右交替排列，负责图像映射、掩码、训练统计量、显式噪声与动作反归一化。

策略服务应实现项目协议，默认返回 30×80 动作块。评估器默认执行每块前 8 步，以环境成功条件首次成立作为结束，不添加额外成功保持计数；限制裁剪会写入报告。

Ego-π 的 50 个左右交替 token 对应 25 个物理时间步，使用 `--action-horizon 25`；服务声明和评测参数必须一致。输出记录实际动作块及执行长度，不通过复制或补动作把 25 步伪装为 30 步。此项接口验收不改写论文历史结果。自有 π0.5 权重的策略服务见 [推理示例](../examples/pi05/README.md)。

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 --layout grouped80_v1 --representation native --hand xhand --task all --episodes 150 --video --output ./results/xhand_native
```

模型和编码器权重不公开。Native/FAAS 可使用按接口训练的自有策略；DexLatent 还需匹配的本地编码器。当前默认编码器哈希见 codec 配置，可通过 `BENCHMARK7_DEXLATENT_CHECKPOINT` 指定本地文件。论文使用的 sided-v3 codec 与后续工程配置按版本区分，复现时需保持一致。

## 安装检查

```bash
python tools/portability_smoke.py
python tools/representation_smoke.py --representations native faas
python tools/scope_smoke.py
```

这些检查验证环境、动作表征、数据导出及确定性回放；hold-action 诊断结果不作为任务成功率。正式实验结果需使用对应策略权重和正式评估种子。

通信依赖由 `benchmark` 安装选项提供。DexLatent 额外安装 `./simulator[dexlatent]` 并准备匹配的本地编码器；OpenPI/Ego-π/Being-H 策略服务与训练框架仍按各自上游环境安装。

无显示器环境可在执行上述命令前设置 `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`（需要 OSMesa），以避免使用未配置的 EGL 驱动。

公开版通信 smoke 默认只检查 Native/FAAS 的两种排列；加入 `--representations native faas dexlatent` 前需先准备本地编码器权重。

## 短流程验收

使用自有 checkpoint 启动策略服务后，评测器可用 `--diagnostic --max-steps 16` 执行短验收；仅诊断模式允许缩短轨迹。诊断输出标记 `diagnostic_only: true`，诊断策略不能用于生成正式评测报告。数据与转换结果检查见 `tools/dataset_smoke.py`。

环境版本记录使用随源码提供的 `evaluation/models.json` 的 SHA-256，模型清单包含全部 42 个模型 XML 与控制定义的冻结哈希。

## 其他策略

`serve80.handler_for` 可以包装任何实现 `infer(observation, noise=...)` 的使用者策略，返回有限、已反归一化的 H×80 动作块。它不要求模型本身有 80 维动作头；内部 token 到物理动作的映射由策略适配器负责。也可通过安装的工厂函数加载自己的策略：

```bash
python evaluation/serve80.py --factory my_policy:load --checkpoint /path/to/checkpoint \
  --layout arm40_v1 --representation native --action-horizon 30
```

`my_policy.load(checkpoint_path)` 返回配置好的策略对象，模型加载与归一化规则由使用者实现。π0.5 示例提供了具体实现。
